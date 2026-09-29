"""Benchmark harness for MLPG (assembly / Cholesky / solve / total).

The stage split is measured by wrapping ``hms.core.generation.banded_cholesky``
and ``hms.core.generation.banded_solve`` with timers; assembly is the remainder
of the ``mlpg`` wall time.  That decomposition matches the numbers reported for
earlier performance work (assembly / Cholesky / solve / total), so new results
stay comparable with them.

Usage
-----
    python tools/bench_mlpg.py                 # default matrix of cases
    python tools/bench_mlpg.py --repeat 9
    python tools/bench_mlpg.py --cases 2400:36:2
    python tools/bench_mlpg.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hms.core import generation                                    # noqa: E402
from hms.core.generation import mlpg, window_bandwidth             # noqa: E402

#: Default (T, dim, dynamic_streams) cases, the shapes used for the reported
#: MLPG timings.  "dynamic_streams" counts the delta streams only: 1 =
#: static+delta (S=2, bandwidth 4), 2 = static+delta+delta-delta (S=3,
#: bandwidth 8).
DEFAULT_CASES = [
    (256, 30, 1),
    (1000, 30, 1),
    (1000, 30, 2),
    (2400, 36, 1),
    (2400, 36, 2),
]


def make_statistics(n_frames: int, dim: int, n_streams: int, seed: int = 0):
    """Statistics with the shape and scale a trained model produces.

    Static frames come first, then one block per delta stream, exactly the
    row order `hms.core.features.add_dynamic_features` produces.
    """
    rng = np.random.default_rng(seed)
    means = np.cumsum(rng.normal(scale=0.05, size=(n_frames * n_streams, dim)),
                      axis=0)
    # static variances ~0.2, delta variances much smaller, both log-normal
    scales = np.where(np.arange(n_frames * n_streams)[:, None] < n_frames,
                      -1.6, -4.5)
    variances = np.exp(scales + rng.normal(scale=0.7, size=means.shape))
    return means, variances


def measure(means, variances, stream_sizes, window, repeat, warmup):
    """Return per-stage median times in milliseconds for one case."""
    real_chol = generation.banded_cholesky
    real_solve = generation.banded_solve
    timings: dict[str, list[float]] = {"cholesky": [], "solve": [], "total": []}

    for run in range(warmup + repeat):
        record = {"cholesky": 0.0, "solve": 0.0}

        def timed_chol(a_band, bandwidth, _record=record):
            start = time.perf_counter()
            out = real_chol(a_band, bandwidth)
            _record["cholesky"] += time.perf_counter() - start
            return out

        def timed_solve(lower, b, bandwidth, _record=record):
            start = time.perf_counter()
            out = real_solve(lower, b, bandwidth)
            _record["solve"] += time.perf_counter() - start
            return out

        generation.banded_cholesky = timed_chol
        generation.banded_solve = timed_solve
        try:
            start = time.perf_counter()
            mlpg(means, variances, stream_sizes, window=window)
            total = time.perf_counter() - start
        finally:
            generation.banded_cholesky = real_chol
            generation.banded_solve = real_solve

        if run >= warmup:
            timings["cholesky"].append(record["cholesky"] * 1e3)
            timings["solve"].append(record["solve"] * 1e3)
            timings["total"].append(total * 1e3)

    best = {key: min(values) for key, values in timings.items()}
    median = {key: statistics.median(values) for key, values in timings.items()}
    best["assembly"] = best["total"] - best["cholesky"] - best["solve"]
    median["assembly"] = (median["total"] - median["cholesky"] - median["solve"])
    return best, median


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=7,
                        help="timed repetitions per case (best/median reported)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--cases", type=str, default=None,
                        help="comma separated T:dim:dynamic_streams triples")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the raw numbers to this JSON file")
    args = parser.parse_args(argv)

    cases = DEFAULT_CASES
    if args.cases:
        cases = [tuple(int(v) for v in case.split(":"))
                 for case in args.cases.split(",")]

    rows = []
    print(f"numpy {np.__version__}   python {sys.version.split()[0]}   "
          f"window={args.window}   repeat={args.repeat}  (best of {args.repeat})")
    header = (f"{'T':>5} {'dim':>4} {'dyn':>3} {'S':>3} {'bw':>3} "
              f"{'assembly':>9} {'cholesky':>9} {'solve':>9} {'total':>9}")
    print(header)
    print("-" * len(header))
    for n_frames, dim, dyn_streams in cases:
        n_streams = 1 + dyn_streams
        stream_sizes = (dim,) * n_streams
        means, variances = make_statistics(n_frames, dim, n_streams)
        best, median = measure(means, variances, stream_sizes, args.window,
                               args.repeat, args.warmup)
        row = {
            "n_frames": n_frames, "dim": dim, "dynamic_streams": dyn_streams,
            "n_streams": n_streams,
            "bandwidth": window_bandwidth(stream_sizes, args.window),
            "best": best, "median": median,
        }
        rows.append(row)
        print(f"{n_frames:>5} {dim:>4} {dyn_streams:>3} {n_streams:>3} "
              f"{row['bandwidth']:>3} "
              f"{best['assembly']:>9.3f} {best['cholesky']:>9.3f} "
              f"{best['solve']:>9.3f} {best['total']:>9.3f}")

    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
