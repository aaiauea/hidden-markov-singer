"""Small MLPG vs MLPG + experimental GV benchmark (synthetic statistics).

Uses the same deterministic statistics as ``tools/bench_mlpg.py``. The GV target
is set to ``target_factor * GV(MLPG output)`` for every static feature; this is
a controlled oversmoothing example, *not* a measured training/speech-quality
result. Timings are medians in ms; memory is the separately measured peak of
Python + NumPy allocations observed by tracemalloc (not a resident-set size).

    python tools/bench_gv.py --frames 1000 --dim 30 --iterations 20
    python tools/bench_gv.py --frames 256 --dim 16 --repeat 3 --json out.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_mlpg import make_statistics                         # noqa: E402
from hms.core.generation import mlpg                            # noqa: E402
from hms.core.gv import optimize_global_variance, trajectory_variance  # noqa: E402


def peak_memory(operation) -> int:
    """Approximate peak allocated bytes, measured separately from timings."""
    gc.collect()
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--dynamic-streams", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--weight", type=float, default=3.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--target-factor", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.frames < 2 or args.dim < 1 or args.window < 1 \
            or args.repeat < 1 or args.warmup < 0 \
            or not np.isfinite(args.target_factor) or args.target_factor < 0:
        parser.error("frames >= 2, dim >= 1, window >= 1, repeat >= 1, "
                     "warmup >= 0 and finite non-negative target-factor required")

    sizes = (args.dim,) * (args.dynamic_streams + 1)
    means, variances = make_statistics(args.frames, args.dim, len(sizes))
    reference = mlpg(means, variances, sizes, window=args.window)
    target = args.target_factor * trajectory_variance(reference)

    def baseline():
        return mlpg(means, variances, sizes, window=args.window)

    def with_gv():
        initial = baseline()
        return optimize_global_variance(
            initial, variances, sizes, target, window=args.window,
            weight=args.weight, iterations=args.iterations)

    def measure(operation):
        samples = []
        for run in range(args.warmup + args.repeat):
            started = time.perf_counter()
            operation()
            elapsed = 1e3 * (time.perf_counter() - started)
            if run >= args.warmup:
                samples.append(elapsed)
        return statistics.median(samples)

    baseline_ms = measure(baseline)
    with_gv_ms = measure(with_gv)
    after, steps_used = optimize_global_variance(
        reference, variances, sizes, target, window=args.window,
        weight=args.weight, iterations=args.iterations, return_steps=True)
    before_gv = trajectory_variance(reference)
    after_gv = trajectory_variance(after)
    memory_mlpg = peak_memory(baseline)
    memory_gv_only = peak_memory(lambda: optimize_global_variance(
        reference, variances, sizes, target, window=args.window,
        weight=args.weight, iterations=args.iterations))
    memory_with_gv = peak_memory(with_gv)
    result = {
        "frames": args.frames, "static_dim": args.dim,
        "dynamic_streams": args.dynamic_streams, "gv_iterations_max": args.iterations,
        "gv_iterations_used": steps_used,
        "gv_weight": args.weight, "target_factor": args.target_factor,
        "mlpg_median_ms": baseline_ms, "mlpg_gv_median_ms": with_gv_ms,
        "variance_before_mean": float(before_gv.mean()),
        "variance_after_mean": float(after_gv.mean()),
        "variance_target_mean": float(target.mean()),
        "max_abs_change": float(np.max(np.abs(after - reference))),
        "mlpg_peak_bytes": memory_mlpg,
        "gv_only_peak_bytes": memory_gv_only,
        "mlpg_gv_peak_bytes": memory_with_gv,
        "peak_delta_bytes": memory_with_gv - memory_mlpg,
    }
    print(f"T={args.frames}, static_dim={args.dim}, "
          f"dynamic_streams={args.dynamic_streams}, "
          f"NumPy {np.__version__}, Python {sys.version.split()[0]}")
    print(f"GV: iterations={steps_used}/{args.iterations} maximum, "
          f"weight={args.weight:g}, synthetic target factor={args.target_factor:g}")
    print(f"baseline MLPG median:     {baseline_ms:.3f} ms")
    print(f"MLPG + GV median:         {with_gv_ms:.3f} ms")
    print(f"mean variance (static features): "
          f"before={before_gv.mean():.5g}, after={after_gv.mean():.5g}, "
          f"target={target.mean():.5g}")
    print(f"maximum absolute trajectory change: {result['max_abs_change']:.5g}")
    print(f"peak allocated: MLPG={memory_mlpg / 1048576:.2f} MiB, "
          f"GV alone={memory_gv_only / 1048576:.2f} MiB, "
          f"MLPG + GV={memory_with_gv / 1048576:.2f} MiB "
          f"(difference={result['peak_delta_bytes'] / 1048576:+.2f} MiB)")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
