"""Micro-benchmark for ``LeftToRightHMM.viterbi``.

The workload mirrors the calls made by ``hms demo`` training (measured with a
wrapper around ``viterbi`` on the demo corpus: 1,251 calls, 57,798 frames). The
synthetic workload built here has the same call count and state histogram and
56,459 frames, because the lengths are sampled rather than copied:

* state counts drawn from the demo model: 5 (495 calls), 3 (198), 2 (144), 1 (414)
* 2 GMM components for the 5-state HMMs and 1 otherwise
* feature dimension 72 (the demo model's mel-cepstral + delta layout)
* utterance lengths T sampled from a log-normal with median 30 frames, clipped
  to [14, 180] (demo: median 30, p90 96, max 180, min 14)

Everything is seeded, so two checkouts running this script see identical inputs.
The script only uses the public API, so it can be run unchanged against an
older checkout (``PYTHONPATH=/path/to/old/tree``).

It reports the median and range of several timed repetitions (after a
warm-up), a hash of every returned path and score (equal hashes across two
checkouts mean bit-identical outputs on this workload), and optionally the
peak traced memory of one pass (run separately, because tracing slows
allocation-heavy code).

Examples::

    python tools/bench_viterbi.py                 # timings + output hash
    python tools/bench_viterbi.py --memory        # peak traced memory
    python tools/bench_viterbi.py --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from hms.core.gmm import DiagGMM
from hms.core.hmm import HMMState, LeftToRightHMM

DIM = 72
STATE_HISTOGRAM = {5: 495, 3: 198, 2: 144, 1: 414}   # demo calls per n_states
COMPONENTS = {5: 2, 3: 1, 2: 1, 1: 1}


def build_workload(seed: int = 0, n_calls: int = 1251):
    """Return ``(hmms_by_n, [(n_states, X), ...])`` built deterministically."""
    rng = np.random.default_rng(seed)
    hmms = {}
    for n in sorted(STATE_HISTOGRAM):
        hmm = LeftToRightHMM(n_states=n, allow_skip=False)
        for i in range(n):
            k = COMPONENTS[n]
            weights = rng.dirichlet(np.ones(k))
            means = rng.normal(0.0, 1.0, size=(k, DIM))
            variances = rng.uniform(0.2, 2.0, size=(k, DIM))
            hmm.states[i] = HMMState(gmm=DiagGMM(weights, means, variances, "diag"))
        hmm.self_loops = rng.uniform(0.3, 0.9, size=n)
        hmms[n] = hmm

    labels = []
    for n, count in STATE_HISTOGRAM.items():
        labels += [n] * count
    labels = np.array(labels)[rng.permutation(len(labels))][:n_calls]
    lengths = np.clip(np.exp(np.log(30.0) + 0.9 * rng.standard_normal(len(labels))),
                      14, 180).astype(int)
    calls = []
    for n, t in zip(labels, lengths):
        calls.append((int(n), rng.normal(0.0, 1.0, size=(int(t), DIM))))
    return hmms, calls


def run_once(hmms, calls):
    return [hmms[n].viterbi(X) for n, X in calls]


def output_digest(results) -> str:
    h = hashlib.sha256()
    for path, score in results:
        h.update(np.asarray(path, dtype=np.int64).tobytes())
        h.update(np.float64(score).tobytes())
    return h.hexdigest()


def environment() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=os.path.dirname(__file__) or ".",
                             check=False).stdout.strip()
    except OSError:
        sha = ""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "hms_module": sys.modules["hms.core.hmm"].__file__,
        "git_head": sha,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory", action="store_true",
                        help="also report peak traced memory of one pass")
    parser.add_argument("--json", default=None, help="write results as JSON")
    args = parser.parse_args(argv)

    hmms, calls = build_workload(args.seed)
    frames = sum(len(X) for _, X in calls)
    for _ in range(args.warmup):
        run_once(hmms, calls)

    # Pure viterbi time (emission scores included, as in production).
    times = []
    results = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        results = run_once(hmms, calls)
        times.append(time.perf_counter() - t0)

    # Emission-only time, to show how much of viterbi is Gaussian scoring.
    emission_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for n, X in calls:
            hmms[n].emission_log_likelihood(X)
        emission_times.append(time.perf_counter() - t0)

    report = {
        "calls": len(calls),
        "frames": frames,
        "viterbi_seconds_median": statistics.median(times),
        "viterbi_seconds_min": min(times),
        "viterbi_seconds_max": max(times),
        "emission_seconds_median": statistics.median(emission_times),
        "output_sha256": output_digest(results),
        "repeats": args.repeats,
        "environment": environment(),
    }

    if args.memory:
        tracemalloc.start()
        run_once(hmms, calls)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["peak_traced_bytes"] = int(peak)

    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
