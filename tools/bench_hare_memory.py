"""Benchmark HARE memory insertion and NPZ serialization on synthetic frames.

No vocoding, feature extraction, alignment or synthesis is timed.  The
baseline is an ID -> already-built observation dict, not an equivalent memory
(no width/uniqueness checks), plus a compressed numeric-only NPZ of the raw
feature slices (no IDs, phonemes, states, contexts or feature spec).

    python tools/bench_hare_memory.py --frames 10000 --repeat 5
    python tools/bench_hare_memory.py --frames 1000 --repeat 3 --json result.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms.core.features import FeatureSpec  # noqa: E402
from hms.hare import AcousticMemory, AcousticObservation  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10000)
    parser.add_argument("--n-mcep", type=int, default=30)
    parser.add_argument("--n-band", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.n_mcep < 2 or args.n_band < 2 \
            or args.repeat < 1 or args.warmup < 0:
        parser.error("frames >= 1, n-mcep/n-band >= 2, repeat >= 1 and warmup >= 0 required")

    spec = FeatureSpec(n_mcep=args.n_mcep, n_band=args.n_band)
    rng = np.random.default_rng(args.seed)
    # Already-encoded FeatureSpec layout: F0 + mcep + aperiodicity.  Values are
    # synthetic, so they say nothing about synthesis quality or a real corpus.
    static = np.empty((args.frames, spec.static_dim), dtype=np.float64)
    static[:, :1 + spec.n_mcep] = rng.normal(size=(args.frames, 1 + spec.n_mcep))
    static[:, 1 + spec.n_mcep:] = rng.uniform(size=(args.frames, spec.n_band))
    phones = ("a", "i", "n", "sil")

    def observations():
        for i, frame in enumerate(static):
            yield AcousticObservation.from_static(
                f"take:{i}", phones[i % len(phones)], i % 3, frame, spec,
                {"pre_phone": phones[(i - 1) % len(phones)],
                 "future_phone": phones[(i + 1) % len(phones)]})

    prebuilt = list(observations())

    def dict_index():
        return {o.observation_id: o for o in prebuilt}

    def add_one_by_one():
        memory = AcousticMemory(spec)
        for item in prebuilt:
            memory.add(item)
        return memory

    def add_batch():
        memory = AcousticMemory(spec)
        memory.add_batch(prebuilt)
        return memory

    def construct_and_add():
        memory = AcousticMemory(spec)
        memory.add_batch(observations())
        return memory

    def median_ms(operation):
        samples = []
        for i in range(args.warmup + args.repeat):
            gc.collect()
            start = time.perf_counter()
            result = operation()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= args.warmup:
                samples.append(elapsed)
            del result
        return statistics.median(samples)

    def peak_bytes(operation):
        # Separate from timing: tracing allocations changes performance.
        gc.collect()
        tracemalloc.start()
        try:
            result = operation()
            _, peak = tracemalloc.get_traced_memory()
            del result
            return peak
        finally:
            tracemalloc.stop()

    results = {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "frames": args.frames,
        "n_mcep": spec.n_mcep, "n_band": spec.n_band,
        "seed": args.seed, "warmup": args.warmup, "repeat": args.repeat,
        "dict_index_median_ms": median_ms(dict_index),
        "insert_single_prebuilt_median_ms": median_ms(add_one_by_one),
        "insert_batch_prebuilt_median_ms": median_ms(add_batch),
        "construct_and_insert_batch_median_ms": median_ms(construct_and_add),
        "construct_and_insert_batch_peak_bytes": peak_bytes(construct_and_add),
        "raw_feature_bytes": int(static[:, 1:].nbytes),
    }
    memory = add_batch()
    with tempfile.TemporaryDirectory(prefix="hms-hare-bench-") as tmp:
        path = Path(tmp) / "memory.npz"
        raw_path = Path(tmp) / "numeric-only.npz"

        def save_memory():
            return memory.save(path)

        def save_raw():
            np.savez_compressed(raw_path, static=static[:, 1:])

        def load_raw():
            with np.load(raw_path, allow_pickle=False) as data:
                return data["static"]

        results.update({
            "save_numeric_only_median_ms": median_ms(save_raw),
            "load_numeric_only_median_ms": median_ms(load_raw),
            "save_memory_median_ms": median_ms(save_memory),
            "load_memory_median_ms": median_ms(lambda: AcousticMemory.load(path)),
            "save_memory_peak_bytes": peak_bytes(save_memory),
            "numeric_only_npz_bytes": raw_path.stat().st_size,
            "memory_npz_bytes": path.stat().st_size,
        })
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
