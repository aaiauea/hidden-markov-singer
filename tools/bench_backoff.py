"""Benchmark ``Trainer.build_backoff()`` on a reproducible training corpus.

The corpus is either the deterministic ``hms demo`` corpus (generated into a
temporary directory) or a directory with ``labels.tsv`` and ``wav/`` (for
example ``examples/minimal`` after ``generate_corpus.py``).

One full training run is performed first.  The arguments that ``build_backoff``
receives in that run are then captured, and ``build_backoff`` is timed on those
exact arguments.  Isolating the call this way keeps the (unchanged) WORLD
analysis out of the measurement.  Counters (alignment calls and frames, GMM fits
and fitted frames) come from a separate instrumented call, and ``--memory``
reports Python allocation peaks, which slow allocation-heavy code, so it is also
a separate pass.  ``--hash`` prints a digest of the trained backoff arrays:
equal digests across two checkouts mean bit-identical models on this corpus.

Examples::

    python tools/bench_backoff.py --corpus demo --repeats 5
    python tools/bench_backoff.py --corpus demo --memory --hash
    python tools/bench_backoff.py --corpus examples/minimal/out --fs 22050
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from hms.config import (load_parameters, load_phoneme_set,
                        training_config_from_parameters)
from hms.core import gmm as gmm_module
from hms.core import hmm as hmm_module
from hms.core import trainer as trainer_module


def _prepare_corpus(args, workdir: Path):
    if args.corpus == "demo":
        from hms.data.demo_singer import make_dataset
        info = make_dataset(workdir / "demo", fs=args.fs, label_jitter_ms=0.0,
                            seed=args.seed)
        return Path(info["labels"]), Path(info["wav_dir"])
    root = Path(args.corpus)
    return root / "labels.tsv", root / "wav"


def _build_trainer(args, labels: Path, wav_dir: Path):
    config = training_config_from_parameters(
        load_parameters(args.parameters or None))
    config.label_file = str(labels)
    config.wav_dir = str(wav_dir)
    config.fs = args.fs
    config.time_unit = "seconds"
    config.seed = args.seed
    phonemes = load_phoneme_set(args.phonemes or None)
    return trainer_module.Trainer(config, phonemes, log=lambda m: None)


def _digest(model_backoff) -> str:
    digest = hashlib.sha256()
    for klass in sorted(model_backoff):
        arrays = model_backoff[klass].to_arrays()
        for key in sorted(arrays):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()[:16]


def _instrument(counters: dict, timers: dict):
    """Wrap alignment and GMM-fit entry points; returns an undo callable."""
    originals = []

    def wrap(owner, name, key, frames_of):
        raw = owner.__dict__[name]
        is_classmethod = isinstance(raw, classmethod)
        bound = getattr(owner, name)

        def call(*args, **kwargs):
            if is_classmethod:
                args = args[1:]
            counters[key + "_calls"] = counters.get(key + "_calls", 0) + 1
            counters[key + "_frames"] = (counters.get(key + "_frames", 0)
                                         + int(frames_of(*args)))
            start = time.perf_counter()
            try:
                return bound(*args, **kwargs)
            finally:
                timers[key] = timers.get(key, 0.0) + time.perf_counter() - start

        setattr(owner, name, classmethod(call) if is_classmethod else call)
        originals.append((owner, name, raw))

    wrap(hmm_module.LeftToRightHMM, "viterbi", "alignment",
         lambda self, X, *a: len(X))
    wrap(gmm_module.DiagGMM, "fit", "gmm_fit", lambda X, *a, **k: len(X))

    def undo():
        for owner, name, raw in originals:
            setattr(owner, name, raw)
    return undo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="demo",
                        help="'demo' or a directory with labels.tsv and wav/")
    parser.add_argument("--fs", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--phonemes", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--memory", action="store_true",
                        help="also report tracemalloc peak of one build_backoff "
                             "call and the process peak RSS")
    parser.add_argument("--hash", action="store_true",
                        help="print a digest of the trained backoff arrays")
    parser.add_argument("--json", default=None, help="write results here")
    args = parser.parse_args(argv)

    result = {"python": platform.python_version(),
              "numpy": np.__version__, "corpus": args.corpus, "fs": args.fs}
    with tempfile.TemporaryDirectory(prefix="hms-bench-") as tmp:
        labels, wav_dir = _prepare_corpus(args, Path(tmp))
        trainer = _build_trainer(args, labels, wav_dir)

        captured = {}
        original_build = trainer_module.Trainer.build_backoff

        def capture(self, features, voiced):
            captured["args"] = (features, voiced)
            return original_build(self, features, voiced)

        trainer_module.Trainer.build_backoff = capture
        start = time.perf_counter()
        try:
            model = trainer.train()
        finally:
            trainer_module.Trainer.build_backoff = original_build
        result["train_wall_s"] = round(time.perf_counter() - start, 3)
        features, voiced = captured["args"]
        result["phone_frames"] = int(sum(len(s) for seqs in features.values()
                                         for s in seqs))
        result["backoff_classes"] = sorted(model.backoff)
        if args.hash:
            result["backoff_sha256_16"] = _digest(model.backoff)

        # timed repeats (no instrumentation)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            trainer.build_backoff(features, voiced)
            times.append(time.perf_counter() - start)
        result["build_backoff_s"] = {"median": round(statistics.median(times), 4),
                                     "min": round(min(times), 4),
                                     "max": round(max(times), 4),
                                     "repeats": args.repeats}

        # counters from one instrumented call
        counters, timers = {}, {}
        undo = _instrument(counters, timers)
        try:
            trainer.build_backoff(features, voiced)
        finally:
            undo()
        result["counters"] = counters
        result["instrumented_s"] = {k: round(v, 4) for k, v in timers.items()}

        if args.memory:
            tracemalloc.start()
            trainer.build_backoff(features, voiced)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result["build_backoff_tracemalloc_peak_MB"] = round(peak / 2**20, 2)
            result["process_maxrss_MB"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
