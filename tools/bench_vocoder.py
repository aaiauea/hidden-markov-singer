"""Benchmark the pure-numpy vocoder backends: MLSA vs the builtin fallback.

Measures, for a matrix of utterance lengths:

* wall time and **RTF** (render time / audio duration), best of ``--repeat``;
* **peak Python heap** used by the synthesis call (``tracemalloc``), which
  covers the numpy buffers the backend allocates;
* a sanity RMS/peak so a "fast" backend that renders silence is visible.

Both backends get the *same* parameters, analysed once from the demo corpus
with whichever WORLD backend is available (``native`` by default, falling back
to the builtin estimator), so the comparison is backend-for-backend and not
analysis-for-analysis.

Usage
-----
    python tools/bench_vocoder.py
    python tools/bench_vocoder.py --lengths 1 5 20 60 --repeat 5
    python tools/bench_vocoder.py --vocoders mlsa builtin native
    python tools/bench_vocoder.py --fs 44100 --json bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hms.core.features import AcousticFrameSequence          # noqa: E402
from hms.data import wavio                                    # noqa: E402
from hms.data.demo_singer import make_dataset                 # noqa: E402
from hms.vocoder import available_backends, get_vocoder       # noqa: E402

#: Utterance lengths in seconds, unless overridden on the command line.
DEFAULT_LENGTHS = (1.0, 5.0, 20.0, 40.0)

#: Backends compared by default (both are always available).
DEFAULT_VOCODERS = ("mlsa", "builtin")


def build_parameters(fs: int, duration: float, frame_period: float,
                     analysis: str, source_dir: Path) -> AcousticFrameSequence:
    """Analyse the demo corpus once and tile it to ``duration`` seconds."""
    info = make_dataset(source_dir, fs=fs, seed=3)
    path = sorted(Path(info["wav_dir"]).glob("*.wav"))[0]
    signal, _ = wavio.read_wav(path)
    analyzer = get_vocoder(analysis)
    base = analyzer.analyze_to_sequence(signal, fs, frame_period=frame_period)
    n_frames = int(round(duration * 1000.0 / frame_period))
    repeat = int(np.ceil(n_frames / len(base)))
    return AcousticFrameSequence(
        f0=np.tile(base.f0, repeat)[:n_frames],
        sp=np.tile(base.sp, (repeat, 1))[:n_frames],
        ap=np.tile(base.ap, (repeat, 1))[:n_frames],
        frame_period=frame_period, fs=fs, fft_size=base.fft_size)


def measure(vocoder, params: AcousticFrameSequence, repeat: int) -> dict:
    """Best-of-``repeat`` wall time plus the memory peak of one call.

    Time and memory are measured in separate passes: ``tracemalloc`` tracks
    every allocation, so leaving it on during the timed calls would penalise
    backends that allocate many small arrays (the builtin fallback) -- a real
    overhead, but not the synthesis cost this benchmark is comparing.
    """
    expected = int(len(params) * params.frame_period / 1000.0 * params.fs)
    duration = expected / params.fs
    times = []
    audio = None
    for _ in range(repeat):
        start = time.perf_counter()
        audio = vocoder.synthesize(params)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    audio = vocoder.synthesize(params)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(audio) == expected, (len(audio), expected)
    return {
        "seconds": duration,
        "samples": int(expected),
        "time_s": float(min(times)),
        "rtf": float(min(times) / duration) if duration else float("nan"),
        "peak_mb": peak / 1e6,
        "rms": float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))),
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fs", type=int, default=22050,
                        help="sample rate to benchmark at (default 22050)")
    parser.add_argument("--frame-period", type=float, default=5.0)
    parser.add_argument("--lengths", type=float, nargs="+",
                        default=list(DEFAULT_LENGTHS),
                        help="utterance lengths in seconds")
    parser.add_argument("--vocoders", nargs="+", default=list(DEFAULT_VOCODERS))
    parser.add_argument("--analysis", default="native",
                        help="backend used to make the parameters "
                             "(native|pyworld|builtin|mlsa|auto)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="synthesis calls per cell; the best time is kept")
    parser.add_argument("--json", default=None,
                        help="also write the results as JSON")
    parser.add_argument("--corpus-dir", default=None,
                        help="where to put the generated demo corpus")
    args = parser.parse_args(argv)

    import tempfile
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir \
        else Path(tempfile.mkdtemp(prefix="hms-bench-vocoder-"))

    available = available_backends()
    print(f"hms vocoder benchmark: fs={args.fs} frame_period={args.frame_period}"
          f"ms repeat={args.repeat}")
    print("backends available: "
          + ", ".join(f"{name}={'yes' if ok else 'no'}"
                      for name, ok in available.items()))
    print("analysis backend: " + args.analysis)
    print()

    results = {}
    for length in args.lengths:
        params = build_parameters(args.fs, length, args.frame_period,
                                  args.analysis, corpus_dir)
        key = f"{length:g}s"
        results[key] = {}
        for name in args.vocoders:
            if name not in available:
                continue
            vocoder = get_vocoder(name, fs=args.fs,
                                  frame_period=args.frame_period)
            # warm up once so the first-call caches (mel warping fit, filter
            # bank construction) do not land on whichever backend goes first
            vocoder.synthesize(params)
            results[key][name] = measure(vocoder, params, args.repeat)

    header = (f"{'length':>7} {'backend':>8} {'samples':>10} {'time ms':>9} "
              f"{'RTF':>8} {'RTF xRT':>8} {'peak MB':>8} {'rms':>8} {'peak':>7}")
    print(header)
    print("-" * len(header))
    for length, cells in results.items():
        for name, cell in cells.items():
            print(f"{length:>7} {name:>8} {cell['samples']:>10d} "
                  f"{cell['time_s'] * 1000:>9.1f} {cell['rtf']:>8.4f} "
                  f"{1.0 / cell['rtf'] if cell['rtf'] else float('nan'):>7.1f}x "
                  f"{cell['peak_mb']:>8.1f} {cell['rms']:>8.4f} "
                  f"{cell['peak']:>7.3f}")

    # -- summary: is MLSA faster / smaller than the builtin fallback? --------
    print()
    summary = {}
    for length, cells in results.items():
        if "mlsa" in cells and "builtin" in cells:
            time_ratio = cells["mlsa"]["time_s"] / cells["builtin"]["time_s"]
            mem_ratio = cells["mlsa"]["peak_mb"] / cells["builtin"]["peak_mb"]
            summary[length] = {"time_ratio_mlsa_over_builtin": time_ratio,
                               "peak_mb_ratio_mlsa_over_builtin": mem_ratio}
            verdict = "faster" if time_ratio < 1.0 else "slower"
            smaller = "smaller" if mem_ratio < 1.0 else "larger"
            print(f"{length:>7} MLSA vs builtin: {time_ratio:6.2f}x time "
                  f"({verdict}), {mem_ratio:6.2f}x peak heap ({smaller})")
    if not summary:
        print("(need both 'mlsa' and 'builtin' in --vocoders for a comparison)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"fs": args.fs, "frame_period": args.frame_period,
             "repeat": args.repeat, "analysis": args.analysis,
             "results": results, "summary": summary}, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
