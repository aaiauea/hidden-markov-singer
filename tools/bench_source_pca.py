"""Benchmark Phase 1 of the source model: cycles -> PCA coefficients -> cycles.

This is the experiment behind `docs/source_model.md`.  It answers one question
with numbers:

    can a handful of PCA coefficients preserve the characteristic shape of a
    source cycle well enough to be worth modelling?

Pipeline (no part of it touches the HMM, the trainer or the vocoder)::

    audio ──► pitch-synchronous residual cycles (128 samples)
          ──► PCA fitted on a train split
          ──► k coefficients per cycle ──► reconstructed cycles
          ──► MSE / RMSE / relative RMSE / explained variance

Reported for k = 4, 8 and 16 (plus 1 and 32 as the useful extremes), on three
views of the same data:

``all cycles``
    every extracted cycle, basis fitted on the first half and measured on the
    second -- the honest "how much of the *whole corpus* does this basis carry"
    number, including different notes, vowels and consonants.
``per pitch``
    cycles grouped by their measured period (i.e. by note), fitted and measured
    within each group.  This is the conditioning a Phase 2 source HMM would
    provide for free, and it is what the shape representation can do when the
    context is fixed.
``longest steady stretch``
    the longest run of near-constant period: the cleanest case, useful as an
    upper bound and as a regression reference.

Usage
-----
    python tools/bench_source_pca.py                      # demo corpus, 22050 Hz
    python tools/bench_source_pca.py --wav my_singing.wav --fs 44100
    python tools/bench_source_pca.py --components 4 8 16 --out out/source
    python tools/bench_source_pca.py --json out/source.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hms.data import wavio                                          # noqa: E402
from hms.data.demo_singer import make_dataset                       # noqa: E402
from hms.source import SourcePCA, get_source_model                  # noqa: E402
from hms.source.cycles import place_cycles                          # noqa: E402
from hms.source.residual import whiten                              # noqa: E402

#: Component counts reported unless overridden on the command line.
DEFAULT_COMPONENTS = (1, 4, 8, 16, 32)

#: Components the summary sentence quotes: "a 128-sample cycle in 4-8 numbers".
HEADLINE_COMPONENTS = (4, 8, 16)


def load_audio(args) -> Tuple[np.ndarray, int]:
    """Demo corpus (default) or a WAV file.

    The demo corpus is concatenated into one long signal: every phrase then
    contributes its cycles to the same fit, which is the "mixed material" case
    this benchmark is about (an utterance boundary is just another unvoiced gap
    to the extractor).
    """
    if args.wav:
        signal, fs = wavio.read_wav(args.wav)
        if args.fs and int(args.fs) != fs:
            raise SystemExit(f"{args.wav} is {fs} Hz, but --fs {args.fs} was given")
        return signal, fs
    info = make_dataset(args.source_dir, fs=int(args.fs), seed=int(args.seed))
    pieces, fs = [], int(args.fs)
    for path in sorted(Path(info["wav_dir"]).glob("*.wav")):
        piece, fs = wavio.read_wav(path)
        pieces.append(piece)
    return np.concatenate(pieces), fs


def steady_run(periods: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    """Mask of the longest run of cycles whose period is (nearly) constant."""
    if len(periods) == 0:
        return np.zeros(0, dtype=bool)
    best_start, best_stop, run_start = 0, 0, 0
    for i in range(1, len(periods) + 1):
        broke = (i == len(periods) or abs(float(periods[i]) - float(periods[run_start]))
                 > tolerance * float(periods[run_start]))
        if broke:
            if i - run_start > best_stop - best_start:
                best_start, best_stop = run_start, i
            run_start = i
    mask = np.zeros(len(periods), dtype=bool)
    mask[best_start:best_stop] = True
    return mask


def group_by_period(periods: np.ndarray, min_cycles: int) -> List[np.ndarray]:
    """Indices of cycles sharing a period (i.e. a note), biggest groups first."""
    groups = []
    for value in np.unique(periods):
        index = np.flatnonzero(periods == value)
        if len(index) >= min_cycles:
            groups.append(index)
    return sorted(groups, key=len, reverse=True)


def measure(cycles: np.ndarray, components: Sequence[int],
            split: float = 0.5) -> Dict[int, dict]:
    """Fit and evaluate each component count on a train/test split of ``cycles``."""
    if len(cycles) < 8:
        return {}
    cut = max(1, int(round(len(cycles) * split)))
    train, test = cycles[:cut], cycles[cut:]
    if len(test) == 0:
        return {}
    results = {}
    for k in components:
        pca = SourcePCA.fit(train, n_components=k)
        report = pca.report(test)
        results[int(k)] = {
            "components": int(k),
            "explained_variance_ratio": float(pca.cumulative_explained_variance),
            "mse": report["mse"],
            "rmse": report["rmse"],
            "relative_rmse": report["relative_rmse"],
            "median_relative_error": report["median_relative_error"],
            "p90_relative_error": report["p90_relative_error"],
            "mean_correlation": report["mean_correlation"],
            "explained_variance": report["explained_variance"],
            "n_train": int(cut),
            "n_test": int(len(test)),
        }
    return results


def print_table(title: str, results: Dict[int, dict], note: str = "") -> None:
    if not results:
        print(f"\n{title}: not enough cycles")
        return
    print(f"\n{title}{('  (' + note + ')') if note else ''}")
    header = (f"{'k':>4} {'expl. var':>10} {'MSE':>9} {'RMSE':>8} {'rel RMSE':>9} "
              f"{'median err':>11} {'p90 err':>8} {'corr':>7} {'data var':>9}")
    print(header)
    print("-" * len(header))
    for k in sorted(results):
        cell = results[k]
        print(f"{k:>4} {cell['explained_variance_ratio']:>10.3f} {cell['mse']:>9.4f} "
              f"{cell['rmse']:>8.4f} {cell['relative_rmse']:>9.3f} "
              f"{cell['median_relative_error']:>11.3f} "
              f"{cell['p90_relative_error']:>8.3f} {cell['mean_correlation']:>7.3f} "
              f"{cell['explained_variance']:>9.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", default=None,
                        help="analyse this WAV instead of the generated demo corpus")
    parser.add_argument("--fs", type=int, default=22050, help="sample rate")
    parser.add_argument("--frame-period", type=float, default=5.0, help="frame period (ms)")
    parser.add_argument("--backend", default="voice", choices=("voice", "residual"),
                        help="source backend to analyse with")
    parser.add_argument("--cycle-length", type=int, default=128,
                        help="samples per source vector")
    parser.add_argument("--components", type=int, nargs="+",
                        default=list(DEFAULT_COMPONENTS),
                        help=f"component counts to report "
                             f"(default {list(DEFAULT_COMPONENTS)})")
    parser.add_argument("--min-group", type=int, default=60,
                        help="minimum cycles per pitch group to report one")
    parser.add_argument("--out", default=None,
                        help="write residual and reconstructed cycles as WAVs here")
    parser.add_argument("--json", default=None, help="write the numbers as JSON")
    parser.add_argument("--source-dir", default=None,
                        help="where to write the generated demo corpus")
    parser.add_argument("--seed", type=int, default=3, help="demo corpus seed")
    parser.add_argument("--f0-file", default=None,
                        help="read an external F0 track (.npy or one Hz per line) "
                             "instead of estimating it")
    args = parser.parse_args()
    args.source_dir = args.source_dir or (Path("out") / "bench_source_corpus")

    signal, fs = load_audio(args)
    f0 = None
    if args.f0_file:
        path = Path(args.f0_file)
        f0 = (np.load(path) if path.suffix == ".npy"
              else np.loadtxt(path, comments="#", ndmin=1))

    model = get_source_model(args.backend, cycle_length=args.cycle_length,
                             fs=fs, frame_period=args.frame_period)
    sequence = model.analyze(signal, fs=fs, frame_period=args.frame_period, f0=f0)
    cycles = sequence.excitation

    print(f"backend            : {sequence.backend} "
          f"(cycle_length={sequence.cycle_length}, n_mcep={model.n_mcep})")
    print(f"audio              : {len(signal) / fs:.2f} s, {fs} Hz, "
          f"{sequence.n_frames} frames @ {args.frame_period:g} ms")
    print(f"voiced frames      : {sequence.voiced.sum()} / {sequence.n_frames} "
          f"({sequence.voiced_fraction * 100:.1f} %)")
    print(f"valid source units : {sequence.n_units} extracted cycles "
          f"({len(cycles)} source vectors)")
    if sequence.n_units:
        period = sequence.periods
        unit_f0 = sequence.unit_f0
        print(f"cycle period       : {period.min()} .. {period.max()} samples, "
              f"median {int(np.median(period))} "
              f"(F0 {unit_f0.min():.1f} .. {unit_f0.max():.1f} Hz, "
              f"median {np.median(unit_f0):.1f} Hz)")
        print(f"cycle noise level  : median {np.median(sequence.noise_level):.3f} "
              f"(0 = clean excitation event, 1 = noise)")
    if len(cycles) == 0:
        print("\nno cycles extracted -- nothing to fit")
        return 0

    # -- 1. all cycles, train/test split over the corpus -------------------
    overall = measure(cycles, args.components)
    print_table("all cycles", overall,
                "basis fitted on the first half, measured on the second")

    # -- 2. per pitch group (the conditioning Phase 2 would add) -----------
    grouped = []
    for index in group_by_period(sequence.periods, args.min_group):
        group = measure(cycles[index], args.components)
        if group:
            grouped.append((float(fs / np.median(sequence.periods[index])),
                            int(len(index)), group))
    if grouped:
        print("\nper pitch (same period, basis fitted and measured within the group)")
        header = (f"{'F0 Hz':>7} {'cycles':>7} " +
                  " ".join(f"{'k=' + str(k):>21}" for k in sorted(overall)))
        print(header)
        print("-" * len(header))
        for f0_hz, count, group in grouped[:8]:
            cells = " ".join(
                f"{group[k]['explained_variance_ratio']:>9.3f}/"
                f"{group[k]['relative_rmse']:>10.3f}" if k in group else f"{'-':>20}"
                for k in sorted(overall))
            print(f"{f0_hz:>7.1f} {count:>7d} {cells}")
        print("(cells: explained variance / held-out relative RMSE)")

    # -- 3. the steadiest stretch ------------------------------------------
    mask = steady_run(sequence.periods)
    steady = measure(cycles[mask], args.components) if mask.sum() >= 8 else {}
    if steady:
        print_table("longest steady stretch", steady,
                    f"{int(mask.sum())} cycles at "
                    f"~{fs / np.median(sequence.periods[mask]):.0f} Hz")

    # -- 4. the source round trip (no PCA involved) ------------------------
    reconstruction = model.synthesize(sequence)
    residual = whiten(signal, fs, args.frame_period,
                      fft_size=getattr(model, "fft_size", None),
                      n_mcep=model.n_mcep, f_min=model.f_min)
    covered = np.zeros(len(signal), dtype=bool)
    for epoch, period in zip(sequence.epochs, sequence.periods):
        covered[epoch:min(len(signal), epoch + period)] = True
    if covered.any():
        original, rebuilt = residual[covered], reconstruction[covered]
        print(f"\ncycle round trip   : correlation "
              f"{np.corrcoef(original, rebuilt)[0, 1]:.4f}, relative error "
              f"{np.linalg.norm(original - rebuilt) / max(np.linalg.norm(original), 1e-12):.4f} "
              f"over {int(covered.sum())} samples")
        print("                     (residual -> cycles -> residual, no PCA: the loss "
              "is the fixed-length resampling)")

    # -- 5. answer the headline question -----------------------------------
    print("\nverdict")
    for k in HEADLINE_COMPONENTS:
        if k in overall:
            cell = overall[k]
            print(f"  k={k:<3d} {cell['explained_variance_ratio'] * 100:5.1f} % of the "
                  f"basis variance, held-out relative RMSE {cell['relative_rmse']:.3f}, "
                  f"mean cycle correlation {cell['mean_correlation']:.3f}")
    if grouped and HEADLINE_COMPONENTS[1] in grouped[0][2]:
        cell = grouped[0][2][HEADLINE_COMPONENTS[1]]
        print(f"  within one pitch (k={HEADLINE_COMPONENTS[1]}): relative RMSE "
              f"{cell['relative_rmse']:.3f}, correlation {cell['mean_correlation']:.3f}")
    if steady and HEADLINE_COMPONENTS[1] in steady:
        cell = steady[HEADLINE_COMPONENTS[1]]
        print(f"  steady stretch (k={HEADLINE_COMPONENTS[1]}): relative RMSE "
              f"{cell['relative_rmse']:.3f}, correlation {cell['mean_correlation']:.3f}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        wavio.write_wav(out / "residual.wav", residual, fs)
        wavio.write_wav(out / "cycles_roundtrip.wav", reconstruction, fs)
        for k in HEADLINE_COMPONENTS:
            if k not in overall:
                continue
            pca = SourcePCA.fit(cycles, n_components=k)
            decoded = model.decode(pca.encode(cycles), pca, sequence.gains)
            wavio.write_wav(out / f"cycles_pca{k}.wav",
                            place_cycles(decoded, sequence.epochs, sequence.periods,
                                         sequence.n_samples), fs)
        basis_path = SourcePCA.fit(cycles, n_components=max(HEADLINE_COMPONENTS)
                                   ).save(out / "source_pca.npz")
        print(f"\nwrote {out}/residual.wav, cycles_roundtrip.wav, cycles_pca*.wav "
              f"and {basis_path.name}")

    if args.json:
        payload = {
            "backend": sequence.backend, "fs": fs,
            "frame_period": args.frame_period, "cycle_length": sequence.cycle_length,
            "n_frames": sequence.n_frames, "n_units": sequence.n_units,
            "voiced_fraction": sequence.voiced_fraction,
            "all_cycles": overall,
            "per_pitch": [{"f0_hz": f0_hz, "cycles": count, "results": group}
                          for f0_hz, count, group in grouped],
            "steady": steady,
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
