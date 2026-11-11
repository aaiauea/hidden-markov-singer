"""Reproducible, synthesis-independent probe of the experimental V2V frontend.

The script generates its probes with HMS's existing deterministic synthetic
singer. It reports extraction latency/memory, F0 on a known 220 Hz tone, LPC
prediction residual, LPC-vs-periodogram spectral correlation, and a tiny
nearest-centroid vowel probe trained at one pitch/loudness and tested at
others. The vowel score is a synthetic sanity check, not a speech-recognition
or cross-speaker V2V result.

Examples::

    python tools/bench_v2v.py --backend builtin
    python tools/bench_v2v.py --backend auto --seconds 5 --json /tmp/v2v.json

No HMS model is trained or synthesized by this benchmark.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hms.core.v2v import V2VAnalysis, V2VAnalysisConfig, V2VFrontend  # noqa: E402
from hms.data.demo_singer import DemoSinger, SegmentSpec, SingerConfig  # noqa: E402


def make_voice_probe(fs: int, seconds: float) -> np.ndarray:
    """Deterministic source-filter audio long enough for the timed probe."""
    phones = ("a", "i", "u", "e", "o", "m", "a", "sil")
    notes = (60.0, 64.0, 57.0, 62.0, 59.0, 60.0, 67.0, None)
    segment_ms = max(100.0, seconds * 1000.0 / len(phones))
    script = [SegmentSpec(phone, segment_ms, note)
              for phone, note in zip(phones, notes)]
    singer = DemoSinger(SingerConfig(
        fs=fs, seed=2026, jitter=0.0, vibrato_semitones=0.0,
        scoop_semitones=0.0, drift_semitones=0.0))
    audio = singer.render(script)
    n_samples = max(1, int(round(seconds * fs)))
    if len(audio) < n_samples:
        audio = np.tile(audio, int(np.ceil(n_samples / max(len(audio), 1))))
    return np.ascontiguousarray(audio[:n_samples], dtype=np.float64)


def make_tone(fs: int, duration: float = 0.8, f0: float = 220.0
              ) -> np.ndarray:
    time_axis = np.arange(int(round(fs * duration)), dtype=np.float64) / fs
    return 0.25 * np.sin(2.0 * np.pi * f0 * time_axis)


def peak_traced_bytes(operation) -> int:
    """Python/NumPy traced allocation peak (not process RSS)."""
    gc.collect()
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()


def _centered_frames(audio: np.ndarray, centers: np.ndarray,
                     length: int) -> np.ndarray:
    offsets = np.arange(length, dtype=np.int64) - length // 2
    indices = centers[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < len(audio))
    frames = np.zeros(indices.shape, dtype=np.float64)
    if valid.any():
        rows, columns = np.nonzero(valid)
        frames[rows, columns] = audio[indices[rows, columns]]
    return frames


def spectral_metrics(audio: np.ndarray,
                     analysis: V2VAnalysis) -> dict:
    """Compare mean-centred LPC log power with the frame periodogram."""
    if not len(analysis):
        return {"median_correlation": None, "median_log_power_rmse": None,
                "usable_frames": 0}
    n_fft = 2 * (analysis.lpc_log_spectrum.shape[1] - 1)
    frames = _centered_frames(audio, analysis.center_sample_indices,
                              analysis.frame_length_samples)
    windowed = frames * np.hanning(analysis.frame_length_samples)[None, :]
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    log_power = np.log(np.maximum(
        spectrum.real * spectrum.real + spectrum.imag * spectrum.imag, 1e-12))
    log_power -= log_power.mean(axis=1, keepdims=True)
    predicted = analysis.lpc_log_spectrum
    numerator = np.sum(log_power * predicted, axis=1)
    denominator = np.sqrt(np.sum(log_power * log_power, axis=1)
                          * np.sum(predicted * predicted, axis=1))
    correlation = np.divide(numerator, denominator,
                            out=np.full_like(numerator, np.nan),
                            where=denominator > 1e-12)
    frame_rmse = np.sqrt(np.mean((log_power - predicted) ** 2, axis=1))
    usable = (np.isfinite(correlation) & np.isfinite(frame_rmse)
              & (analysis.energy_rms > 1e-5))
    if not usable.any():
        return {"median_correlation": None, "median_log_power_rmse": None,
                "usable_frames": 0}
    return {
        "median_correlation": float(np.median(correlation[usable])),
        "median_log_power_rmse": float(np.median(frame_rmse[usable])),
        "usable_frames": int(usable.sum()),
    }


def vowel_probe(frontend: V2VFrontend, fs: int) -> dict:
    """Synthetic same-singer vowel classification across pitch/gain changes."""
    phones = ("a", "e", "i", "o", "u")
    duration_ms = 600.0
    train_note, train_gain = 60.0, 1.0
    test_notes, test_gains = (48.0, 72.0), (0.25, 2.0)

    def observe(phone: str, note: float, gain: float, seed: int) -> np.ndarray:
        singer = DemoSinger(SingerConfig(
            fs=fs, seed=seed, jitter=0.0, vibrato_semitones=0.0,
            scoop_semitones=0.0, drift_semitones=0.0))
        audio = singer.render([SegmentSpec(phone, duration_ms, note)]) * gain
        result = frontend.analyze(audio, fs)
        stable = ((result.frame_times_s >= 0.12)
                  & (result.frame_times_s <= duration_ms / 1000.0 - 0.12))
        if not stable.any():
            raise RuntimeError("demo vowel probe produced no central analysis frames")
        return result.lpc_cepstra[stable]

    training = {phone: observe(phone, train_note, train_gain, 100 + index)
                for index, phone in enumerate(phones)}
    all_training = np.concatenate(list(training.values()), axis=0)
    mean = all_training.mean(axis=0)
    scale = np.maximum(all_training.std(axis=0), 1e-3)
    centroids = np.stack([
        ((values - mean) / scale).mean(axis=0)
        for values in training.values()
    ])

    correct = 0
    trials = 0
    for phone_index, phone in enumerate(phones):
        for note in test_notes:
            for gain_index, gain in enumerate(test_gains):
                values = observe(phone, note, gain,
                                 1000 + 100 * phone_index + 10 * gain_index
                                 + int(note))
                vector = (np.median(values, axis=0) - mean) / scale
                predicted = int(np.argmin(np.sum((centroids - vector) ** 2,
                                                 axis=1)))
                correct += int(predicted == phone_index)
                trials += 1
    return {
        "labels": list(phones),
        "classifier": "nearest centroid on LPC cepstra only",
        "training_note_midi": train_note,
        "training_gain": train_gain,
        "test_notes_midi": list(test_notes),
        "test_gain_scales": list(test_gains),
        "correct": correct,
        "trials": trials,
        "accuracy": correct / trials if trials else None,
        "scope": "synthetic same-singer vowels; no held-out speaker or real speech",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("builtin", "auto", "native", "pyworld"),
                        default="builtin",
                        help="vocoder F0 backend; builtin is the portable baseline")
    parser.add_argument("--fs", type=int, default=22050)
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="duration of the source-filter waveform used for timing")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.fs <= 0 or not np.isfinite(args.seconds) or args.seconds <= 0 \
            or args.repeat < 1 or args.warmup < 0:
        parser.error("fs/seconds/repeat must be positive and warmup non-negative")

    config = V2VAnalysisConfig(vocoder=args.backend)
    frontend = V2VFrontend(config)
    audio = make_voice_probe(args.fs, args.seconds)
    for _ in range(args.warmup):
        frontend.analyze(audio, args.fs)

    samples = []
    latest = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        latest = frontend.analyze(audio, args.fs)
        samples.append(1000.0 * (time.perf_counter() - started))
    assert latest is not None
    extraction_ms = float(statistics.median(samples))
    traced_peak = peak_traced_bytes(lambda: frontend.analyze(audio, args.fs))
    output_bytes = sum(np.asarray(value).nbytes for value in (
        latest.frame_times_s, latest.center_sample_indices, latest.f0_hz,
        latest.energy_rms, latest.lpc_coefficients,
        latest.reflection_coefficients, latest.prediction_error_fraction,
        latest.lpc_cepstra, latest.lpc_log_spectrum,
        latest.spectrum_frequencies_hz))
    candidate_bytes = latest.candidate_features.nbytes

    calibration_audio = make_tone(args.fs)
    calibration = frontend.analyze(calibration_audio, args.fs)
    central_tone = (calibration.voiced
                    & (calibration.frame_times_s >= 0.1)
                    & (calibration.frame_times_s <= 0.6))
    measured_f0 = (float(np.median(calibration.f0_hz[central_tone]))
                   if central_tone.any() else None)
    f0_error_cents = (float(1200.0 * np.log2(measured_f0 / 220.0))
                      if measured_f0 and measured_f0 > 0 else None)
    active_frames = latest.energy_rms > 1e-5
    median_prediction_error = (
        float(np.median(latest.prediction_error_fraction[active_frames]))
        if active_frames.any() else None)
    max_reflection = (float(np.max(np.abs(latest.reflection_coefficients)))
                      if latest.reflection_coefficients.size else 0.0)
    spectral = spectral_metrics(audio, latest)
    phones = vowel_probe(frontend, args.fs)

    result = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "configuration": {
            "requested_backend": args.backend,
            "actual_backend": latest.backend,
            "sample_rate_hz": args.fs,
            "frame_period_ms": config.frame_period_ms,
            "frame_length_ms": config.frame_length_ms,
            "lpc_order": config.lpc_order,
            "cepstral_order": config.cepstral_order,
            "candidate_feature_dim": len(latest.candidate_feature_names),
            "spectrum_bins": config.spectrum_bins,
        },
        "frontend_benchmark": {
            "source_seconds": len(audio) / args.fs,
            "samples": len(audio),
            "repeat": args.repeat,
            "warmup": args.warmup,
            "median_extraction_ms": extraction_ms,
            "real_time_factor": (len(audio) / args.fs) / (extraction_ms / 1000.0),
            "tracemalloc_peak_bytes": traced_peak,
            "retained_output_bytes": output_bytes,
            "candidate_matrix_bytes": candidate_bytes,
        },
        "frame_alignment": {
            "frames": len(latest),
            "first_time_s": (float(latest.frame_times_s[0])
                             if len(latest) else None),
            "last_time_s": (float(latest.frame_times_s[-1])
                            if len(latest) else None),
            "nominal_hop_ms": latest.frame_period_ms,
            "audio_duration_seconds": len(audio) / args.fs,
        },
        "f0_calibration": {
            "known_tone_hz": 220.0,
            "measured_median_hz": measured_f0,
            "error_cents": f0_error_cents,
            "voiced_fraction": float(calibration.voiced.mean())
            if len(calibration) else 0.0,
        },
        "lpc_diagnostics": {
            "median_normalized_prediction_error_fraction": median_prediction_error,
            "maximum_absolute_reflection_coefficient": max_reflection,
            "reflection_limit": config.reflection_limit,
            "median_lpc_log_spectrum_vs_periodogram_correlation":
                spectral["median_correlation"],
            "median_log_power_envelope_rmse": spectral["median_log_power_rmse"],
            "spectral_comparison_frames": spectral["usable_frames"],
        },
        "synthetic_vowel_probe": phones,
        "interpretation_limits": [
            "The synthetic classifier measures vowel separability for one hand-built voice only.",
            "LPC prediction-residual and spectral-fit metrics do not prove source-independent content encoding.",
            "No phoneme/context recognizer, aligner, target-feature mapper or HMS render is evaluated.",
        ],
    }

    print(f"HMS V2V frontend benchmark | Python {result['environment']['python']} "
          f"| NumPy {np.__version__} | backend {latest.backend}")
    print(f"Input: {len(audio) / args.fs:.3f} s at {args.fs} Hz; "
          f"{len(latest)} frames x {len(latest.candidate_feature_names)} candidate dimensions")
    print(f"Extraction median: {extraction_ms:.3f} ms "
          f"({result['frontend_benchmark']['real_time_factor']:.1f}x real time)")
    print(f"Memory: traced peak {traced_peak / 1048576:.2f} MiB; "
          f"retained arrays {output_bytes / 1048576:.2f} MiB "
          f"(+ candidate matrix {candidate_bytes / 1024:.1f} KiB)")
    print(f"Known 220 Hz tone: median {measured_f0!r} Hz, "
          f"error {f0_error_cents!r} cents")
    print(f"LPC: median relative prediction error {median_prediction_error!r}; "
          f"max |reflection| {max_reflection:.4f}; "
          f"median log-spectrum correlation {spectral['median_correlation']!r}; "
          f"median centred log-power RMSE "
          f"{spectral['median_log_power_rmse']!r}")
    print(f"Synthetic vowel probe: {phones['correct']}/{phones['trials']} "
          f"({phones['accuracy']:.1%}) across pitch/gain changes "
          "(same synthetic singer; not a real-speech score)")
    print("No HMS target model or synthesis path was called.")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n",
                             encoding="utf-8")
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
