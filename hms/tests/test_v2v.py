"""Focused tests for the experimental LPC/WORLD V2V analysis side path."""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.dsp import frame_signal
from hms.core.v2v import (V2VAnalysisConfig, V2VFrontend,
                          _lpc_cepstral_coefficients, analyze_v2v)
from hms.vocoder import get_vocoder

FS = 22050
FRAME_PERIOD_MS = 5.0


def config(**overrides) -> V2VAnalysisConfig:
    values = {"vocoder": "builtin", "frame_period_ms": FRAME_PERIOD_MS}
    values.update(overrides)
    return V2VAnalysisConfig(**values)


def harmonic_signal(frequency: float = 220.0, duration: float = 0.35,
                    amplitude: float = 0.2, fs: int = FS) -> np.ndarray:
    time = np.arange(int(round(duration * fs)), dtype=np.float64) / fs
    return amplitude * (np.sin(2.0 * np.pi * frequency * time)
                        + 0.2 * np.sin(4.0 * np.pi * frequency * time))


def assert_finite_analysis(result) -> None:
    for name in ("frame_times_s", "center_sample_indices", "f0_hz",
                 "energy_rms", "lpc_coefficients", "reflection_coefficients",
                 "prediction_error_fraction", "lpc_cepstra",
                 "lpc_log_spectrum", "spectrum_frequencies_hz",
                 "candidate_features"):
        assert np.isfinite(np.asarray(getattr(result, name))).all(), name


def test_lpc_extraction_is_deterministic_with_explicit_shapes():
    rng = np.random.default_rng(22)
    audio = harmonic_signal() + rng.normal(0.0, 0.002, size=round(0.35 * FS))
    settings = config(lpc_order=14, cepstral_order=18, spectrum_bins=129)

    first = analyze_v2v(audio, FS, settings)
    second = analyze_v2v(audio, FS, settings)

    assert first.backend == "builtin"
    assert first.lpc_coefficients.shape == (len(first), 15)
    assert first.reflection_coefficients.shape == (len(first), 14)
    assert first.lpc_cepstra.shape == (len(first), 18)
    assert first.lpc_log_spectrum.shape == (len(first), 129)
    assert first.candidate_features.shape == (len(first), 21)
    assert len(first.candidate_feature_names) == 21
    assert np.array_equal(first.f0_hz, second.f0_hz)
    assert np.array_equal(first.lpc_coefficients, second.lpc_coefficients)
    assert np.array_equal(first.lpc_cepstra, second.lpc_cepstra)
    assert_finite_analysis(first)


def test_lpc_cepstrum_reconstructs_the_all_pole_log_amplitude():
    # A stable second-order polynomial has an infinite causal cepstrum. Forty
    # terms are enough to recover its log-amplitude response to float precision;
    # this guards the distinction between raw LPC coefficients and cepstra.
    polynomial = np.array([[1.0, -0.4, 0.12]])
    cepstra = _lpc_cepstral_coefficients(polynomial, cepstral_order=40)[0]
    frequencies = np.linspace(0.0, np.pi, 129)
    reconstructed = sum(
        cepstra[index - 1] * np.cos(index * frequencies)
        for index in range(1, len(cepstra) + 1))
    response = np.fft.rfft(polynomial, n=256, axis=1)[0]
    expected = -np.log(np.abs(response))

    assert np.allclose(reconstructed, expected, atol=1e-12)


def test_silence_has_finite_zero_shape_features_and_unvoiced_f0():
    result = analyze_v2v(np.zeros(int(0.2 * FS)), FS, config())

    assert len(result) > 0
    assert not result.voiced.any()
    assert np.array_equal(result.f0_hz, np.zeros(len(result)))
    assert np.array_equal(result.energy_rms, np.zeros(len(result)))
    expected_lpc = np.zeros((len(result), result.lpc_order + 1))
    expected_lpc[:, 0] = 1.0
    assert np.array_equal(result.lpc_coefficients, expected_lpc)
    assert not result.reflection_coefficients.any()
    assert not result.prediction_error_fraction.any()
    assert not result.lpc_cepstra.any()
    assert not result.lpc_log_spectrum.any()
    assert_finite_analysis(result)


def test_trailing_silence_holds_spectral_shape_not_other_features():
    settings = config()
    voice = harmonic_signal(frequency=220.0, duration=0.3)
    audio = np.concatenate((voice, np.zeros(int(0.2 * FS))))
    result = analyze_v2v(audio, FS, settings)

    voiced_active = np.flatnonzero(result.voiced & (result.energy_rms > 0.0))
    assert voiced_active.size
    last_voice = int(voiced_active[-1])
    silent_tail = np.flatnonzero(
        (np.arange(len(result)) > last_voice) & (result.energy_rms == 0.0))
    assert silent_tail.size

    candidate = result.candidate_features
    cepstral_slice = slice(0, settings.cepstral_order)
    assert np.allclose(candidate[silent_tail, cepstral_slice],
                       candidate[last_voice, cepstral_slice])
    # Only spectral shape is edge-held; source voicing, energy and the true
    # silent-frame LPC identity solution are retained as observations.
    assert np.allclose(result.lpc_cepstra[silent_tail],
                       result.lpc_cepstra[last_voice])
    assert np.allclose(result.lpc_log_spectrum[silent_tail],
                       result.lpc_log_spectrum[last_voice])
    assert not result.f0_hz[silent_tail].any()
    expected_identity = np.zeros((len(silent_tail), result.lpc_order + 1))
    expected_identity[:, 0] = 1.0
    assert np.array_equal(result.lpc_coefficients[silent_tail],
                          expected_identity)
    assert not candidate[silent_tail, settings.cepstral_order].any()
    assert not candidate[silent_tail, settings.cepstral_order + 1].any()
    assert np.all(candidate[silent_tail, -1] == -240.0)

    # Without a valid voiced frame there is no edge value to carry forward.
    all_silence = analyze_v2v(np.zeros(int(0.2 * FS)), FS, settings)
    assert not all_silence.candidate_features[:, cepstral_slice].any()


def test_noisy_unvoiced_input_keeps_stable_finite_lpc_parameters():
    rng = np.random.default_rng(5)
    noise = rng.normal(0.0, 0.04, size=int(0.25 * FS))
    settings = config(lpc_order=16, reflection_limit=0.98)
    result = analyze_v2v(noise, FS, settings)

    assert len(result) > 10
    assert not result.voiced.any()  # builtin voicing also checks spectral flatness
    assert np.max(np.abs(result.reflection_coefficients)) <= 0.98
    assert_finite_analysis(result)
    # Reflection clipping plus Levinson recursion should leave stable AR poles.
    for polynomial in result.lpc_coefficients[::7]:
        assert np.max(np.abs(np.roots(polynomial))) <= 1.0 + 1e-7


def test_very_short_signal_and_short_lpc_window_are_safe():
    settings = config(frame_length_ms=0.1, lpc_order=16)
    result = analyze_v2v(np.array([0.25]), FS, settings)

    assert len(result) > 0
    assert result.frame_length_samples >= settings.lpc_order + 2
    assert result.lpc_coefficients.shape == (len(result), 17)
    assert_finite_analysis(result)

    empty = analyze_v2v(np.zeros(0), FS, settings)
    assert len(empty) == 0
    assert empty.lpc_coefficients.shape == (0, 17)
    assert empty.candidate_features.shape == (0, settings.cepstral_order + 3)
    assert empty.backend == "not-run"
    assert_finite_analysis(empty)


def test_non_contiguous_audio_matches_a_contiguous_copy():
    source = harmonic_signal(duration=0.4)
    interleaved = np.zeros(source.size * 2, dtype=np.float64)
    interleaved[::2] = source
    non_contiguous = interleaved[::2]
    assert not non_contiguous.flags.c_contiguous

    strided = analyze_v2v(non_contiguous, FS, config())
    contiguous = analyze_v2v(non_contiguous.copy(), FS, config())

    assert np.array_equal(strided.f0_hz, contiguous.f0_hz)
    assert np.array_equal(strided.energy_rms, contiguous.energy_rms)
    assert np.array_equal(strided.lpc_coefficients, contiguous.lpc_coefficients)
    assert np.array_equal(strided.lpc_log_spectrum, contiguous.lpc_log_spectrum)


def test_f0_and_lpc_share_the_backend_frame_count_and_time_grid():
    tone = harmonic_signal(frequency=220.0, duration=0.3)
    noise = np.random.default_rng(14).normal(0.0, 0.04, int(0.25 * FS))
    silence = np.zeros(int(0.1 * FS))
    audio = np.concatenate((tone, noise, silence, tone))
    settings = config()
    vocoder = get_vocoder("builtin", fs=FS, frame_period=FRAME_PERIOD_MS)
    expected_f0, _sp, _ap = vocoder.analyze(
        audio, fs=FS, frame_period=FRAME_PERIOD_MS,
        f0_floor=settings.f0_floor, f0_ceil=settings.f0_ceil,
        f0_estimation=settings.f0_estimation, refine_f0=settings.refine_f0)

    result = V2VFrontend(settings, vocoder=vocoder).analyze(audio, FS)
    expected_f0 = np.asarray(expected_f0, dtype=np.float64).reshape(-1)
    expected_f0 = np.where(np.isfinite(expected_f0) & (expected_f0 > 0),
                           expected_f0, 0.0)

    assert np.array_equal(result.f0_hz, expected_f0)
    assert len(result.frame_times_s) == len(result.f0_hz) == len(result.lpc_cepstra)
    assert result.frame_times_s[0] == 0.0
    assert np.allclose(np.diff(result.frame_times_s), FRAME_PERIOD_MS / 1000.0)
    assert np.array_equal(
        result.center_sample_indices,
        np.rint(result.frame_times_s * FS).astype(np.int64))
    assert result.voiced.mean() > 0.4
    # Steady middle of the voiced source remains close to the known F0.  The
    # builtin estimator is intentionally approximate; WORLD backends are the
    # preferred analyser where installed.
    middle = result.voiced & (result.frame_times_s > 0.08) \
        & (result.frame_times_s < 0.24)
    assert middle.any()
    assert np.median(result.f0_hz[middle]) == pytest.approx(220.0, rel=0.04)
    noisy_region = ((result.frame_times_s > 0.34)
                    & (result.frame_times_s < 0.53))
    assert not result.voiced[noisy_region].any()
    assert_finite_analysis(result)


@pytest.mark.parametrize(
    ("fundamental_hz", "partial_amplitudes"),
    [(110.0, (0.05, 0.8, 0.3, 0.1)),
     (165.0, (1.0, 0.5, 0.2, 0.1)),
     (220.0, (1.0, 0.5, 0.2, 0.1))])
def test_builtin_tracker_follows_110_hz_fundamental_without_harming_midrange(
        fundamental_hz, partial_amplitudes):
    time = np.arange(int(0.6 * FS), dtype=np.float64) / FS
    audio = 0.2 * sum(
        amplitude * np.sin(2.0 * np.pi * fundamental_hz * harmonic * time)
        for harmonic, amplitude in enumerate(partial_amplitudes, start=1))
    result = analyze_v2v(audio, FS, config(f0_floor=71.0, f0_ceil=800.0))
    hop = int(round(FS * FRAME_PERIOD_MS / 1000.0))
    original_f0_frames = len(frame_signal(audio, 4 * hop, hop))
    assert len(result) == original_f0_frames

    interior = ((result.frame_times_s >= 0.12)
                & (result.frame_times_s <= 0.48)
                & result.voiced)
    estimates = result.f0_hz[interior]
    assert estimates.size > 0
    median_f0 = float(np.median(estimates))
    assert median_f0 == pytest.approx(fundamental_hz, rel=0.04)
    if fundamental_hz == 110.0:
        # The strong second harmonic makes the former short-window tracker
        # settle near 220 Hz; keep this an explicit octave-error regression.
        assert median_f0 < 165.0


def test_lpc_shape_features_are_separate_from_overall_loudness():
    quiet = analyze_v2v(harmonic_signal(amplitude=0.02), FS, config())
    loud = analyze_v2v(harmonic_signal(amplitude=0.2), FS, config())

    assert np.allclose(loud.energy_rms, quiet.energy_rms * 10.0,
                       rtol=1e-10, atol=1e-12)
    assert np.allclose(loud.lpc_coefficients, quiet.lpc_coefficients,
                       rtol=1e-9, atol=1e-10)
    assert np.allclose(loud.lpc_cepstra, quiet.lpc_cepstra,
                       rtol=1e-9, atol=1e-10)
    assert np.allclose(loud.lpc_log_spectrum, quiet.lpc_log_spectrum,
                       rtol=1e-9, atol=1e-10)
    assert not np.allclose(loud.candidate_features[:, -1],
                           quiet.candidate_features[:, -1])


def test_frontend_rejects_non_mono_non_finite_and_invalid_rate_inputs():
    frontend = V2VFrontend(config())
    with pytest.raises(ValueError, match="mono 1-D"):
        frontend.analyze(np.zeros((100, 2)), FS)
    with pytest.raises(ValueError, match="finite"):
        frontend.analyze(np.array([0.0, np.nan]), FS)
    with pytest.raises(ValueError, match="positive integer"):
        frontend.analyze(np.zeros(100), 0)


def test_feature_geometry_validation_prevents_lpc_spectrum_truncation():
    with pytest.raises(ValueError, match="spectrum_bins is too small"):
        V2VAnalysisConfig(lpc_order=32, spectrum_bins=8)
