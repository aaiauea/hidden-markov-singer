"""Source models: the voice backend, the generic backend and the shared API.

The audio used here is generated locally (a pulse train through a few
resonators) rather than taken from the demo corpus, so these tests are fast,
deterministic and independent of the corpus fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.source import (SOURCE_BACKENDS, SourceFrame, SourceModel, SourcePCA,
                        SourceSequence, available_backends, get_source_model,
                        source_model_for)
from hms.source.residual import whiten


# --------------------------------------------------------------------------
# A synthetic voice: glottal pulses through a formant filter
# --------------------------------------------------------------------------


def glottal_pulse(length: int) -> np.ndarray:
    """A Rosenberg-style pulse, differentiated (radiation at the lips)."""
    t = np.linspace(0.0, 1.0, length, endpoint=False)
    opening = 0.5 * (1.0 - np.cos(np.pi * t / 0.4)) * (t < 0.4)
    closing = np.cos(np.pi * (t - 0.4) / (2 * 0.6)) * (t >= 0.4)
    pulse = np.where(t < 0.4, opening, closing)
    pulse = np.diff(pulse, prepend=pulse[0])
    return pulse / max(np.max(np.abs(pulse)), 1e-9)


def resonator(freq: float, bandwidth: float, fs: int, length: int = 600) -> np.ndarray:
    """Impulse response of a two-pole resonator."""
    r = np.exp(-np.pi * bandwidth / fs)
    w = 2 * np.pi * freq / fs
    n = np.arange(length)
    return (r ** n) * np.sin((n + 1) * w) / max(np.sin(w), 1e-9)


def pulse_period(fs: int, f0: float) -> int:
    """The integer period of the synthetic pulse train for a nominal F0.

    Pulses are placed on a whole number of samples: a fractional spacing makes
    the onsets alternate (24, 25, 24, ...) and the pulse train is then only
    periodic at every second pulse, which is an analysis artefact, not a voice.
    """
    return max(2, int(round(fs / f0)))


def synthetic_voice(f0: float = 220.0, duration: float = 1.0, fs: int = 22050,
                    formants=((700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0)),
                    noise: float = 1e-4, seed: int = 0) -> np.ndarray:
    """A steady "vowel": glottal pulses through a formant cascade.

    The effective pitch is ``fs / pulse_period(fs, f0)``; use that when
    comparing against the analysed track.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    period = pulse_period(fs, f0)
    pulse = glottal_pulse(max(4, int(0.6 * period)))
    train = np.zeros(n + len(pulse))
    position = 0
    while position < n:
        train[position:position + len(pulse)] += pulse[:len(train) - position]
        position += period
    signal = train[:n]
    for freq, bandwidth in formants:
        signal = np.convolve(signal, resonator(freq, bandwidth, fs), mode="same")
    signal /= max(np.abs(signal).max(), 1e-9)
    return signal + noise * rng.standard_normal(n)


# --------------------------------------------------------------------------
# Voice backend: analysis
# --------------------------------------------------------------------------


def test_voice_analysis_extracts_one_cycle_per_period():
    fs, f0, duration = 22050, 220.0, 1.0
    signal = synthetic_voice(f0=f0, duration=duration, fs=fs)
    pitch = fs / pulse_period(fs, f0)
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs)

    assert isinstance(sequence, SourceSequence)
    assert sequence.backend == "voice"
    assert sequence.n_samples == len(signal)
    assert sequence.voiced_fraction > 0.9
    expected = pitch * duration
    assert 0.7 * expected <= sequence.n_units <= 1.3 * expected
    assert sequence.excitation.shape[1] == model.cycle_length
    assert np.isfinite(sequence.excitation).all()
    rms = np.sqrt((sequence.excitation ** 2).mean(axis=1))
    assert np.allclose(rms, 1.0, atol=1e-9)
    # the measured pitch is the pitch that went in
    assert np.median(sequence.unit_f0) == pytest.approx(pitch, rel=0.02)
    assert np.median(sequence.periods) == pytest.approx(pulse_period(fs, f0), abs=1.0)
    assert (np.diff(sequence.epochs) > 0).all()
    assert (sequence.gains > 0).all()
    assert ((sequence.noise_level >= 0) & (sequence.noise_level <= 1)).all()


def test_voice_analysis_uses_a_supplied_f0_track():
    fs, f0 = 22050, 300.0
    signal = synthetic_voice(f0=f0, fs=fs, duration=0.5)
    hop = 110
    n_frames = len(signal) // hop
    track = np.full(n_frames, f0)
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs, f0=track)
    assert len(sequence.f0) == len(track)
    assert np.array_equal(sequence.f0, track)
    assert np.median(sequence.unit_f0) == pytest.approx(f0, rel=0.05)
    assert sequence.n_units > 50


def test_voice_analysis_sanitises_a_broken_f0_track():
    fs, f0 = 22050, 250.0
    signal = synthetic_voice(f0=f0, fs=fs, duration=0.5)
    track = np.full(100, f0)
    track[:10] = 0.0
    track[10:15] = np.nan
    track[15:20] = -100.0
    track[20:25] = np.inf
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs, f0=track)
    assert np.isfinite(sequence.f0).all()
    assert (sequence.f0 >= 0).all()
    assert not sequence.voiced[:25].any()
    assert sequence.voiced[30:].all()
    assert sequence.n_units > 0


def test_voice_analysis_is_deterministic():
    signal = synthetic_voice(fs=22050, duration=0.4)
    model = get_source_model("voice", fs=22050)
    first = model.analyze(signal, fs=22050)
    second = model.analyze(signal, fs=22050)
    assert np.array_equal(first.excitation, second.excitation)
    assert np.array_equal(first.epochs, second.epochs)
    assert np.array_equal(first.f0, second.f0)
    assert np.array_equal(first.gains, second.gains)


def test_voice_analysis_rejects_non_finite_audio():
    signal = np.zeros(4000)
    signal[100] = np.nan
    with pytest.raises(ValueError):
        get_source_model("voice", fs=22050).analyze(signal, fs=22050)


@pytest.mark.parametrize("duration", (0.0, 1.0 / 22050, 0.01, 0.05))
def test_voice_analysis_on_extremely_short_audio(duration):
    """A signal shorter than one period is still a well-formed sequence."""
    fs = 22050
    model = get_source_model("voice", fs=fs)
    signal = synthetic_voice(fs=fs, duration=0.2)[:int(round(duration * fs))]
    sequence = model.analyze(signal, fs=fs)
    assert isinstance(sequence, SourceSequence)
    assert sequence.n_samples == len(signal)
    assert sequence.n_units == len(sequence.epochs) == len(sequence.gains)
    # every kept cycle spans at least MIN_PERIOD_SAMPLES and fits in the signal
    for epoch, period in zip(sequence.epochs, sequence.periods):
        assert 4 <= period
        assert 0 <= epoch and epoch + period <= len(signal)
    assert np.isfinite(sequence.excitation).all()
    # the codec must cope with an empty or tiny cycle set
    pca = SourcePCA.fit(np.vstack([sequence.excitation, np.zeros((4, model.cycle_length))]),
                        n_components=4)
    coefficients = model.encode(sequence, pca)
    assert coefficients.shape == (sequence.n_units, 4)
    assert np.isfinite(model.decode(coefficients, pca)).all()
    assert model.synthesize(sequence).shape == (max(len(signal), 0),)


def test_fully_unvoiced_and_silent_input_produce_no_cycles():
    fs = 22050
    model = get_source_model("voice", fs=fs)
    for signal in (np.zeros(fs // 2), np.random.default_rng(0).standard_normal(fs // 2)):
        sequence = model.analyze(signal, fs=fs)
        assert sequence.n_units == 0
        assert not sequence.voiced.any()
        assert sequence.excitation.shape == (0, model.cycle_length)
        assert (sequence.noise_level == 1.0).all()
        assert (model.synthesize(sequence) == 0.0).all()
        # a PCA of *something else* must still accept an empty sequence
        pca = SourcePCA.fit(np.random.default_rng(1).standard_normal((8, model.cycle_length)),
                            n_components=2)
        assert model.encode(sequence, pca).shape == (0, 2)


def test_empty_audio_is_a_well_formed_empty_sequence():
    model = get_source_model("voice", fs=22050)
    sequence = model.analyze(np.zeros(0), fs=22050)
    assert sequence.n_frames == 0 and sequence.n_units == 0
    assert sequence.duration == 0.0
    assert model.synthesize(sequence).shape == (0,)


@pytest.mark.parametrize("f0", (55.0, 60.0, 110.0, 900.0, 1200.0, 1800.0))
def test_unusual_pitch_ranges_are_handled(f0):
    """A supplied track is authoritative in HMS, and it must win over the range."""
    fs = 22050
    model = get_source_model("voice", fs=fs)
    signal = synthetic_voice(f0=f0, fs=fs, duration=0.6)
    period = pulse_period(fs, f0)
    track = np.full(len(signal) // 110, fs / period)
    sequence = model.analyze(signal, fs=fs, f0=track)
    assert sequence.n_units > 10, f"no cycles at {f0} Hz"
    assert np.isfinite(sequence.excitation).all()
    assert np.median(sequence.unit_f0) == pytest.approx(fs / period, rel=0.03)
    assert (sequence.periods >= 4).all()
    # the epochs follow the supplied period (the first one may be snapped by up
    # to the refinement window, and a dropped cycle doubles one gap)
    spacing = np.diff(sequence.epochs)
    assert np.median(spacing) == pytest.approx(period, abs=2)
    assert (spacing >= 4).all()
    assert (spacing <= 2 * period + 2).all()


def test_the_estimator_covers_the_singing_range():
    """Without a track, ordinary singing pitches are estimated within a few %."""
    fs = 22050
    model = get_source_model("voice", fs=fs)
    for f0 in (110.0, 220.0, 440.0, 900.0):
        pitch = fs / pulse_period(fs, f0)
        signal = synthetic_voice(f0=f0, fs=fs, duration=0.6)
        sequence = model.analyze(signal, fs=fs)
        assert sequence.n_units > 10
        assert np.median(sequence.unit_f0) == pytest.approx(pitch, rel=0.05)


def test_the_default_estimator_matches_the_shared_one():
    """The default F0 path is the shared estimator, bit for bit.

    The source backend must not quietly change what the rest of HMS would have
    estimated: with ``f0_window_periods=0`` its track is exactly
    ``hms.core.dsp.autocorrelation_f0`` output, and the frame grid is unchanged
    even when the opt-in wide window is used.
    """
    from hms.core.dsp import autocorrelation_f0
    from hms.source import estimate_f0

    fs, hop = 22050, 110
    signal = synthetic_voice(f0=220.0, fs=fs, duration=0.6)
    track = estimate_f0(signal, fs, hop, 50.0, 2000.0, window_periods=0.0)
    assert np.array_equal(track, autocorrelation_f0(signal, fs, 4 * hop, hop, 50.0, 2000.0))
    # the wide window sees the same grid (plus the zero-padded tail it frames)
    wide = estimate_f0(signal, fs, hop, 50.0, 2000.0, window_periods=3.0)
    assert len(wide) >= len(track)
    assert np.median(wide[wide > 0]) == pytest.approx(220.0, rel=0.05)


def test_a_wide_f0_window_resolves_a_low_pitch():
    """The opt-in window: a low note under a strong low formant.

    The default ``4 * hop`` window locks onto the formant; widening the window
    to three periods of ``f0_floor`` is what makes the fundamental win.  This is
    the documented use of ``f0_window_periods`` (see docs/source_model.md).
    """
    fs, f0 = 22050, 60.0
    # a formant right at the frequency the short window mistakes for the pitch
    signal = synthetic_voice(f0=f0, fs=fs, duration=1.0,
                             formants=((735.0, 60.0), (1200.0, 90.0)))
    pitch = fs / pulse_period(fs, f0)
    narrow = get_source_model("voice", fs=fs).analyze(signal, fs=fs)
    wide = get_source_model("voice", fs=fs, f0_window_periods=3.0).analyze(signal, fs=fs)
    assert abs(np.median(narrow.unit_f0) - pitch) > 0.2 * pitch    # formant lock-in
    assert np.median(wide.unit_f0) == pytest.approx(pitch, rel=0.03)
    assert wide.n_frames == narrow.n_frames                        # same frame grid


def test_a_wild_f0_track_produces_valid_cycles_not_garbage():
    """Out-of-range estimates are clamped to a period, never to a broken vector."""
    fs = 22050
    signal = synthetic_voice(f0=250.0, fs=fs, duration=0.5)
    track = np.full(len(signal) // 110, 250.0)
    track[20:25] = 20000.0          # absurdly high
    track[30:35] = 1.0              # absurdly low
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs, f0=track)
    assert np.isfinite(sequence.excitation).all()
    assert np.allclose(np.sqrt((sequence.excitation ** 2).mean(axis=1)), 1.0)
    assert (sequence.periods >= 4).all()
    assert sequence.periods.max() <= int(np.ceil(fs / model.f0_floor)) + 1


def test_analysis_of_an_utterance_that_starts_and_ends_mid_period():
    fs = 22050
    signal = synthetic_voice(f0=180.0, fs=fs, duration=1.2)
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal[137:len(signal) - 91], fs=fs)
    assert sequence.n_units > 100
    for epoch, period in zip(sequence.epochs, sequence.periods):
        assert epoch >= 0
        assert epoch + period <= sequence.n_samples


# --------------------------------------------------------------------------
# Voice backend: the source round trip and compactness
# --------------------------------------------------------------------------


def test_synthesized_cycles_reproduce_the_residual():
    """cycles -> sample grid must give the residual back (no PCA involved)."""
    fs, f0 = 22050, 220.0
    signal = synthetic_voice(f0=f0, fs=fs, duration=0.6)
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs)
    reconstruction = model.synthesize(sequence)
    residual = whiten(signal, fs, model.default_frame_period)

    covered = np.zeros(len(signal), dtype=bool)
    for epoch, period in zip(sequence.epochs, sequence.periods):
        covered[epoch:min(len(signal), epoch + period)] = True
    assert covered.mean() > 0.9
    original, rebuilt = residual[covered], reconstruction[covered]
    correlation = np.corrcoef(original, rebuilt)[0, 1]
    assert correlation > 0.99
    assert np.linalg.norm(original - rebuilt) / np.linalg.norm(original) < 0.05


def test_a_few_coefficients_carry_the_excitation_shape_of_a_steady_note():
    """The Phase 1 success criterion, on a controlled steady note."""
    fs, f0 = 22050, 220.0
    signal = synthetic_voice(f0=f0, fs=fs, duration=1.5)
    period = pulse_period(fs, f0)
    # the exact track: a supplied track is authoritative, so the measured cycles
    # are one period long and the only variation is the excitation shape itself
    track = np.full(len(signal) // 110, fs / period)
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs, f0=track)
    cycles = sequence.excitation
    assert len(cycles) > 150
    assert (sequence.periods == period).mean() > 0.95

    cut = len(cycles) // 2
    train, test = cycles[:cut], cycles[cut:]
    errors = {}
    for k in (1, 4, 8, 16):
        pca = SourcePCA.fit(train, n_components=k)
        report = pca.report(test)
        assert report["n_vectors"] == len(test)
        errors[k] = report["relative_rmse"]
    # more components can only reduce the error (a little slack for the one
    # stray cycle the split can put on the wrong side)
    assert errors[1] > errors[4] > errors[8] - 0.01
    assert errors[8] >= errors[16] - 0.02
    assert errors[8] < 0.5

    pca = SourcePCA.fit(train, n_components=8)
    report = pca.report(test)
    assert pca.cumulative_explained_variance > 0.9
    assert report["relative_rmse"] < 0.5
    assert report["median_relative_error"] < 0.3
    assert report["mean_correlation"] > 0.9
    # 128 samples -> 8 numbers is the compression this phase is about
    assert pca.n_components * 16 == pca.cycle_length
    # the codec and the model agree on what a reconstruction is
    assert np.allclose(model.decode(model.encode(test, pca), pca), pca.reconstruct(test))


def test_encode_decode_with_a_basis_trained_on_the_sequence():
    fs = 22050
    signal = synthetic_voice(f0=200.0, fs=fs, duration=0.8)
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs)
    pca = SourcePCA.fit(sequence.excitation, n_components=8)
    coefficients = model.encode(sequence, pca)
    assert coefficients.shape == (sequence.n_units, 8)
    decoded = model.decode(coefficients, pca)
    assert decoded.shape == sequence.excitation.shape
    with_gain = model.decode(coefficients, pca, sequence.gains)
    assert with_gain.shape == decoded.shape
    assert np.allclose(with_gain, decoded * sequence.gains[:, None])
    assert not np.allclose(with_gain, decoded)
    with pytest.raises(ValueError):
        model.decode(coefficients, pca, sequence.gains[:-1])

    # synthesising from coefficients (rather than the stored vectors) works too
    from_coefficients = model.synthesize(sequence.with_coefficients(coefficients), pca=pca)
    assert from_coefficients.shape == (sequence.n_samples,)
    assert np.isfinite(from_coefficients).all()
    correlation = np.corrcoef(from_coefficients[500:5000],
                              model.synthesize(sequence)[500:5000])[0, 1]
    assert correlation > 0.9
    with pytest.raises(ValueError):
        model.synthesize(sequence, pca=pca)      # no coefficients on this one
    with pytest.raises(TypeError):
        model.synthesize(sequence.excitation)


# --------------------------------------------------------------------------
# The generic, pitch-free backend
# --------------------------------------------------------------------------


def test_generic_backend_needs_no_pitch_and_covers_every_frame():
    fs = 22050
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(fs // 2) * 0.1          # noise: no pitch at all
    model = get_source_model("residual", fs=fs)
    sequence = model.analyze(signal, fs=fs)
    assert sequence.backend == "residual"
    assert sequence.n_units == int(np.ceil(len(signal) / sequence.hop))
    assert np.isfinite(sequence.excitation).all()
    assert np.allclose(np.sqrt((sequence.excitation ** 2).mean(axis=1)), 1.0)
    assert (sequence.periods > 0).all()
    assert sequence.epochs[-1] + sequence.periods[-1] == len(signal)
    assert ((sequence.noise_level >= 0) & (sequence.noise_level <= 1)).all()


def test_generic_backend_round_trip_and_codec():
    fs = 22050
    signal = synthetic_voice(fs=fs, duration=0.5)
    model = get_source_model("residual", fs=fs)
    sequence = model.analyze(signal, fs=fs)
    reconstruction = model.synthesize(sequence)
    residual = whiten(signal, fs, model.default_frame_period, n_mcep=model.n_mcep)
    assert np.corrcoef(residual, reconstruction)[0, 1] > 0.99

    pca = SourcePCA.fit(sequence.excitation, n_components=8)
    coefficients = model.encode(sequence, pca)
    assert coefficients.shape == (sequence.n_units, 8)
    assert np.isfinite(model.decode(coefficients, pca)).all()


def test_generic_backend_keeps_an_optional_f0_track_aligned():
    fs = 22050
    signal = synthetic_voice(fs=fs, duration=0.4)
    model = get_source_model("residual", fs=fs)
    n_frames = int(np.ceil(len(signal) / model.hop(fs, 5.0)))
    track = np.full(n_frames + 10, 200.0)               # too long: clipped, not stretched
    track[:3] = 0.0
    sequence = model.analyze(signal, fs=fs, f0=track)
    assert len(sequence.f0) == n_frames
    assert sequence.f0[:3].sum() == 0.0
    assert sequence.voiced[3:].all()
    assert sequence.n_units == n_frames                 # f0 never changes the units

    short = model.analyze(signal, fs=fs, f0=np.full(5, 200.0))
    assert len(short.f0) == n_frames
    assert short.f0[5:].sum() == 0.0


def test_generic_backend_on_empty_audio():
    model = get_source_model("residual", fs=22050)
    sequence = model.analyze(np.zeros(0), fs=22050)
    assert sequence.n_units == 0 and sequence.n_frames == 0
    assert sequence.excitation.shape == (0, model.cycle_length)
    assert model.synthesize(sequence).shape == (0,)


# --------------------------------------------------------------------------
# Shared representation and registry
# --------------------------------------------------------------------------


def test_registry_and_backend_contract():
    assert set(SOURCE_BACKENDS) == {"voice", "residual"}
    assert all(available_backends().values())
    assert isinstance(get_source_model("voice"), SourceModel)
    assert isinstance(get_source_model("residual"), SourceModel)
    assert isinstance(source_model_for(None), SourceModel)
    assert source_model_for("residual").name == "residual"
    with pytest.raises(ValueError):
        get_source_model("glottal-pulse-2000")
    with pytest.raises(ValueError):
        get_source_model("voice", cycle_length=2)
    with pytest.raises(ValueError):
        get_source_model("voice", f0_floor=800.0, f0_ceil=200.0)
    with pytest.raises(ValueError):
        get_source_model("residual", unit_frames=0)


def test_sequence_validates_its_own_shapes():
    good = dict(f0=np.zeros(4), voiced=np.zeros(4, dtype=bool), noise_level=np.ones(4),
                excitation=np.zeros((3, 16)), gains=np.ones(3),
                epochs=np.arange(3) * 10, periods=np.full(3, 10),
                cycle_length=16, fs=22050, frame_period=5.0, n_samples=100)
    sequence = SourceSequence(**good)
    assert len(sequence) == 4 and sequence.n_frames == 4 and sequence.n_units == 3
    assert sequence.hop == 110
    assert sequence.unit_f0[0] == pytest.approx(2205.0)

    with pytest.raises(ValueError):
        SourceSequence(**{**good, "gains": np.ones(2)})
    with pytest.raises(ValueError):
        SourceSequence(**{**good, "voiced": np.zeros(3, dtype=bool)})
    with pytest.raises(ValueError):
        SourceSequence(**{**good, "excitation": np.zeros((3, 8))})
    with pytest.raises(ValueError):
        SourceSequence(**{**good, "coefficients": np.zeros((2, 4))})


def test_sequence_frames_view_and_coefficients():
    fs = 22050
    signal = synthetic_voice(fs=fs, duration=0.3)
    model = get_source_model("voice", fs=fs)
    sequence = model.analyze(signal, fs=fs)
    frames = list(sequence.frames())
    assert len(frames) == sequence.n_units
    assert all(isinstance(frame, SourceFrame) for frame in frames)
    frame = frames[0]
    assert frame.excitation.shape == (sequence.cycle_length,)
    assert frame.is_valid
    assert frame.voiced and frame.f0 > 0
    assert frame.gain == sequence.gains[0]
    assert frame.epoch == sequence.epochs[0]
    assert frame.period == sequence.periods[0]
    assert frame.index == 0
    assert 0.0 <= frame.noise_level <= 1.0
    assert frame.source_coefficients is None

    with_coefficients = sequence.with_coefficients(np.zeros((sequence.n_units, 3)))
    assert with_coefficients.coefficients.shape == (sequence.n_units, 3)
    assert sequence.coefficients is None                  # the original is untouched
    assert next(with_coefficients.frames()).source_coefficients.shape == (3,)
    assert not SourceFrame().is_valid
    assert not SourceFrame(excitation=np.full(4, np.nan)).is_valid
