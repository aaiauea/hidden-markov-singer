"""MLSA vocoder: formulation, contract and edge cases.

The MLSA backend shares every parameter convention with the other backends, so
the generic contract lives in `test_vocoder.py`.  What is specific to MLSA is
the filter itself (the warped log-spectrum exponential, which must reproduce the
project's own mel-cepstrum reconstruction), the mixed excitation, and the
guarantees synthesis has to keep on degenerate input (silence, one frame,
malformed arrays, non-contiguous views).
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.features import (AcousticFrameSequence, DEFAULT_N_MCEP, idct2,
                               mel_band_count, power_to_mcep)
from hms.data import wavio
from hms.vocoder import get_vocoder
from hms.vocoder.mlsa import (FILTER_PERIODS, LOG_POWER_CLIP, MLSAVocoder,
                              mlsa_log_amplitude, mlsa_response,
                              mel_warping_factor)

FS = 22050
FRAME_PERIOD = 5.0
FFT_SIZE = 1024
BINS = FFT_SIZE // 2 + 1


def vocoder(**kwargs) -> MLSAVocoder:
    kwargs.setdefault("fs", FS)
    kwargs.setdefault("frame_period", FRAME_PERIOD)
    return MLSAVocoder(**kwargs)


def sequence(f0, sp=None, ap=None, fs=FS, fft_size=FFT_SIZE) -> AcousticFrameSequence:
    """A frame sequence with sensible envelopes for the frames given in f0."""
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    n = len(f0)
    bins = fft_size // 2 + 1
    if sp is None:
        sp = np.full((n, bins), 1e-3)
    if ap is None:
        ap = np.full((n, bins), 0.3)
    return AcousticFrameSequence(f0=f0, sp=np.asarray(sp, dtype=np.float64),
                                 ap=np.asarray(ap, dtype=np.float64),
                                 frame_period=FRAME_PERIOD, fs=fs,
                                 fft_size=fft_size)


def envelope(power: float, bins: int = BINS) -> np.ndarray:
    """A smooth vowel-like envelope (formants at ~700/1500/2500 Hz)."""
    freqs = np.linspace(0.0, FS / 2.0, bins)
    shape = sum(np.exp(-0.5 * ((freqs - centre) / width) ** 2)
                for centre, width in ((700.0, 150.0), (1500.0, 250.0),
                                      (2500.0, 400.0)))
    return power * (0.02 + shape)


# --------------------------------------------------------------------------
# The filter formulation
# --------------------------------------------------------------------------

def test_warping_factor_is_mel_like_and_sample_rate_dependent():
    factors = [mel_warping_factor(fs) for fs in (8000, 16000, 22050, 44100,
                                                 48000)]
    assert all(0.0 < factor < 0.9 for factor in factors)
    # a wider band needs more warping to put the mel scale on the all-pass axis
    assert factors == sorted(factors)
    assert factors[0] == pytest.approx(0.36, abs=0.03)     # 8 kHz
    assert factors[-1] == pytest.approx(0.60, abs=0.03)    # 48 kHz
    assert mel_warping_factor(22050) == mel_warping_factor(22050)  # cached


def test_filter_reproduces_the_projects_own_cepstral_reconstruction():
    """The MLSA response must *be* the mel-cepstrum HMS models.

    ``mcep_to_power`` reconstructs a curve by interpolating between mel band
    centres; the MLSA filter evaluates the same coefficients as a Fourier
    series on the warped axis.  At the analysis knots the two have to agree to
    machine precision -- that is the contract that makes the vocoder consume
    the acoustic representation instead of inventing a parallel one.
    """
    rng = np.random.default_rng(0)
    sp = np.abs(rng.normal(size=(4, BINS))) * 0.1 + 1e-4
    mcep = power_to_mcep(sp, FFT_SIZE, FS, DEFAULT_N_MCEP)
    amplitudes = mlsa_log_amplitude(sp, FFT_SIZE, FS, DEFAULT_N_MCEP)

    n_bands = mel_band_count(DEFAULT_N_MCEP)
    padded = np.zeros((sp.shape[0], n_bands))
    padded[:, :DEFAULT_N_MCEP] = mcep
    knots = idct2(padded)                       # log power at the knots
    beta = np.pi * (np.arange(n_bands) + 0.5) / n_bands
    evaluated = 2.0 * (amplitudes @ np.cos(
        np.outer(np.arange(DEFAULT_N_MCEP), beta)))
    assert np.abs(evaluated - knots).max() < 1e-9
    # and the amplitude coefficients are half the log-power ones
    assert np.allclose(2.0 * amplitudes[:, 0] * np.sqrt(n_bands),
                       mcep[:, 0])


def test_response_is_finite_and_clipped_for_malformed_envelopes():
    ceiling = np.exp(0.5 * LOG_POWER_CLIP)
    response = mlsa_response(np.full((2, DEFAULT_N_MCEP), 1e6), 512, 0.42)
    assert np.isfinite(response).all()
    assert np.abs(response).max() <= ceiling * (1.0 + 1e-12)
    quiet = mlsa_response(np.full((2, DEFAULT_N_MCEP), -1e6), 512, 0.42)
    assert np.isfinite(quiet).all()
    assert np.abs(quiet).max() <= ceiling * (1.0 + 1e-12)


def test_response_is_conjugate_symmetric_so_the_impulse_response_is_real():
    sp = np.abs(np.random.default_rng(1).normal(size=(1, BINS))) + 0.1
    amplitudes = mlsa_log_amplitude(sp, FFT_SIZE, FS, DEFAULT_N_MCEP)
    for n_fft in (256, 512, 1024):
        response = mlsa_response(amplitudes, n_fft, 0.42)
        impulse = np.fft.irfft(response, n_fft, axis=1)
        assert np.isfinite(impulse).all()
        # a real impulse response round-trips to (almost) the same response
        again = np.fft.rfft(impulse, n_fft, axis=1)
        assert np.abs(again - response).max() < 1e-9


def test_default_filter_length_truncation_is_negligible():
    """The documented FIR cut-off (FILTER_PERIODS frame hops) must not be audible.

    The response decays like ``alpha**n``; the default length is two frame
    periods.  Synthesising the same parameters with a four times longer filter
    has to agree far below the noise floor of the backend.
    """
    sp = np.tile(envelope(1e-2), (40, 1))
    ap = np.tile(0.4 * np.ones(BINS), (40, 1))
    f0 = np.full(40, 220.0)
    hop = int(round(FS * FRAME_PERIOD / 1000.0))
    default = vocoder(filter_length=None).synthesize(sequence(f0, sp, ap))
    long_filter = vocoder(filter_length=4 * FILTER_PERIODS * hop).synthesize(
        sequence(f0, sp, ap))
    error = np.sqrt(np.mean((default - long_filter) ** 2))
    assert error < 1e-4 * np.sqrt(np.mean(long_filter ** 2))


# --------------------------------------------------------------------------
# Determinism and the excitation
# --------------------------------------------------------------------------

def test_synthesis_is_deterministic_and_seed_controls_only_the_noise():
    seq = sequence(np.full(30, 180.0), np.tile(envelope(1e-2), (30, 1)),
                   np.tile(0.5 * np.ones(BINS), (30, 1)))
    first = vocoder().synthesize(seq)
    assert np.array_equal(first, vocoder().synthesize(seq))
    other = vocoder(seed=7).synthesize(seq)
    assert not np.array_equal(first, other)
    assert len(first) == len(other)


@pytest.mark.parametrize("f0", [100.0, 200.0, 400.0, 800.0])
def test_pulse_train_has_unit_mean_square_at_any_f0(f0):
    """The excitation level must not follow the note.

    Without the per-pulse ``sqrt(period)`` amplitude a higher note would place
    more (equally loud) pulses and the output level would rise with F0.
    """
    v = vocoder()
    samples = 100 * 110
    train = v._pulse_train(np.full(100, f0), FS, 110, samples + 512, samples)
    # every placed pulse carries one period's energy; the last whole period
    # that fits may fall short, so the mean square is 1 to within one period
    assert np.sqrt((train[:samples] ** 2).mean()) == pytest.approx(1.0,
                                                                   rel=0.03)
    assert np.count_nonzero(train) > 0
    # ... and the pulse energy sits at one period spacing, not between pulses
    length = 8192
    segment = train[:length]
    acf = np.correlate(segment, segment, "full")[length - 1:]
    period = int(round(FS / f0))
    around_period = float(np.max(acf[period - 2: period + 3]))
    around_half = float(np.max(np.abs(acf[period // 2 - 2: period // 2 + 3])))
    assert around_period > 4.0 * around_half


def test_voiced_tone_is_periodic_at_the_requested_pitch():
    seq = sequence(np.full(120, 220.0), np.tile(envelope(1e-2), (120, 1)),
                   np.tile(0.05 * np.ones(BINS), (120, 1)))
    audio = vocoder().synthesize(seq)
    assert np.isfinite(audio).all()
    # normalised autocorrelation at multiples of the 220 Hz period
    middle = audio[FS // 10: FS // 10 + 4 * FS // 10]
    acf = np.correlate(middle, middle, "full")[len(middle) - 1:]
    period = FS / 220.0
    lags = [int(round(k * period)) for k in range(1, 6)]
    ratio = [acf[lag] / acf[0] for lag in lags]
    assert min(ratio) > 0.6                      # strongly periodic
    assert acf[lags[0]] > 0.6 * acf[int(round(0.5 * period))]
    # the harmonic structure sits at multiples of 220 Hz
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle)))) ** 2
    freqs = np.fft.rfftfreq(len(middle), 1.0 / FS)
    harmonics = sum(float(spectrum[np.argmin(np.abs(freqs - k * 220.0))])
                    for k in range(1, 12))
    between = sum(float(spectrum[np.argmin(np.abs(freqs - (k + 0.5) * 220.0))])
                  for k in range(1, 12))
    assert harmonics > 20 * between


def test_unvoiced_frames_are_noise_shaped_by_the_envelope():
    n = 200
    ap = np.ones((n, BINS))                      # fully aperiodic
    low = np.zeros((n, BINS)); low[:, :64] = 1.0
    high = np.zeros((n, BINS)); high[:, -128:] = 1.0
    y_low = vocoder().synthesize(sequence(np.zeros(n), low, ap))
    y_high = vocoder().synthesize(sequence(np.zeros(n), high, ap))
    assert np.isfinite(y_low).all() and np.isfinite(y_high).all()
    assert np.sqrt((y_low ** 2).mean()) > 1e-3   # not silence
    assert np.sqrt((y_high ** 2).mean()) > 1e-3

    def centroid(y):
        spectrum = np.abs(np.fft.rfft(y[:4096])) ** 2
        return float((np.arange(len(spectrum)) * spectrum).sum()
                     / spectrum.sum())

    assert centroid(y_high) > 5 * centroid(y_low)


def test_aperiodicity_mixes_pulse_and_noise_per_frequency_band():
    """``ap`` is a per-bin ratio, not a per-frame scalar.

    A frame that is periodic below 2 kHz and noise above it must render with a
    clean low band and a noisy high band; a backend that averages ``ap`` over
    the frame (as the builtin fallback's excitation does) loses that structure.
    """
    n = 200
    f0 = np.full(n, 200.0)
    freqs = np.linspace(0.0, FS / 2.0, BINS)
    ap = np.where(freqs < 2000.0, 0.02, 0.98)
    sp = np.full((n, BINS), 1e-2)
    audio = vocoder().synthesize(sequence(f0, sp, np.tile(ap, (n, 1))))

    segment = audio[FS // 10: FS // 10 + 8192]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    frequencies = np.fft.rfftfreq(len(segment), 1.0 / FS)

    def harmonic_to_noise(low, high):
        band = (frequencies >= low) & (frequencies < high)
        harmonic = np.zeros_like(band)
        for k in range(1, int(high / 200.0) + 2):
            harmonic |= np.abs(frequencies - k * 200.0) < 25.0
        return 10.0 * np.log10(spectrum[band & harmonic].mean()
                               / max(spectrum[band & ~harmonic].mean(), 1e-30))

    low_band = harmonic_to_noise(200.0, 1800.0)
    high_band = harmonic_to_noise(2500.0, 7000.0)
    assert low_band > 12.0                     # the pulse dominates below 2 kHz
    assert high_band < 8.0                     # the noise dominates above it
    assert low_band - high_band > 8.0


def test_an_unvoiced_frame_is_pure_noise_whatever_the_aperiodicity():
    """``ap`` only mixes the two excitations on voiced frames."""
    n = 60
    sp = np.tile(envelope(1e-2), (n, 1))
    quiet = sequence(np.zeros(n), sp, np.full((n, BINS), 0.001))
    loud = sequence(np.zeros(n), sp, np.full((n, BINS), 0.999))
    assert np.array_equal(vocoder().synthesize(quiet),
                          vocoder().synthesize(loud))


def test_output_level_is_tied_to_the_envelope_not_the_pitch():
    """Unit-mean-square pulses keep the loudness constant across the range."""
    levels = []
    for f0 in (100.0, 200.0, 400.0, 800.0):
        seq = sequence(np.full(200, f0), np.tile(envelope(1e-2), (200, 1)),
                       np.tile(0.2 * np.ones(BINS), (200, 1)))
        levels.append(np.sqrt((vocoder().synthesize(seq) ** 2).mean()))
    levels = np.asarray(levels)
    assert levels.min() / levels.max() > 0.7     # within ~3 dB across 3 octaves
    assert np.allclose(levels, levels.mean(), rtol=0.3)


def test_changing_f0_and_spectral_coefficients_stay_finite():
    frames = 200
    glide = np.linspace(120.0, 520.0, frames)
    step = np.concatenate([np.full(frames // 2, 180.0),
                           np.full(frames - frames // 2, 440.0)])
    envelope_a = np.tile(envelope(1e-2), (frames, 1))
    envelope_b = np.tile(envelope(1e-4) * np.linspace(1.0, 8.0, BINS),
                         (frames, 1))
    for f0 in (glide, step):
        for sp in (envelope_a, envelope_b):
            audio = vocoder().synthesize(sequence(f0, sp))
            assert np.isfinite(audio).all()
            assert np.abs(audio).max() > 1e-6
            assert len(audio) == int(frames * FRAME_PERIOD / 1000.0 * FS)


def test_changing_spectral_coefficients_change_the_timbre():
    frames = 120
    f0 = np.full(frames, 200.0)
    sp_a = np.tile(envelope(1e-2), (frames, 1))
    sp_b = np.tile(envelope(1e-4), (frames, 1))
    y_a = vocoder().synthesize(sequence(f0, sp_a))
    y_b = vocoder().synthesize(sequence(f0, sp_b))
    assert not np.allclose(y_a, y_b, atol=1e-6)
    # the two envelopes differ by 10x in level, and so must the outputs
    assert np.sqrt((y_a ** 2).mean()) > 3 * np.sqrt((y_b ** 2).mean())


# --------------------------------------------------------------------------
# Degenerate and hostile input
# --------------------------------------------------------------------------

def test_silence_renders_silence():
    n = 50
    audio = vocoder().synthesize(sequence(
        np.zeros(n), np.zeros((n, BINS)), np.ones((n, BINS))))
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() < 1e-3


@pytest.mark.parametrize("n_frames", [0, 1, 2, 3])
def test_short_utterances(n_frames):
    expected = int(n_frames * FRAME_PERIOD / 1000.0 * FS)
    audio = vocoder().synthesize(sequence(np.full(n_frames, 200.0)))
    # no frames -> no samples, matching the builtin backend and WORLD
    assert len(audio) == expected
    assert np.isfinite(audio).all()
    if n_frames:
        assert np.sqrt((audio ** 2).mean()) > 1e-6


def test_non_contiguous_inputs_match_their_contiguous_copies():
    n = 120
    f0 = np.linspace(150.0, 350.0, 2 * n)
    sp = np.tile(envelope(1e-2), (2 * n, 1))
    ap = np.tile(0.3 * np.ones(BINS), (2 * n, 1))
    v = vocoder()
    view = AcousticFrameSequence(f0=f0[::2], sp=sp[::2], ap=ap[::2],
                                 frame_period=FRAME_PERIOD, fs=FS,
                                 fft_size=FFT_SIZE)
    assert not view.sp.flags["C_CONTIGUOUS"]
    copy = AcousticFrameSequence(f0=f0[::2].copy(), sp=sp[::2].copy(),
                                 ap=ap[::2].copy(), frame_period=FRAME_PERIOD,
                                 fs=FS, fft_size=FFT_SIZE)
    assert np.array_equal(v.synthesize(view), v.synthesize(copy))
    # ... and the caller's arrays are never written to
    assert (f0[::2] == view.f0).all() and (sp[::2] == view.sp).all()


def test_frame_count_mismatch_is_rejected_with_a_clear_error():
    seq = sequence(np.full(10, 200.0), np.full((7, BINS), 1e-3),
                   np.full((7, BINS), 0.3))
    with pytest.raises(ValueError, match="same frames"):
        vocoder().synthesize(seq)


def test_malformed_parameters_do_not_produce_nan_or_inf():
    n = 12
    f0 = np.array([np.nan, np.inf, -np.inf, -4.0] + [200.0] * (n - 4))
    sp = np.full((n, BINS), np.nan)
    sp[4] = np.inf
    sp[5] = -1.0
    ap = np.full((n, BINS), np.nan)
    ap[6] = np.inf
    ap[7] = 2.0
    audio = vocoder().synthesize(sequence(f0, sp, ap))
    assert np.isfinite(audio).all()
    # absurd envelopes are clipped by the documented log-power guard
    huge = vocoder().synthesize(sequence(np.full(8, 200.0),
                                         np.full((8, BINS), 1e30),
                                         np.full((8, BINS), 0.5)))
    assert np.isfinite(huge).all()
    assert np.abs(huge).max() > 0.0


def test_f0_at_or_above_nyquist_does_not_hang_or_explode():
    audio = vocoder().synthesize(sequence(np.full(20, 0.5 * FS)))
    assert np.isfinite(audio).all()
    assert len(audio) == int(20 * FRAME_PERIOD / 1000.0 * FS)


def test_analysis_is_shared_with_the_builtin_backend(example_wav_path):
    """`--vocoder mlsa` must be usable for extract/train, too."""
    signal, fs = wavio.read_wav(example_wav_path)
    mlsa = get_vocoder("mlsa", fs=fs, frame_period=FRAME_PERIOD)
    builtin = get_vocoder("builtin", fs=fs, frame_period=FRAME_PERIOD)
    a = mlsa.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    b = builtin.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    assert np.array_equal(a.f0, b.f0)
    assert np.array_equal(a.sp, b.sp)
    assert np.array_equal(a.ap, b.ap)
    assert a.fft_size == b.fft_size


# --------------------------------------------------------------------------
# Integration through the existing selection interface
# --------------------------------------------------------------------------

def test_vocoder_selection_and_config_validation():
    from hms.core.synthesizer import SynthesisConfig
    from hms.core.trainer import TrainingConfig

    assert get_vocoder("mlsa").name == "mlsa"
    assert SynthesisConfig(vocoder="mlsa").vocoder == "mlsa"
    assert TrainingConfig(vocoder="mlsa").vocoder == "mlsa"
    with pytest.raises(ValueError):
        SynthesisConfig(vocoder="nonsense")
    with pytest.raises(ValueError):
        TrainingConfig(vocoder="nonsense")
    # `auto` keeps resolving to the pre-MLSA precedence
    auto = get_vocoder("auto")
    assert auto.name in ("pyworld", "native", "builtin")


def test_end_to_end_synthesis_through_a_trained_model(trained_model, short_score):
    """The model's mel-cepstra reach the filter through the normal pipeline."""
    from hms.core.synthesizer import Synthesizer, SynthesisConfig

    config = SynthesisConfig(seed=0, vibrato=False, vocoder="mlsa")
    result = Synthesizer(trained_model, config).synthesize(short_score)
    assert np.isfinite(result.audio).all()
    expected = int(len(result.params.f0) * result.params.frame_period
                   / 1000.0 * result.params.fs)
    assert len(result.audio) == expected
    assert np.abs(result.audio).max() > 1e-3
    # deterministic for the same model, score and seed
    again = Synthesizer(trained_model, SynthesisConfig(seed=0, vibrato=False,
                                                       vocoder="mlsa")) \
        .synthesize(short_score)
    assert np.array_equal(result.audio, again.audio)
    # ... and the builtin backend is untouched by the new code path
    other = Synthesizer(trained_model, SynthesisConfig(seed=0, vibrato=False,
                                                       vocoder="builtin")) \
        .synthesize(short_score)
    assert not np.array_equal(result.audio, other.audio)


def test_synthesised_audio_reanalyses_to_the_requested_pitch(example_wav_path):
    """A real WORLD analysis of an MLSA render must find the pitch back."""
    from hms.vocoder.base import VocoderUnavailable

    try:
        world = get_vocoder("native")
    except VocoderUnavailable:                       # pragma: no cover - env
        pytest.skip("native WORLD backend unavailable")
    signal, fs = wavio.read_wav(example_wav_path)
    original = world.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    vocab = get_vocoder("mlsa", fs=fs, frame_period=FRAME_PERIOD)
    rendered = vocab.synthesize(original)
    analysis = world.analyze_to_sequence(rendered, fs, frame_period=FRAME_PERIOD)

    both = original.voiced & analysis.voiced
    assert both.sum() > 10
    ratio = analysis.f0[both] / original.f0[both]
    assert np.median(ratio) == pytest.approx(1.0, rel=0.02)

    log_in = np.log(np.maximum(original.sp[both].mean(axis=0), 1e-30))
    log_out = np.log(np.maximum(analysis.sp[both].mean(axis=0), 1e-30))
    log_in -= log_in.mean()
    log_out -= log_out.mean()
    assert np.corrcoef(log_in, log_out)[0, 1] > 0.9
    assert np.abs(log_out - log_in).mean() < 2.0


def test_mlsa_does_not_change_the_builtin_backend():
    """The existing backend's code path and output are untouched."""
    seq = sequence(np.full(30, 220.0), np.tile(envelope(1e-2), (30, 1)))
    audio = get_vocoder("builtin").synthesize(seq)
    assert np.isfinite(audio).all()
    # builtin still applies its documented peak policy, MLSA does not
    assert np.abs(audio).max() <= 0.99 + 1e-9
    assert get_vocoder("builtin").name == "builtin"
