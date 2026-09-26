"""Vocoder backends: the contract every backend must honour.

The engine only ever asks a vocoder for three things -- analyse a signal into
(f0, sp, ap), give those back as samples, and describe the frame geometry -- so
these tests check exactly that, for whichever backend is installed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.features import AcousticFrameSequence
from hms.data import wavio
from hms.vocoder import BACKENDS, available_backends, get_vocoder
from hms.vocoder.base import VocoderUnavailable, limit_peak

FRAME_PERIOD = 5.0


def native_vocoder():
    try:
        return get_vocoder("native")
    except VocoderUnavailable as exc:            # pragma: no cover - env
        pytest.skip(f"native WORLD backend unavailable: {exc}")


def builtin_vocoder():
    return get_vocoder("builtin")


def test_backend_registry():
    assert BACKENDS == ("auto", "pyworld", "native", "builtin")
    availability = available_backends()
    assert set(availability) == {"pyworld", "native", "builtin"}
    assert availability["builtin"] is True
    with pytest.raises(ValueError):
        get_vocoder("nonsense")


def test_missing_pyworld_is_reported_not_crashed():
    try:
        get_vocoder("pyworld")
    except VocoderUnavailable as exc:
        assert "pyworld" in str(exc).lower()
    else:                                        # pragma: no cover - env
        pytest.skip("pyworld is installed here")


def test_analysis_geometry(example_wav_path):
    vocoder = native_vocoder()
    signal, fs = wavio.read_wav(example_wav_path)
    sequence = vocoder.analyze_to_sequence(signal, fs,
                                           frame_period=FRAME_PERIOD)
    assert isinstance(sequence, AcousticFrameSequence)
    n_frames = len(sequence)
    assert n_frames > 10
    bins = vocoder.fft_size_for(fs) // 2 + 1      # rate-dependent for WORLD
    assert sequence.f0.shape == (n_frames,)
    assert sequence.sp.shape == (n_frames, bins)
    assert sequence.ap.shape == (n_frames, bins)
    assert np.isfinite(sequence.sp).all() and (sequence.sp >= 0).all()
    assert np.isfinite(sequence.ap).all()
    assert (sequence.ap >= 0).all() and (sequence.ap <= 1.0).all()
    # the demo phrases contain vowels, so some frames must be voiced
    assert sequence.voiced.any()
    assert sequence.duration == pytest.approx(n_frames * FRAME_PERIOD / 1000.0)


def test_synthesis_length_matches_the_analysis(example_wav_path):
    vocoder = native_vocoder()
    signal, fs = wavio.read_wav(example_wav_path)
    sequence = vocoder.analyze_to_sequence(signal, fs,
                                           frame_period=FRAME_PERIOD)
    audio = vocoder.synthesize(sequence)
    assert audio.ndim == 1
    assert np.isfinite(audio).all()
    # within one frame plus the synthesis window of the source length
    assert abs(len(audio) - len(signal)) < 2 * vocoder.fft_size


def test_analysis_synthesis_roundtrip_preserves_pitch(example_wav_path):
    """The pitch we put in must come back out of a real WORLD round trip."""
    vocoder = native_vocoder()
    signal, fs = wavio.read_wav(example_wav_path)
    original = vocoder.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    audio = vocoder.synthesize(original)
    roundtrip = vocoder.analyze_to_sequence(audio, fs, frame_period=FRAME_PERIOD)

    both = original.voiced & roundtrip.voiced
    assert both.sum() > 10
    ratio = roundtrip.f0[both] / original.f0[both]
    assert np.median(ratio) == pytest.approx(1.0, rel=0.02)

    # the spectral envelope must survive as a shape, even though WORLD's
    # analysis/synthesis gain is not exactly unity (it depends on how the
    # source harmonics line up with the envelope, not on a fixed factor)
    log_in = np.log(np.maximum(original.sp[both].mean(axis=0), 1e-30))
    log_out = np.log(np.maximum(roundtrip.sp[both].mean(axis=0), 1e-30))
    log_in, log_out = log_in - log_in.mean(), log_out - log_out.mean()
    correlation = np.corrcoef(log_in, log_out)[0, 1]
    assert correlation > 0.8
    assert np.abs(log_out - log_in).mean() < 1.5


def test_native_synthesize_matches_the_raw_c_call_sample_for_sample():
    """The class wrapper must be transparent: no hidden, peak-dependent gain.

    Regression test for an RMS mismatch between the two paths (0.526 raw vs
    0.199 through the class on identical parameters).  The difference was a
    peak normalisation applied only on the class path, which also made an
    utterance's level depend on its own peak.  Identical parameters must now
    come back identical, including when WORLD's output exceeds +/-1.
    """
    vocoder = native_vocoder()
    fs = 22050
    fft_size = vocoder.fft_size_for(fs)
    bins = fft_size // 2 + 1
    n_frames = 80
    f0 = np.full(n_frames, 233.0)
    freqs = np.linspace(0.0, fs / 2.0, bins)
    envelope = 1.0 / (1.0 + (freqs / 900.0) ** 2) + 1e-4
    # a loud, very periodic vowel: WORLD's synthesis overshoots 1.0 here, which
    # is exactly the case the old code rescaled
    sp = np.repeat(envelope[None, :], n_frames, axis=0) * 400.0
    ap = np.full((n_frames, bins), 0.05)          # already inside [1e-4, 1]
    sequence = AcousticFrameSequence(f0=f0, sp=sp, ap=ap, frame_period=FRAME_PERIOD,
                                     fs=fs, fft_size=fft_size)

    reference = np.zeros(int(vocoder._lib.hms_synth_length(
        n_frames, fs, FRAME_PERIOD)))
    vocoder._lib.hms_synthesize(
        vocoder._ptr(np.ascontiguousarray(f0)), n_frames,
        vocoder._ptr(np.ascontiguousarray(sp)),
        vocoder._ptr(np.ascontiguousarray(ap)),
        fft_size, FRAME_PERIOD, fs, len(reference),
        vocoder._ptr(reference))
    assert np.abs(reference).max() > 1.0, "test needs a peak above full scale"

    audio = vocoder.synthesize(sequence)
    assert len(audio) == len(reference)
    assert np.array_equal(audio, reference), (
        "class path differs from the raw C call: max |diff| "
        f"{np.abs(audio - reference).max():.3e}")


def test_unvoiced_frames_stay_unvoiced(example_wav_path):
    """All-breath parameters must render (near) silence, not a pulse train.

    The backend returns its synthesis verbatim, so the level check goes through
    `limit_peak` -- the shared headroom helper -- instead of a hidden gain
    inside the backend (which would also have rescaled this near-silent render
    because of one onset spike).
    """
    vocoder = native_vocoder()
    signal, fs = wavio.read_wav(example_wav_path)
    sequence = vocoder.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    sequence.f0[:] = 0.0                       # pretend it is all breath
    audio = vocoder.synthesize(sequence)
    assert np.isfinite(audio).all()
    assert np.abs(limit_peak(audio)).max() <= 1.0
    # it is a *noise* excitation, not a pulse train: no periodicity, so the
    # normalised autocorrelation must stay low at every plausible pitch lag
    middle = audio[len(audio) // 3: len(audio) // 3 + 8192]
    middle = middle - middle.mean()
    ac = np.correlate(middle, middle, mode="full")[len(middle) - 1:]
    ac = ac / (ac[0] + 1e-30)
    lo, hi = int(fs / 1000.0), int(fs / 50.0)
    assert ac[lo:hi].max() < 0.5, "unvoiced frames rendered a periodic signal"


def test_flat_pitch_track_is_synthesised_at_the_requested_frequency():
    """A hand-built 220 Hz vowel: the backend must render it, not a copy."""
    vocoder = native_vocoder()
    fs = 22050
    fft_size = vocoder.fft_size_for(fs)
    n_frames = 100
    frame_period = FRAME_PERIOD
    rng = np.random.default_rng(0)
    bins = fft_size // 2 + 1
    f0 = np.full(n_frames, 220.0)
    # a simple falling envelope, and moderate aperiodicity everywhere
    freqs = np.linspace(0, fs / 2, bins)
    envelope = 1.0 / (1.0 + (freqs / 1500.0) ** 2) + 1e-4
    sp = np.repeat(envelope[None, :], n_frames, axis=0)
    sp *= np.exp(rng.normal(0, 0.02, size=sp.shape))
    ap = np.full((n_frames, bins), 0.2)
    sequence = AcousticFrameSequence(f0=f0, sp=sp, ap=ap,
                                     frame_period=frame_period, fs=fs,
                                     fft_size=fft_size)
    audio = vocoder.synthesize(sequence)
    assert len(audio) > 0.4 * fs
    analysis = vocoder.analyze_to_sequence(audio, fs, frame_period=frame_period)
    voiced = analysis.f0 > 0
    assert voiced.mean() > 0.5
    assert np.median(analysis.f0[voiced]) == pytest.approx(220.0, rel=0.03)


def test_builtin_backend_is_a_real_fallback():
    """The pure-numpy backend must produce something usable, not silence."""
    vocoder = builtin_vocoder()
    fs = 22050
    duration = 0.4
    t = np.arange(int(duration * fs)) / fs
    # a synthetic 220 Hz vowel-ish tone with harmonics
    signal = sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 6))
    signal = 0.3 * signal / np.abs(signal).max()

    sequence = vocoder.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    assert len(sequence) > 5
    assert sequence.sp.shape[1] == vocoder.n_bins
    voiced = sequence.f0 > 0
    if voiced.any():                             # the F0 estimate may need an
        assert np.median(sequence.f0[voiced]) == pytest.approx(220.0, rel=0.1)
    audio = vocoder.synthesize(sequence)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() > 0.01
    assert np.sqrt((audio ** 2).mean()) > 0.001


def test_builtin_handles_an_unvoiced_signal():
    vocoder = builtin_vocoder()
    fs = 22050
    rng = np.random.default_rng(0)
    noise = 0.05 * rng.normal(size=int(0.3 * fs))
    sequence = vocoder.analyze_to_sequence(noise, fs, frame_period=FRAME_PERIOD)
    assert not sequence.voiced.any()
    audio = vocoder.synthesize(sequence)
    assert np.isfinite(audio).all()


def test_frame_geometry_helpers(example_wav_path):
    vocoder = native_vocoder()
    assert vocoder.n_bins == vocoder.fft_size // 2 + 1
    assert repr(vocoder)
    signal, fs = wavio.read_wav(example_wav_path)
    sequence = vocoder.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    assert sequence.fft_size == vocoder.fft_size_for(fs)
    assert sequence.fs == fs


def test_params_files_roundtrip(tmp_path, example_wav_path):
    vocoder = native_vocoder()
    signal, fs = wavio.read_wav(example_wav_path)
    sequence = vocoder.analyze_to_sequence(signal, fs, frame_period=FRAME_PERIOD)
    path = tmp_path / "params.npz"
    wavio.save_params(path, sequence)
    loaded = wavio.load_params(path)
    assert np.allclose(loaded.f0, sequence.f0)
    assert loaded.sp.shape == sequence.sp.shape
    assert loaded.frame_period == sequence.frame_period
    assert loaded.fs == sequence.fs
