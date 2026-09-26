"""Feature extraction / WORLD parameter mapping."""

from __future__ import annotations

import numpy as np
import pytest

from hms.core import labels as labels_module
from hms.core.features import (DEFAULT_N_MCEP, AcousticFrameSequence,
                               FeatureSpec, add_dynamic_features,
                               aperiodicity_to_bands, bands_to_aperiodicity,
                               dct2, hz_to_mel, hz_to_semitone, idct2,
                               mel_band_count, mel_filterbank, mel_to_hz,
                               mcep_to_power, power_to_logmel, power_to_mcep,
                               remove_dynamic_features, semitone_to_hz,
                               split_streams)
from hms.core.generation import stack_streams, unstack_streams


def test_mel_scale_roundtrip():
    frequencies = np.array([0.0, 100.0, 440.0, 1000.0, 8000.0, 20000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(frequencies)), frequencies, rtol=1e-9)


def test_dct_is_exactly_invertible():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 17))
    assert np.allclose(idct2(dct2(x)), x, atol=1e-10)


def test_mel_filterbank_normalised_and_ordered():
    bank = mel_filterbank(20, 1024, 16000)
    assert bank.shape == (20, 513)
    assert (bank >= 0).all()
    # every filter has some weight, and filters sit at increasing frequencies
    assert (bank.sum(axis=1) > 0).all()
    centres = [np.argmax(row) for row in bank]
    assert centres == sorted(centres)


@pytest.mark.parametrize("n_mcep", [12, 30])
def test_spectral_envelope_roundtrip_is_accurate(n_mcep):
    """mel-cepstrum -> power -> mel-cepstrum must be faithful (gain included)."""
    rng = np.random.default_rng(1)
    fft_size, fs = 1024, 22050
    bins = fft_size // 2 + 1
    freqs = np.linspace(0, fs / 2, bins)
    # a smooth formant-like envelope plus mild ripple
    envelope = (1.0 / (1 + ((freqs - 700) / 200) ** 2)
                + 0.4 / (1 + ((freqs - 1800) / 350) ** 2)
                + 1e-5)
    sp = np.repeat(envelope[None, :], 5, axis=0) * np.exp(
        rng.normal(0, 0.05, size=(5, bins)))

    mcep = power_to_mcep(sp, fft_size, fs, n_mcep)
    assert mcep.shape == (5, n_mcep)
    reconstructed = mcep_to_power(mcep, fft_size, fs, n_mcep)

    # Evaluate on a *finer* grid than the analysis bands: measuring on the
    # analysis grid itself would only check the transform against itself.
    grid = 2 * mel_band_count(n_mcep)
    error = np.abs(power_to_logmel(reconstructed, grid, fft_size, fs)
                   - power_to_logmel(sp, grid, fft_size, fs))
    assert error.mean() < 0.1, "envelope representation should be near lossless"
    # absolute level must survive too (c0 is part of the feature vector)
    assert abs(np.mean(power_to_logmel(reconstructed, grid, fft_size, fs))
               - np.mean(power_to_logmel(sp, grid, fft_size, fs))) < 0.05


def test_envelope_band_spacing_resolves_the_formant_region():
    """The analysis grid must be much finer than one band per coefficient.

    The envelope is rebuilt by interpolating between mel band centres, so a
    formant (or valley) that falls *between* two knots is flattened, and the
    reconstructed curve can then peak at a knot instead.  The historical
    one-band-per-coefficient grid had 307-421 Hz spacing between 1.8 and
    3.2 kHz -- coarse enough to put a knot at 2365 Hz, which is the frequency
    the demo vowels used to show a spurious peak at.
    """
    fs = 44100
    knots = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(fs / 2.0),
                                  mel_band_count(DEFAULT_N_MCEP) + 2)[1:-1])
    spacing = np.diff(knots)[(knots[:-1] >= 1800) & (knots[:-1] <= 3200)]
    assert spacing.size and spacing.max() < 250.0


def test_narrow_formant_is_not_replaced_by_a_peak_at_a_band_knot():
    """A formant between two knots must not come back as a peak *at* a knot.

    Regression test for the constant ~2.4 kHz peak in the demo vowel spectra:
    with the old 32-band grid (one band per cepstral coefficient, knot #13 at
    2365 Hz) a narrow third formant at 2.5 kHz was reconstructed as a peak at
    2369 Hz.  The reconstruction's peak has to sit nearer the real formant than
    the knot that used to capture it.
    """
    fs, fft_size, n_mcep = 44100, 2048, DEFAULT_N_MCEP
    formant_hz, knot_hz = 2500.0, 2365.5
    freqs = np.linspace(0.0, fs / 2.0, fft_size // 2 + 1)
    db = (-30.0
          + 34.0 * np.exp(-0.5 * ((freqs - 800.0) / 250.0) ** 2)      # F1
          + 26.0 * np.exp(-0.5 * ((freqs - 1200.0) / 250.0) ** 2)     # F2
          + 22.0 * np.exp(-0.5 * ((freqs - formant_hz) / 150.0) ** 2)  # narrow F3
          - 14.0 * np.exp(-0.5 * ((freqs - knot_hz) / 70.0) ** 2)     # narrow valley
          - 40.0 * np.exp(-0.5 * ((freqs - 6000.0) / 2500.0) ** 2))   # spectral tilt
    sp = (10.0 ** (db / 10.0))[None, :]

    reconstructed = mcep_to_power(power_to_mcep(sp, fft_size, fs, n_mcep),
                                  fft_size, fs, n_mcep)[0]
    window = (freqs >= 2100.0) & (freqs <= 2800.0)
    true_peak = float(freqs[window][np.argmax(sp[0][window])])
    decoded_peak = float(freqs[window][np.argmax(reconstructed[window])])

    assert true_peak == pytest.approx(formant_hz, abs=100.0)
    assert abs(decoded_peak - formant_hz) < abs(decoded_peak - knot_hz), (
        f"envelope peak reconstructed at {decoded_peak:.0f} Hz, i.e. pinned to "
        f"the band knot at {knot_hz:.0f} Hz instead of the formant at "
        f"{formant_hz:.0f} Hz")


def test_aperiodicity_bands_roundtrip():
    fs, fft_size, n_band = 22050, 1024, 5
    bins = fft_size // 2 + 1
    rng = np.random.default_rng(2)
    ap = np.clip(rng.uniform(0.05, 0.95, size=(6, bins)), 0, 1)
    bands = aperiodicity_to_bands(ap, n_band, fs)
    assert bands.shape == (6, n_band)
    # band means must lie inside the input range and be ordered by band
    assert (bands >= ap.min() - 1e-9).all() and (bands <= ap.max() + 1e-9).all()
    restored = bands_to_aperiodicity(bands, n_band, fs, bins)
    assert restored.shape == ap.shape
    assert (restored >= 1e-4).all() and (restored <= 1.0).all()
    # a constant aperiodicity must come back constant
    flat = np.full((3, bins), 0.4)
    back = bands_to_aperiodicity(aperiodicity_to_bands(flat, n_band, fs),
                                 n_band, fs, bins)
    assert np.allclose(back, 0.4, atol=1e-6)


def test_encode_decode_roundtrip_preserves_pitch_and_level():
    spec = FeatureSpec(fs=22050, fft_size=1024, n_mcep=20)
    fs = spec.fs
    rng = np.random.default_rng(3)
    n_frames, bins = 40, spec.n_bins
    f0 = np.full(n_frames, 220.0)
    sp = np.exp(rng.normal(0, 0.5, size=(n_frames, bins)))
    ap = np.clip(rng.uniform(0.05, 0.6, size=(n_frames, bins)), 0.01, 1.0)

    features = spec.encode(f0, sp, ap)
    assert features.shape == (n_frames, spec.static_dim)
    decoded = spec.decode(features)
    assert np.allclose(decoded.f0, f0, rtol=1e-6)
    # spectral shape (in band-density space) is preserved -- measured on a
    # finer grid than the envelope is sampled on, so the reconstruction has to
    # be right between the knots too, not just at them
    grid = 2 * mel_band_count(spec.n_mcep)
    density_before = power_to_logmel(sp, grid, spec.fft_size, fs)
    density_after = power_to_logmel(decoded.sp, grid, spec.fft_size, fs)
    assert np.abs(density_after - density_before).mean() < 0.25


def test_unvoiced_frames_do_not_carry_pitch():
    spec = FeatureSpec(fs=22050, fft_size=1024, n_mcep=12)
    f0 = np.array([220.0, 220.0, 0.0, 0.0, 200.0])
    sp = np.ones((5, spec.n_bins)) * 1e-3
    ap = np.full((5, spec.n_bins), 0.5)
    features = spec.encode(f0, sp, ap)
    # 220 Hz is three semitones below the C4 reference
    assert features[0, 0] == pytest.approx(12 * np.log2(220.0 / spec.f0_ref_hz))
    # unvoiced frames are stored as 0.0 semitones (a NaN would poison the GMMs)
    assert features[2, 0] == 0.0 and features[3, 0] == 0.0
    # ... so voicing cannot be recovered from feature 0 alone: a decoder that
    # wants silence must be told which frames are unvoiced, which is exactly
    # what the synthesizer does by passing a pitch track with NaN in it.
    assert (spec.decode(features).f0 > 0).all()
    voiced_track = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1) / spec.f0_ref_hz),
                            np.nan)
    assert (spec.decode(features, f0_semitones=voiced_track).f0[2:4] == 0.0).all()


def test_decode_accepts_an_explicit_pitch_track():
    """The pitch model overrides feature dimension 0 at synthesis time."""
    spec = FeatureSpec(fs=22050, fft_size=1024, n_mcep=12)
    sp = np.ones((3, spec.n_bins)) * 1e-3
    ap = np.full((3, spec.n_bins), 0.5)
    features = spec.encode(np.array([100.0, 100.0, 100.0]), sp, ap)
    ninety_hz = 12.0 * np.log2(90.0 / spec.f0_ref_hz)
    decoded = spec.decode(features, f0_semitones=np.full(3, ninety_hz))
    assert np.allclose(decoded.f0, 90.0, rtol=1e-6)
    # NaN (unvoiced) becomes 0 Hz
    decoded = spec.decode(features, f0_semitones=np.array([np.nan, 0.0, np.nan]))
    assert decoded.f0[0] == 0.0 and decoded.f0[2] == 0.0
    assert decoded.f0[1] > 0


def test_dynamic_features_shapes_and_edges():
    rng = np.random.default_rng(4)
    static = rng.normal(size=(10, 3))
    both = add_dynamic_features(static, True, True)
    assert both.shape == (10, 9)
    assert np.allclose(both[:, :3], static)
    # constant input -> zero deltas
    constant = np.ones((6, 2))
    result = add_dynamic_features(constant, True, False)
    assert np.allclose(result[:, 2:], 0.0)
    # edge replication: the first delta is half the interior one for a ramp
    ramp = np.arange(8, dtype=float)[:, None]
    delta = add_dynamic_features(ramp, True, False)[:, 1]
    assert delta[0] == pytest.approx(0.5)
    assert delta[3] == pytest.approx(1.0)
    assert np.allclose(remove_dynamic_features(both, 3), static)


def test_single_frame_and_empty_inputs():
    assert add_dynamic_features(np.zeros((1, 4)), True, False).shape == (1, 8)
    assert add_dynamic_features(np.zeros((0, 4)), True, False).shape == (0, 8)
    spec = FeatureSpec(fs=22050, fft_size=512, n_mcep=8)
    assert spec.encode(np.zeros(0), np.zeros((0, 257)),
                       np.zeros((0, 257))).shape == (0, spec.static_dim)


def test_invalid_spectral_shape_is_rejected():
    spec = FeatureSpec(fs=22050, fft_size=1024, n_mcep=12)
    with pytest.raises(ValueError):
        spec.encode(np.ones(3), np.ones((3, 100)), np.ones((3, 513)))


def test_stream_helpers_are_inverse():
    spec = FeatureSpec(n_mcep=6, n_band=3, use_delta=True)
    rng = np.random.default_rng(5)
    features = rng.normal(size=(9, spec.dim))
    stacked = stack_streams(features, spec.stream_sizes)
    assert stacked.shape == (18, spec.static_dim)
    assert np.allclose(unstack_streams(stacked, 9, spec.static_dim), features)
    parted = split_streams(features, (spec.static_dim, spec.static_dim))
    assert len(parted) == 2 and parted[0].shape == (9, spec.static_dim)
    with pytest.raises(ValueError):
        split_streams(features, (spec.static_dim,))


def test_feature_spec_serialisation_roundtrip():
    spec = FeatureSpec(fs=22050, n_mcep=24, n_band=4, use_delta2=True,
                       frame_period=10.0)
    restored = FeatureSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()
    assert restored.dim == spec.dim
    assert restored.stream_sizes == spec.stream_sizes


def test_note_helpers():
    assert labels_module.midi_to_hz(69) == pytest.approx(440.0)
    assert labels_module.hz_to_midi(440.0) == pytest.approx(69.0)
    semitones = hz_to_semitone(np.array([261.6255653]), 261.6255653)
    assert semitones[0] == pytest.approx(0.0, abs=1e-6)
    assert semitone_to_hz(np.array([12.0]), 261.6255653)[0] == pytest.approx(
        523.2511306, rel=1e-6)


def test_acoustic_frame_sequence_reports_duration():
    sequence = AcousticFrameSequence(f0=np.zeros(200), sp=np.zeros((200, 513)),
                                     ap=np.zeros((200, 513)), frame_period=5.0,
                                     fs=22050, fft_size=1024)
    assert len(sequence) == 200
    assert sequence.duration == pytest.approx(1.0)
    assert not sequence.voiced.any()
