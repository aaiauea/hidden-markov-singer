"""Pitch conditioning, vibrato, and the duration model.

The central claim of HMS is that the *requested note* drives F0 while the
statistical model only supplies variation around it.  These tests pin that
contract down: the note-relative transform must be exact, the residual must be
what the model stores, and shifting the note must shift the output one-for-one.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core import labels as labels_module
from hms.core.duration import (DEFAULT_STATE_FRAMES, DurationModel,
                               DurationStats)
from hms.core.pitch import (PitchModel, PitchStats, Vibrato,
                            absolute_semitones, estimate_vibrato,
                            interpolate_gaps, note_relative_pitch)

F_REF = 261.6255653005986            # C4, the default reference of FeatureSpec


def vibrato_track(rate_hz=5.0, depth=0.5, length=800, frame_period=5.0,
                  fs=22050.0, base=261.6255653):
    """An F0 track with a clean superimposed vibrato (no noise)."""
    frames = np.arange(length)
    t = frames * frame_period / 1000.0
    semitones = depth * np.sin(2 * np.pi * rate_hz * t)
    return base * 2 ** (semitones / 12.0), np.ones(length, bool)


def test_note_relative_pitch_is_exact_for_voiced_frames():
    notes = np.array([60.0, 62.0, 64.0, 67.0])
    note_hz = labels_module.midi_to_hz(notes)
    # sing exactly the notes, but 20 cents sharp on the last one
    f0 = note_hz * np.array([1.0, 1.0, 1.0, 2 ** (20 / 1200)])
    relative = note_relative_pitch(f0, note_hz, np.ones(4, bool), F_REF)
    assert np.allclose(relative[:3], 0.0, atol=1e-9)
    assert relative[3] == pytest.approx(0.2, abs=1e-6)


def test_note_relative_pitch_interpolates_unvoiced_frames():
    note_hz = np.full(9, 261.6255653)
    f0 = np.full(9, 261.6255653)
    f0[4] = 0.0                                  # an unvoiced frame
    voiced = np.ones(9, bool)
    voiced[4] = False
    relative = note_relative_pitch(f0, note_hz, voiced, F_REF)
    assert np.isfinite(relative).all()
    assert relative[4] == pytest.approx(0.0, abs=1e-9)


def test_note_relative_pitch_survives_a_fully_unvoiced_utterance():
    relative = note_relative_pitch(np.zeros(5), np.full(5, 261.6255653),
                                   np.zeros(5, bool), F_REF)
    assert np.allclose(relative, 0.0)


def test_absolute_semitones_adds_the_note_back():
    notes = np.array([57.0, 60.0, 64.0])
    note_hz = labels_module.midi_to_hz(notes)
    relative = np.array([0.0, -0.3, 0.5])
    absolute = absolute_semitones(relative, note_hz, F_REF)
    expected = 12 * np.log2(note_hz / F_REF) + relative
    assert np.allclose(absolute, expected)
    # a note an octave higher is 12 semitones higher, not "the same pitch"
    octave_up = absolute_semitones(relative, note_hz * 2, F_REF)
    assert np.allclose(octave_up - absolute, 12.0)


def test_interpolate_gaps_is_linear_and_extrapolates_flat():
    values = np.array([np.nan, 1.0, np.nan, np.nan, 3.0, np.nan])
    filled = interpolate_gaps(values)
    assert np.isfinite(filled).all()
    assert filled[0] == pytest.approx(1.0)          # constant extrapolation
    assert filled[5] == pytest.approx(3.0)
    assert filled[2] == pytest.approx(1.0 + 2.0 / 3)
    assert np.allclose(interpolate_gaps(np.full(4, np.nan)), 0.0)
    assert np.allclose(interpolate_gaps(np.arange(4.0)), np.arange(4.0))


def test_vibrato_is_disabled_by_default_and_has_no_delay():
    vibrato = Vibrato()
    assert not vibrato.enabled
    assert np.allclose(vibrato.contour(100, 5.0, 0), 0.0)


def test_vibrato_respects_delay_depth_and_period():
    vibrato = Vibrato(enabled=True, rate_hz=5.0, depth_semitones=0.5,
                      delay_ms=200.0, attack_ms=1.0, randomness=0.0,
                      waveform="sine")
    frame_period = 5.0
    contour = vibrato.contour(200, frame_period, note_index=0)
    delay_frames = int(200.0 / frame_period)
    assert np.allclose(contour[:delay_frames], 0.0)
    assert np.abs(contour[delay_frames:]).max() == pytest.approx(0.5, rel=1e-6)
    # one full period later the contour repeats
    period = int(round(1000.0 / (5.0 * frame_period)))
    assert contour[delay_frames + period] == pytest.approx(
        contour[delay_frames], abs=1e-9)
    # a short note gets no time to develop vibrato
    assert np.allclose(vibrato.contour(5, frame_period, 0), 0.0)


def test_vibrato_triangle_waveform_stays_bounded():
    vibrato = Vibrato(enabled=True, waveform="triangle", randomness=0.0,
                      delay_ms=0.0, depth_semitones=0.4)
    contour = vibrato.contour(300, 5.0, 0)
    assert np.abs(contour).max() <= 0.4 + 1e-9


def test_estimate_vibrato_measures_rate_and_depth():
    f0, voiced = vibrato_track(rate_hz=5.0, depth=0.5, length=1000)
    estimate = estimate_vibrato(f0, voiced, 22050.0, 5.0)
    assert estimate is not None
    rate, depth = estimate
    assert rate == pytest.approx(5.0, abs=0.4)
    assert depth == pytest.approx(0.5, abs=0.2)


def test_estimate_vibrato_returns_none_without_enough_data():
    assert estimate_vibrato(np.zeros(10), np.zeros(10, bool), 22050.0, 5.0) is None
    steady, voiced = vibrato_track(depth=0.0, length=600)
    estimate = estimate_vibrato(steady, voiced, 22050.0, 5.0)
    assert estimate is None or estimate[1] < 0.05


def test_pitch_model_collects_per_state_statistics():
    model = PitchModel()
    per_state_pitch = [
        [np.array([0.1, 0.2]), np.array([0.0])],       # state 0, two occurrences
        [np.array([-0.4, -0.3, -0.5])],                # state 1
    ]
    per_state_voiced = [
        [np.array([True, True]), np.array([True])],
        [np.array([True, False, True])],
    ]
    model.add_state_statistics("a", per_state_pitch, per_state_voiced)
    means = model.state_means("a")
    assert means[0] == pytest.approx(0.1, abs=1e-6)
    assert means[1] == pytest.approx(-0.4, abs=1e-6)
    # 5 of the 6 frames collected across both states and both occurrences
    assert model.voiced_prior["a"] == pytest.approx(5 / 6)
    assert model.state_means("unknown").size == 0


def test_pitch_stats_are_clamped_away_from_zero_variance():
    model = PitchModel()
    model.add_state_statistics("a", [[np.array([0.5, 0.5])]],
                               [[np.array([True, True])]])
    assert model.stats["a"][0].variance > 0


def test_voiced_mask_respects_the_phoneme_class():
    model = PitchModel(voiced_prior={"a": 0.99})
    phones = ["a", "s", "sil", "a"]
    state_voiced = np.array([0.4, 0.9, 0.6, 0.4])       # from the HMM states
    phoneme_voiced = [True, False, False, True]
    mask = model.voiced_mask(phones, state_voiced, phoneme_voiced)
    assert mask.tolist() == [True, False, False, True]


def test_generate_puts_the_note_under_the_deviation():
    model = PitchModel()
    note_semitones = np.array([0.0, 0.0, 2.0, 2.0])
    voiced = np.array([True, True, True, False])
    trajectory = np.array([0.1, -0.1, 0.2, 0.3])
    f0 = model.generate(note_semitones, voiced, relative_trajectory=trajectory)
    assert np.allclose(f0[:3], [0.1, -0.1, 2.2])
    assert np.isnan(f0[3])                      # unvoiced frames carry no F0


def test_generate_without_a_trajectory_uses_the_note_itself():
    model = PitchModel()
    f0 = model.generate(np.array([3.0, 3.0]), np.array([True, True]),
                        relative_trajectory=None)
    assert np.allclose(f0, 3.0)


def test_pitch_model_roundtrip():
    model = PitchModel()
    model.add_state_statistics("a", [[np.array([0.2, 0.4])]],
                               [[np.array([True, True])]])
    model.vibrato = Vibrato(enabled=True, rate_hz=6.0, depth_semitones=0.3)
    restored = PitchModel.from_dict(model.to_dict())
    assert np.allclose(restored.state_means("a"), model.state_means("a"))
    assert restored.voiced_prior == model.voiced_prior
    assert restored.vibrato.enabled and restored.vibrato.rate_hz == 6.0


def test_pitch_stats_roundtrip():
    stats = PitchStats(mean=0.25, variance=0.09, count=12)
    again = PitchStats.from_dict(stats.to_dict())
    assert (again.mean, again.variance, again.count) == (0.25, 0.09, 12)


# --------------------------------------------------------------------------
# duration
# --------------------------------------------------------------------------


def fitted_duration_model():
    model = DurationModel()
    model.fit({"a": [40.0, 44.0, 38.0, 42.0],
               "s": [12.0, 14.0, 11.0]})
    return model


def test_duration_model_fits_log_durations():
    model = fitted_duration_model()
    assert model.has("a") and not model.has("zzz")
    assert model.mean_frames("a") == pytest.approx(41.0, rel=0.02)
    assert model.mean_frames("zzz") == DEFAULT_STATE_FRAMES
    # the geometric mean is what the log-space model predicts
    stats = model.stats["a"]
    assert np.exp(stats.mean) == pytest.approx(model.mean_frames("a"))
    assert stats.count == 4


def test_duration_prediction_is_deterministic_without_speaking():
    model = fitted_duration_model()
    phones = ["a", "s", "a", "zzz"]
    mean = model.predict(phones, 5.0, speak=False)
    assert len(mean) == 4
    assert mean[0] == pytest.approx(model.mean_frames("a"))
    assert mean[3] == pytest.approx(DEFAULT_STATE_FRAMES)
    assert np.all(mean >= 1.0)
    # tempo scales the result
    faster = model.predict(phones, 5.0, tempo=2.0, speak=False)
    assert np.allclose(faster, mean / 2.0)
    # sampling introduces variation but stays positive and nearby
    rng = np.random.default_rng(0)
    sampled = np.array([model.predict(phones, 5.0, rng=rng)[0] for _ in range(30)])
    assert (sampled > 0).all()
    assert sampled.std() > 0


def test_more_variance_scale_means_more_timing_variation():
    quiet = fitted_duration_model()
    quiet.variance_scale = 0.0
    rng = np.random.default_rng(0)
    values = [quiet.predict(["a"], 5.0, rng=rng)[0] for _ in range(20)]
    assert np.allclose(values, values[0])       # variance 0 -> deterministic


def test_allocate_covers_exactly_and_fairly():
    counts = DurationModel.allocate(101, [0.1, 0.6, 0.3])
    assert counts.sum() == 101
    assert (counts >= 1).all()
    assert np.argmax(counts) == 1               # the biggest share wins
    # a single frame still goes somewhere sane
    tiny = DurationModel.allocate(1, [0.1, 0.6, 0.3])
    assert tiny.sum() == 1
    # fewer frames than states: one each, from the first state
    starved = DurationModel.allocate(2, [0.5, 0.5, 0.5, 0.5])
    assert starved.sum() == 2
    assert DurationModel.allocate(0, [0.5, 0.5]).sum() == 0
    # zero proportions fall back to a uniform split
    uniform = DurationModel.allocate(9, [0.0, 0.0, 0.0])
    assert uniform.sum() == 9
    assert uniform.max() - uniform.min() <= 1


def test_allocate_is_monotone_in_the_proportions():
    small = DurationModel.allocate(100, [1.0, 2.0, 3.0])
    large = DurationModel.allocate(200, [1.0, 2.0, 3.0])
    assert np.all(large >= small)
    assert abs(large.sum() - 200) == 0


def test_duration_and_stats_roundtrip():
    model = fitted_duration_model()
    model.variance_scale = 1.7
    restored = DurationModel.from_dict(model.to_dict())
    assert restored.mean_frames("a") == pytest.approx(model.mean_frames("a"))
    assert restored.variance_scale == pytest.approx(1.7)
    stats = DurationStats(mean=3.0, variance=0.2, count=5)
    assert DurationStats.from_dict(stats.to_dict()).count == 5
