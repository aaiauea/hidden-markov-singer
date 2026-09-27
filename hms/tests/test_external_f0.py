"""External F0 override: `synthesize(..., f0=...)`.

An explicitly supplied trajectory is *authoritative*: it replaces the generated
contour (score note, learned deviation and generated vibrato) instead of being
added to it, and it must line up with the frame sequence the score produces --
HMS never resizes, interpolates or pads it.  These tests check that contract at
the synthesis/vocoder boundary (`SynthesisResult.params.f0`, the `(f0, sp, ap)`
triple the vocoder actually consumes), because waveform comparison is not
sensitive enough to separate "the right trajectory" from "a similar one".
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core import labels as labels_module
from hms.core.labels import Score, Segment, Utterance
from hms.core.synthesizer import (SynthesisConfig, Synthesizer,
                                  external_f0_to_semitones, synthesize)

F_REF = 261.6255653005986            # C4, the default F0 reference


def render(model, score, f0=None, **config):
    """One deterministic render; `f0` is the feature under test."""
    options = dict(vibrato=False, seed=0)
    options.update(config)
    return Synthesizer(model, SynthesisConfig(**options)).synthesize(
        score, f0=f0)


def frame_count(model, score, **config) -> int:
    """How many frames this score renders into (what an `f0` must match)."""
    return len(render(model, score, **config).params.f0)


def held_note_score(seconds: float = 0.5, note: float = 60.0) -> Score:
    return Score([Utterance("held", [
        Segment("sil", 0.0, 0.1),
        Segment("a", 0.1, 0.1 + seconds, note=note),
        Segment("sil", 0.1 + seconds, 0.2 + seconds)])])


def roughness(result) -> float:
    """Mean frame-to-frame F0 step in semitones (vibrato raises this)."""
    f0 = result.f0_semitones
    voiced = np.isfinite(f0)
    return float(np.abs(np.diff(f0[voiced])).mean())


# --------------------------------------------------------------------------
# 1. no external F0 -> nothing changes
# --------------------------------------------------------------------------

def test_without_an_external_f0_synthesis_is_unchanged(trained_model,
                                                       short_score):
    plain = render(trained_model, short_score)
    explicit_none = render(trained_model, short_score, f0=None)

    assert np.array_equal(plain.audio, explicit_none.audio)
    assert np.array_equal(plain.params.f0, explicit_none.params.f0)
    assert np.array_equal(plain.params.sp, explicit_none.params.sp)
    assert plain.diagnostics == explicit_none.diagnostics
    assert not any("external F0" in message for message in plain.diagnostics)
    # the score still drives F0, exactly as before
    voiced = plain.params.f0 > 0
    assert voiced.any()
    assert np.allclose(plain.params.f0[voiced],
                       labels_module.midi_to_hz(plain.notes[voiced]),
                       rtol=1e-9)


# --------------------------------------------------------------------------
# 2./3. the supplied trajectory reaches the vocoder untouched
# --------------------------------------------------------------------------

def test_a_constant_external_f0_reaches_the_vocoder(trained_model):
    score = held_note_score()
    trajectory = np.full(frame_count(trained_model, score), 220.0)

    result = render(trained_model, score, f0=trajectory)

    assert len(result.params.f0) == len(trajectory)
    assert (result.params.f0 > 0).all()
    assert np.allclose(result.params.f0, 220.0, rtol=1e-9)
    assert np.allclose(result.f0_semitones, 12.0 * np.log2(220.0 / F_REF),
                       rtol=1e-9, atol=1e-12)


def test_a_changing_external_f0_is_preserved_frame_by_frame(trained_model):
    score = held_note_score(seconds=1.0)
    n_frames = frame_count(trained_model, score)
    trajectory = np.linspace(150.0, 320.0, n_frames)

    result = render(trained_model, score, f0=trajectory)

    assert np.allclose(result.params.f0, trajectory, rtol=1e-9)
    assert np.allclose(result.f0_semitones,
                       12.0 * np.log2(trajectory / F_REF), rtol=1e-9,
                       atol=1e-12)
    # the shape of the contour survives too: monotonically rising, no smoothing
    assert (np.diff(result.params.f0) > 0).all()


def test_unvoiced_frames_use_zero_hz(trained_model):
    """The existing convention: 0.0 Hz is unvoiced (NaN internally)."""
    score = held_note_score()
    trajectory = np.full(frame_count(trained_model, score), 220.0)
    trajectory[10:20] = 0.0

    result = render(trained_model, score, f0=trajectory)

    assert not (result.params.f0[10:20] > 0).any()
    assert np.isnan(result.f0_semitones[10:20]).all()
    assert np.allclose(result.params.f0[:10], 220.0, rtol=1e-9)
    assert np.allclose(result.params.f0[20:], 220.0, rtol=1e-9)


def test_external_f0_changes_only_the_pitch(trained_model, short_score):
    """The acoustic trajectory (sp/ap) and the timing are untouched."""
    base = render(trained_model, short_score)
    trajectory = np.full(len(base.params.f0), 250.0)

    result = render(trained_model, short_score, f0=trajectory)

    assert np.array_equal(result.params.sp, base.params.sp)
    assert np.array_equal(result.params.ap, base.params.ap)
    assert len(result.audio) == len(base.audio)
    assert not np.allclose(result.audio, base.audio)
    assert any("external F0 override" in message
               for message in result.diagnostics)


# --------------------------------------------------------------------------
# 4.-6. validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("length", [1, 7])
def test_a_mismatched_trajectory_length_is_an_error(trained_model, length):
    score = held_note_score()
    n_frames = frame_count(trained_model, score)
    assert length != n_frames

    with pytest.raises(ValueError, match=r"external F0 has \d+ frame\(s\) but "
                                         r"the render has \d+"):
        render(trained_model, score, f0=np.full(length, 220.0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(trained_model, bad):
    score = held_note_score()
    trajectory = np.full(frame_count(trained_model, score), 220.0)
    trajectory[7] = bad

    with pytest.raises(ValueError, match="external F0 must be finite") as info:
        render(trained_model, score, f0=trajectory)

    message = str(info.value)
    assert "frame 7" in message
    assert "0.0 Hz" in message            # and says how to mark unvoiced


def test_negative_frequencies_are_rejected(trained_model):
    score = held_note_score()
    trajectory = np.full(frame_count(trained_model, score), 220.0)
    trajectory[3] = -1.5

    with pytest.raises(ValueError, match="non-negative") as info:
        render(trained_model, score, f0=trajectory)

    message = str(info.value)
    assert "frame 3" in message
    assert "-1.5 Hz" in message


def test_an_empty_trajectory_is_rejected(trained_model):
    score = held_note_score()

    with pytest.raises(ValueError, match="external F0 trajectory is empty"):
        render(trained_model, score, f0=np.zeros(0))


def test_a_two_dimensional_trajectory_is_rejected(trained_model):
    score = held_note_score()
    n_frames = frame_count(trained_model, score)

    with pytest.raises(ValueError, match="one value per frame"):
        render(trained_model, score, f0=np.full((n_frames, 1), 220.0))


def test_non_numeric_trajectories_are_rejected(trained_model):
    score = held_note_score()

    with pytest.raises(ValueError, match="real numbers"):
        render(trained_model, score, f0=["a", "b", "c"])


def test_a_scalar_is_rejected_rather_than_broadcast(trained_model):
    score = held_note_score()

    with pytest.raises(ValueError, match="one value per frame"):
        render(trained_model, score, f0=220.0)


def test_out_of_range_frames_are_sung_as_supplied_and_reported(trained_model):
    """Outside the trained range is not a reason to alter the trajectory."""
    score = held_note_score()
    trajectory = np.full(frame_count(trained_model, score), 2000.0)

    result = render(trained_model, score, f0=trajectory)

    assert np.allclose(result.params.f0, 2000.0, rtol=1e-9)
    assert result.params.f0.max() > trained_model.spec.f0_ceil
    assert any("F0 range" in message for message in result.diagnostics)


def test_a_malformed_trajectory_is_still_rejected_when_out_of_range(
        trained_model):
    """OOV handling must not weaken the validation in any direction."""
    score = held_note_score()
    n_frames = frame_count(trained_model, score)
    with pytest.raises(ValueError, match=r"external F0 has \d+ frame\(s\) but "
                                         r"the render has \d+"):
        render(trained_model, score, f0=np.full(n_frames + 1, 5000.0))
    with pytest.raises(ValueError, match="non-negative"):
        render(trained_model, score,
               f0=np.concatenate([[-1.0], np.full(n_frames - 1, 5000.0)]))
    with pytest.raises(ValueError, match="must be finite"):
        render(trained_model, score,
               f0=np.concatenate([[np.nan], np.full(n_frames - 1, 5000.0)]))


# --------------------------------------------------------------------------
# 7./8. nothing is added on top of an external trajectory
# --------------------------------------------------------------------------

@pytest.mark.parametrize("f0_source", ["acoustic", "state_means"])
def test_an_external_f0_receives_no_learned_deviation(trained_model,
                                                      short_score, f0_source):
    learned = render(trained_model, short_score, f0_source=f0_source,
                     pitch_variation=8.0)
    # the learned deviation really does move this render, so the override below
    # is proving something
    voiced = learned.params.f0 > 0
    deviation = 12.0 * np.log2(
        learned.params.f0[voiced]
        / labels_module.midi_to_hz(learned.notes[voiced]))
    assert np.median(np.abs(deviation)) > 0.5

    trajectory = np.full(len(learned.params.f0), 300.0)
    result = render(trained_model, short_score, f0=trajectory,
                    f0_source=f0_source, pitch_variation=8.0)

    assert np.allclose(result.params.f0, trajectory, rtol=1e-9)


def test_an_external_f0_receives_no_generated_vibrato(trained_model,
                                                      short_score):
    shaken = dict(vibrato=True, vibrato_depth=2.0, vibrato_rate=6.0)
    flat = render(trained_model, short_score)
    modulated = render(trained_model, short_score, **shaken)
    assert roughness(modulated) > roughness(flat)      # vibrato is really on

    trajectory = np.full(len(modulated.params.f0), 300.0)
    result = render(trained_model, short_score, f0=trajectory, **shaken)

    assert np.allclose(result.params.f0, trajectory, rtol=1e-9)
    assert roughness(result) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# the conversion helper itself
# --------------------------------------------------------------------------

def test_conversion_uses_the_project_f0_conventions(trained_model):
    spec = trained_model.spec
    semitones = external_f0_to_semitones(
        [0.0, spec.f0_ref_hz, 2.0 * spec.f0_ref_hz], 3, spec)

    assert np.isnan(semitones[0])                 # 0 Hz -> unvoiced -> NaN
    assert semitones[1] == pytest.approx(0.0)
    assert semitones[2] == pytest.approx(12.0)


def test_values_below_the_voicing_threshold_are_unvoiced(trained_model):
    spec = trained_model.spec
    below = float(spec.voiced_threshold) / 2.0

    assert np.isnan(external_f0_to_semitones([below], 1, spec)[0])
    assert np.isfinite(external_f0_to_semitones([spec.voiced_threshold], 1,
                                                spec)[0])


def test_the_functional_entry_point_forwards_an_external_f0(trained_model,
                                                            short_score):
    config = SynthesisConfig(vibrato=False, seed=0)
    base = synthesize(trained_model, short_score, config)
    trajectory = np.linspace(180.0, 360.0, len(base.params.f0))

    result = synthesize(trained_model, short_score, config, f0=trajectory)

    assert np.allclose(result.params.f0, trajectory, rtol=1e-9)
