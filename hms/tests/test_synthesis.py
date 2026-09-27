"""End-to-end synthesis: score -> state sequence -> parameters -> audio.

These are the acceptance tests for the vertical slice: a model trained on a
small corpus must render a score so that the *requested notes* dominate F0, the
requested timing survives, and changing the notes moves the pitch accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core import labels as labels_module
from hms.core.synthesizer import SynthesisConfig, Synthesizer
from hms.vocoder.base import limit_peak

F_REF = 261.6255653005986            # C4, the default F0 reference


def render(model, score, **config):
    options = dict(vibrato=False, seed=0)
    options.update(config)
    return Synthesizer(model, SynthesisConfig(**options)).synthesize(score)


def voiced_deviations(result):
    """F0 error vs. the requested note, in semitones, on voiced frames."""
    f0 = result.params.f0
    notes = result.notes
    voiced = f0 > 0
    expected = labels_module.midi_to_hz(notes[voiced])
    return 12.0 * np.log2(f0[voiced] / expected)


def test_synthesis_config_rejects_unknown_modes_and_invalid_scales():
    with pytest.raises(ValueError, match="f0_source"):
        SynthesisConfig(f0_source="autopitch")
    with pytest.raises(ValueError, match="variance_scale"):
        SynthesisConfig(variance_scale=0.0)
    with pytest.raises(ValueError, match="tempo"):
        SynthesisConfig(tempo=0.0)


def test_synthesis_duration_follows_the_score(trained_model, score):
    result = render(trained_model, score)
    assert result.duration == pytest.approx(score.total_duration, rel=0.02)
    assert len(result.state_sequence) == len(result.params.f0)
    assert len(result.phones) == len(result.params.f0)


def test_generated_audio_is_sane(trained_model, short_score):
    score = short_score
    result = render(trained_model, score)
    audio = result.audio
    assert np.isfinite(audio).all()
    # The vocoder returns WORLD's output verbatim (no hidden peak gain; see
    # `Vocoder.synthesize`), so a very periodic render may overshoot +/-1 and
    # the writer's headroom handling is what brings it back into range.
    assert np.abs(audio).max() < 10.0
    assert np.abs(limit_peak(audio)).max() <= 1.0
    assert np.sqrt((audio ** 2).mean()) > 0.01
    # the render must not be one long silence: voiced frames carry energy
    voiced = result.params.f0 > 0
    assert voiced.mean() > 0.4


def test_the_requested_note_drives_f0(trained_model, short_score):
    """The note, not the training speaker, must set the pitch."""
    score = short_score
    result = render(trained_model, score)
    deviations = voiced_deviations(result)
    assert len(deviations) > 100
    # the model only supplies variation *around* the note
    assert np.median(np.abs(deviations)) < 1.0
    assert np.percentile(np.abs(deviations), 90) < 3.0


def test_transposing_the_score_moves_the_pitch_one_for_one(trained_model):
    """One note, three transpositions: the ratio must be exact."""
    from hms.core.labels import Score, Utterance, Segment

    segments = [Segment("sil", 0.0, 0.1),
                Segment("a", 0.1, 0.8, note=60.0),
                Segment("sil", 0.8, 0.9)]
    score = Score([Utterance(name="held_note", segments=segments)])
    base = render(trained_model, score)
    reference = np.median(base.params.f0[base.params.f0 > 0])
    for semitones in (-5.0, 3.0, 12.0):
        other = render(trained_model, score, transpose=semitones)
        assert len(other.audio) == len(base.audio)
        ratio = np.median(other.params.f0[other.params.f0 > 0]) / reference
        assert ratio == pytest.approx(2 ** (semitones / 12.0), rel=0.01)
        deviations = voiced_deviations(other)
        assert np.median(np.abs(deviations)) < 1.0


def test_pitch_outside_the_analysed_range_is_sung_at_the_requested_f0(
        trained_model):
    """A note above the trained range keeps its pitch instead of the ceiling."""
    from hms.core.labels import Score, Utterance, Segment

    score = Score([Utterance(name="high", segments=[
        Segment("a", 0.0, 0.5, note=60.0)])])
    result = render(trained_model, score, transpose=36.0)   # ~2000 Hz
    spec = trained_model.spec
    assert (result.params.f0 > 0).all()
    assert spec.f0_ceil < 2000.0
    assert np.median(result.params.f0) == pytest.approx(
        labels_module.midi_to_hz(96.0), rel=1e-6)
    assert any("F0 range" in message for message in result.diagnostics)


def test_absurd_midi_note_is_named_and_sang_at_the_nearest_valid_note(
        trained_model):
    """MIDI 200 is not a MIDI note: it must be named, and normalised."""
    from hms.core.labels import Score, Utterance, Segment

    score = Score([Utterance(name="absurd", segments=[
        Segment("a", 0.0, 0.5, note=200.0)])])
    result = render(trained_model, score)
    joined = "\n".join(result.diagnostics)
    assert "MIDI" in joined
    assert "200" in joined
    assert "absurd" in joined
    # it is not a musical note, so it is sung at the highest one there is --
    # and at no other frequency: the old "clamp into the F0 range" answer
    # would have rendered it at f0_ceil instead
    voiced = result.params.f0[result.params.f0 > 0]
    assert voiced.size > 0
    assert voiced.max() == pytest.approx(
        labels_module.midi_to_hz(labels_module.MIDI_NOTE_MAX), rel=1e-6)
    assert voiced.max() > trained_model.spec.f0_ceil


def test_transpose_beyond_the_valid_midi_range_is_named(trained_model):
    from hms.core.labels import Score, Utterance, Segment

    score = Score([Utterance(name="held", segments=[
        Segment("a", 0.0, 0.5, note=60.0)])])
    result = render(trained_model, score, transpose=140.0)  # 60 -> 200
    joined = "\n".join(result.diagnostics)
    assert "MIDI" in joined                 # the note-level warning
    assert "F0 range" in joined             # the out-of-trained-range report
    voiced = result.params.f0[result.params.f0 > 0]
    assert voiced.max() == pytest.approx(
        labels_module.midi_to_hz(127.0), rel=1e-6)


def test_note_beyond_the_trained_f0_range_is_warned(trained_model):
    """A valid MIDI note above the model's F0 range: warned, then sung as asked."""
    from hms.core.labels import Score, Utterance, Segment

    score = Score([Utterance(name="high", segments=[
        Segment("a", 0.0, 0.5, note=96.0)])])   # ~1975 Hz > 800 Hz ceiling
    result = render(trained_model, score)
    joined = "\n".join(result.diagnostics)
    assert "trained F0 range" in joined
    assert "96" in joined
    # 96 is a valid MIDI number: only the trained-range warning, not the
    # invalid-MIDI one
    assert not any("valid MIDI range" in message
                   for message in result.diagnostics)


def test_normal_score_notes_produce_no_pitch_range_diagnostics(
        trained_model, short_score):
    result = render(trained_model, short_score)
    joined = "\n".join(result.diagnostics)
    assert "valid MIDI range" not in joined
    assert "trained F0 range" not in joined


def test_external_f0_render_does_not_warn_about_the_score_notes(
        trained_model):
    """With an external trajectory the score note does not drive the pitch."""
    from hms.core.labels import Score, Utterance, Segment

    score = Score([Utterance(name="high", segments=[
        Segment("a", 0.0, 0.5, note=200.0)])])
    synthesizer = Synthesizer(
        trained_model, SynthesisConfig(vibrato=False, seed=0,
                                       vocoder="builtin"))
    n_frames = len(synthesizer.plan(score)[0])
    result = synthesizer.synthesize(score, f0=np.full(n_frames, 300.0))
    joined = "\n".join(result.diagnostics)
    assert "external F0 override" in joined
    assert "MIDI" not in joined
    assert "trained F0 range" not in joined


def test_an_unseen_note_is_still_sung(trained_model, short_score):
    """Notes outside the training range must not collapse to the trained one."""
    score = short_score
    result = render(trained_model, score, transpose=7.0)
    deviations = voiced_deviations(result)
    assert np.median(np.abs(deviations)) < 1.5


def test_notes_change_pitch_within_one_render(trained_model):
    """Two different notes in one score must produce two different pitches."""
    from hms.core.labels import Score, Utterance, Segment

    segments = [Segment("sil", 0.0, 0.1),
                Segment("a", 0.1, 0.6, note=57.0),
                Segment("a", 0.6, 1.1, note=64.0),
                Segment("sil", 1.1, 1.2)]
    score = Score([Utterance(name="two_notes", segments=segments)])
    result = render(trained_model, score)
    f0 = result.params.f0
    first = f0[(result.notes < 60) & (f0 > 0)]
    second = f0[(result.notes > 60) & (f0 > 0)]
    assert first.size > 20 and second.size > 20
    expected_ratio = labels_module.midi_to_hz(64) / labels_module.midi_to_hz(57)
    assert np.median(second) / np.median(first) == pytest.approx(
        expected_ratio, rel=0.03)


def test_vibrato_adds_a_modulation_that_is_not_in_the_model(trained_model,
                                                           short_score):
    score = short_score
    flat = render(trained_model, score, vibrato=False)
    wide = render(trained_model, score, vibrato=True, vibrato_depth=0.8,
                  vibrato_rate=5.0)
    def roughness(result):
        f0 = result.f0_semitones
        voiced = np.isfinite(f0)
        values = f0[voiced]
        return np.abs(np.diff(values)).mean()
    assert roughness(wide) > roughness(flat)


def test_vibrato_overrides_do_not_mutate_the_saved_pitch_settings(
        trained_model, short_score):
    before = trained_model.pitch_model.vibrato.to_dict()
    render(trained_model, short_score, vibrato=True, vibrato_depth=1.2,
           vibrato_rate=6.2)
    assert trained_model.pitch_model.vibrato.to_dict() == before


def test_score_driven_synthesis_needs_no_learned_pitch_statistics(
        trained_model, short_score):
    """The base F0 path is exactly the supplied note with an empty pitch model."""
    import copy
    from hms.core.pitch import PitchModel

    model = copy.deepcopy(trained_model)
    model.pitch_model = PitchModel()  # no learned state pitch / voicing priors
    config = SynthesisConfig(vibrato=False, seed=0)
    assert config.f0_source == "score"
    result = Synthesizer(model, config).synthesize(short_score)
    voiced = result.params.f0 > 0
    assert voiced.any()
    expected = labels_module.midi_to_hz(result.notes[voiced])
    assert np.allclose(result.params.f0[voiced], expected, rtol=1e-10)
    assert model.pitch_model.stats == {}


def test_state_means_pitch_source_runs_without_the_acoustic_model(
        trained_model, short_score):
    result = render(trained_model, short_score, f0_source="state_means")
    voiced = result.params.f0 > 0
    assert voiced.any()
    deviations = voiced_deviations(result)
    assert np.median(np.abs(deviations)) < 2.0


def test_vibrato_segment_ids_are_unique_across_utterances(trained_model):
    from hms.core.labels import Score, Segment, Utterance
    from hms.core.pitch import PitchModel, Vibrato

    score = Score([
        Utterance("first", [Segment("a", 0.0, 0.5, note=60.0)]),
        Utterance("second", [Segment("a", 0.0, 0.5, note=60.0)]),
    ])
    synthesizer = Synthesizer(trained_model, SynthesisConfig(vibrato=False))
    phones, _states, segment_ids, _notes, _frames, _diagnostics = \
        synthesizer.plan(score)
    segment_ids = np.asarray(segment_ids)
    unique_ids, counts = np.unique(segment_ids, return_counts=True)
    assert len(unique_ids) == 2
    assert counts[0] == counts[1] == 100

    vibrato = Vibrato(enabled=True, rate_hz=5.0, depth_semitones=0.5,
                      delay_ms=0.0, attack_ms=1.0, randomness=0.0)
    pitch_model = PitchModel(vibrato=vibrato)
    contour = pitch_model._vibrato_for_frames(
        segment_ids, np.ones(len(phones), dtype=bool), 5.0, rng=None)
    first = contour[segment_ids == unique_ids[0]]
    second = contour[segment_ids == unique_ids[1]]
    assert np.allclose(first, second), "vibrato phase should restart per utterance segment"


def test_phoneme_shorter_than_state_count_is_safe_to_synthesize(trained_model):
    from hms.core.labels import Score, Segment, Utterance

    score = Score([Utterance("one_frame", [
        Segment("a", 0.0, 0.005, note=60.0),
    ])])
    result = render(trained_model, score)
    assert len(result.phones) == 1
    assert result.state_ids.tolist() == [0]
    assert len(result.params.f0) == 1
    assert np.isfinite(result.audio).all()


def test_silence_pitch_statistics_do_not_transpose_the_sung_notes(
        trained_model, short_score):
    """Only frames that carry pitch may anchor the state-means contour.

    The contour is re-centred so its mean deviation is zero, but a silence
    state's pitch statistic is interpolated rather than sung and can sit far
    from the note.  Counting those frames in the mean used to shift every sung
    note in the utterance (a model whose ``sil`` mean is +9.7 semitones
    rendered the melody a major third flat).
    """
    import copy

    baseline = render(trained_model, short_score, f0_source="state_means")
    assert np.median(np.abs(voiced_deviations(baseline))) < 1.0

    altered = copy.deepcopy(trained_model)
    silence_stats = altered.pitch_model.stats.get("sil")
    assert silence_stats, "the demo model should have a silence pitch state"
    for state in silence_stats:
        state.mean += 12.0

    # the melody must still land on the requested notes after perturbing a
    # statistic that belongs to frames which carry no pitch at all
    result = render(altered, short_score, f0_source="state_means")
    assert (result.params.f0 > 0).any()
    moved = np.median(np.abs(voiced_deviations(result)))
    assert moved < 1.0, (
        f"the melody sits {moved:.2f} semitones off the requested notes when a "
        f"silence state's pitch mean changes")


def test_duration_model_mode_ignores_the_score_timing(trained_model):
    """A score that holds a note far longer than the model ever saw must be
    rendered at the *model's* duration when duration_mode='model'."""
    from hms.core.labels import Score, Utterance, Segment

    score = Score([Utterance(name="stretched", segments=[
        Segment("sil", 0.0, 0.2),
        Segment("a", 0.2, 4.2, note=60.0),      # 4 s of 'a': never in training
        Segment("sil", 4.2, 4.4)])])
    scored = render(trained_model, score)
    modelled = render(trained_model, score, duration_mode="model")
    assert scored.duration == pytest.approx(4.4, rel=0.02)
    assert modelled.duration < 0.5 * scored.duration
    assert modelled.duration > 0.0


def test_tempo_scales_the_modelled_duration(trained_model, short_score):
    base = render(trained_model, short_score, duration_mode="model")
    faster = render(trained_model, short_score, duration_mode="model",
                    tempo=2.0)
    assert faster.duration == pytest.approx(base.duration / 2, rel=0.05)


def test_unknown_phonemes_fall_back_instead_of_crashing(trained_model):
    from hms.core.labels import Score, Utterance, Segment

    segments = [Segment("sil", 0.0, 0.1),
                Segment("qqq", 0.1, 0.4, note=60.0),
                Segment("a", 0.4, 0.9, note=60.0),
                Segment("sil", 0.9, 1.0)]
    score = Score([Utterance(name="novel", segments=segments)])
    result = render(trained_model, score)
    assert result.audio.size > 0
    assert any("qqq" in message for message in result.diagnostics)
    assert any("backoff" in message.lower() for message in result.diagnostics)


def test_silence_is_rendered_unvoiced(trained_model, short_score):
    result = render(trained_model, short_score)
    silence = np.array([phone == trained_model.phoneme_set.silence
                        for phone in result.phones])
    if silence.sum():                     # the demo score starts and ends with it
        assert not (result.params.f0[silence] > 0).any()


def test_synthesis_is_deterministic(trained_model, short_score):
    first = render(trained_model, short_score)
    second = render(trained_model, short_score)
    assert np.allclose(first.audio, second.audio)


def test_variance_scale_changes_the_trajectory(trained_model, short_score):
    tight = render(trained_model, short_score, variance_scale=0.02)
    loose = render(trained_model, short_score, variance_scale=20.0)
    assert not np.allclose(tight.audio, loose.audio, atol=1e-4)
    assert np.isfinite(loose.audio).all()


def test_params_match_the_feature_spec(trained_model, short_score):
    result = render(trained_model, short_score)
    spec = trained_model.spec
    assert result.params.fft_size == spec.fft_size
    assert result.params.sp.shape[0] == len(result.params.f0)
    assert result.params.ap.shape[1] == spec.n_bins
    assert result.params.fs == spec.fs
    assert result.params.frame_period == pytest.approx(spec.frame_period)
    # every frame got a (phoneme, state) pair, states inside the phoneme's range
    counts = {}
    for phone, state in zip(result.phones, result.state_ids):
        counts.setdefault(phone, set()).add(int(state))
    for phone, states in counts.items():
        assert max(states) < trained_model.get_or_backoff(phone).n_states
