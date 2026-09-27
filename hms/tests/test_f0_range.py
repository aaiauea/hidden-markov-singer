"""Out-of-training-range F0: the requested pitch is the pitch that is sung.

`f0_floor`-`f0_ceil` bound the F0 *analyser* that produced the training
features, not what the synthesizer may be asked for.  Before this was fixed,
`Synthesizer._clamp_pitch` collapsed every out-of-range frame onto the boundary
and `FeatureSpec.decode` then zeroed anything the clamp had missed, so a valid
MIDI note above the ceiling was sung at `f0_ceil` and a valid note below the
floor at `f0_floor`.  These tests pin the replacement behaviour:

    requested F0 -> nearest trained acoustic region -> vocoder gets the
                                                      requested F0

The assertions are made on `SynthesisResult.params.f0` -- the `(f0, sp, ap)`
triple the vocoder actually consumes -- and never on a post-hoc pitch
estimator.  A lag-based tracker can be a third of an octave off on material
this spectral model produces (see the search-boundary test at the end), which
is exactly why measuring the waveform cannot settle what HMS asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core import labels as labels_module
from hms.core.dsp import autocorrelation_f0
from hms.core.labels import Score, Segment, Utterance
from hms.core.synthesizer import (SynthesisConfig, Synthesizer,
                                  external_f0_to_semitones)

F_REF = 261.6255653005986            # C4, the default F0 reference


def render(model, score, f0=None, **config):
    options = dict(vibrato=False, seed=0, vocoder="builtin")
    options.update(config)
    return Synthesizer(model, SynthesisConfig(**options)).synthesize(
        score, f0=f0)


def held_note(note: float, seconds: float = 0.4) -> Score:
    """One vowel, one note, silence on both sides."""
    return Score([Utterance("held", [
        Segment("sil", 0.0, 0.1),
        Segment("a", 0.1, 0.1 + seconds, note=note),
        Segment("sil", 0.1 + seconds, 0.2 + seconds)])])


def voiced_params(result) -> np.ndarray:
    """The F0 the vocoder is given, on the frames that carry one."""
    return result.params.f0[result.params.f0 > 0]


def diagnostics_text(result) -> str:
    return "\n".join(result.diagnostics)


# --------------------------------------------------------------------------
# 1. a normal, in-range pitch is untouched and unremarked upon
# --------------------------------------------------------------------------
def test_in_range_pitch_is_sung_exactly_and_says_nothing(trained_model):
    """The baseline: nothing about the in-range path may change."""
    spec = trained_model.spec
    result = render(trained_model, held_note(60.0))

    expected = labels_module.midi_to_hz(60.0)         # 261.63 Hz, in range
    assert spec.f0_floor < expected < spec.f0_ceil
    assert np.allclose(voiced_params(result), expected, rtol=1e-9)
    assert "F0 range" not in diagnostics_text(result)
    assert "valid MIDI range" not in diagnostics_text(result)
    # the internal trajectory agrees with what the vocoder was handed
    internal = result.f0_semitones[np.isfinite(result.f0_semitones)]
    assert np.allclose(12.0 * np.log2(voiced_params(result) / F_REF),
                       internal, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------
# 2./3./4. valid MIDI notes outside the trained range, in both directions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("note", [24.0, 36.0, 20.0, 10.0, 0.0],
                         ids=["midi24", "midi36", "midi20", "midi10", "midi0"])
def test_a_pitch_below_the_trained_range_is_preserved(trained_model, note):
    """2./7./8.: the low side -- asked for, not pulled up to `f0_floor`."""
    spec = trained_model.spec
    expected = float(labels_module.midi_to_hz(note))
    assert expected < spec.f0_floor                  # genuinely out of range

    result = render(trained_model, held_note(note))

    sung = voiced_params(result)
    assert np.allclose(sung, expected, rtol=1e-9)
    assert sung.min() < spec.f0_floor, "F0 was clipped up to the floor"
    joined = diagnostics_text(result)
    assert "trained F0 range" in joined
    assert "requested F0 is preserved" in joined
    assert f"{expected:.1f} Hz" in joined             # names what was asked for


@pytest.mark.parametrize("note", [80.0, 96.0, 108.0, 120.0, 127.0],
                         ids=["midi80", "midi96", "midi108", "midi120",
                              "midi127"])
def test_a_pitch_above_the_trained_range_is_preserved(trained_model, note):
    """3./4./7./8.: the high side, including the extremes of the MIDI range."""
    spec = trained_model.spec
    expected = float(labels_module.midi_to_hz(note))
    assert expected > spec.f0_ceil                    # genuinely out of range

    result = render(trained_model, held_note(note))

    sung = voiced_params(result)
    assert np.allclose(sung, expected, rtol=1e-9)
    assert sung.max() > spec.f0_ceil, "F0 was clipped down to the ceiling"
    joined = diagnostics_text(result)
    assert "trained F0 range" in joined
    assert "requested F0 is preserved" in joined
    assert f"{expected:.1f} Hz" in joined
    # a valid MIDI note is not invalid input, whatever the model was trained on
    assert "valid MIDI range" not in joined


@pytest.mark.parametrize("note", [0.0, 127.0],
                         ids=["lowest-midi", "highest-midi"])
def test_extreme_but_valid_midi_notes_render_audio(trained_model, note):
    """4./9./10.: the extremes still produce a usable, finite render."""
    result = render(trained_model, held_note(note))

    assert np.isfinite(result.params.f0).all()
    assert np.isfinite(result.params.sp).all()
    assert (result.params.sp >= 0).all()
    assert np.isfinite(result.params.ap).all()
    assert ((result.params.ap >= 0) & (result.params.ap <= 1)).all()
    assert np.isfinite(result.audio).all()
    assert result.audio.size > 0
    assert np.sqrt((result.audio ** 2).mean()) > 0.01, "render is silent"
    assert np.abs(result.audio).max() > 0.05


# --------------------------------------------------------------------------
# 5./6. external F0 out of the trained range, in both directions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("hz", [40.0, 55.0, 20.0, 8.1757989],
                         ids=["40hz", "55hz", "20hz", "midi0"])
def test_external_f0_below_the_trained_range_is_sung_as_supplied(
        trained_model, hz):
    """5./7./8.: an authoritative trajectory is honoured out of range too."""
    spec = trained_model.spec
    assert hz < spec.f0_floor
    score = held_note(60.0)                           # in-range note, ignored
    n_frames = len(render(trained_model, score).params.f0)
    trajectory = np.full(n_frames, hz)

    result = render(trained_model, score, f0=trajectory)

    assert np.allclose(result.params.f0, hz, rtol=1e-9)
    assert result.params.f0.max() < spec.f0_floor
    assert np.allclose(result.f0_semitones[result.params.f0 > 0],
                       12.0 * np.log2(hz / F_REF), rtol=1e-9, atol=1e-12)
    assert "trained F0 range" in diagnostics_text(result)


@pytest.mark.parametrize("hz", [1200.0, 2000.0, 4000.0, 12543.85],
                         ids=["1200hz", "2000hz", "4000hz", "midi127"])
def test_external_f0_above_the_trained_range_is_sung_as_supplied(
        trained_model, hz):
    """6./7./8.: and symmetrically above it."""
    spec = trained_model.spec
    assert hz > spec.f0_ceil
    score = held_note(60.0)
    n_frames = len(render(trained_model, score).params.f0)
    trajectory = np.full(n_frames, hz)

    result = render(trained_model, score, f0=trajectory)

    assert np.allclose(result.params.f0, hz, rtol=1e-9)
    assert result.params.f0.min() > spec.f0_ceil
    assert np.allclose(result.f0_semitones[result.params.f0 > 0],
                       12.0 * np.log2(hz / F_REF), rtol=1e-9, atol=1e-12)
    assert "trained F0 range" in diagnostics_text(result)


def test_external_f0_out_of_range_adds_nothing_on_top(trained_model):
    """The override stays authoritative at out-of-range frequencies."""
    score = held_note(60.0)
    n_frames = len(render(trained_model, score).params.f0)
    trajectory = np.full(n_frames, 3000.0)

    shaken = render(trained_model, score, f0=trajectory, vibrato=True,
                    vibrato_depth=2.0, vibrato_rate=6.0, f0_source="acoustic",
                    pitch_variation=8.0)

    assert np.allclose(shaken.params.f0, trajectory, rtol=1e-9)
    roughness = np.abs(np.diff(shaken.f0_semitones)).mean()
    assert roughness == pytest.approx(0.0, abs=1e-9)
    assert "external F0 override" in diagnostics_text(shaken)


# --------------------------------------------------------------------------
# 7./8. the F0 that survives is the one the vocoder receives
# --------------------------------------------------------------------------
def test_the_vocoder_receives_the_requested_f0_not_the_boundary(trained_model):
    """A recording spy on the vocoder, one frame at a time.

    `SynthesisResult.params` *is* the argument handed to `Vocoder.synthesize`,
    so asserting on it is asserting on the real thing -- but the point is
    worth making explicit by also capturing the call.
    """
    seen = {}

    class Spy:
        name = "spy"
        fft_size = trained_model.spec.fft_size

        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, item):
            return getattr(self.inner, item)

        def synthesize(self, params):
            seen["f0"] = np.array(params.f0, dtype=np.float64)
            return self.inner.synthesize(params)

    note = 105.0                                    # ~2217 Hz, way above 800
    expected = float(labels_module.midi_to_hz(note))
    synthesizer = Synthesizer(trained_model,
                              SynthesisConfig(vibrato=False, seed=0,
                                              vocoder="builtin"))
    synthesizer._vocoder = Spy(synthesizer.vocoder)
    result = synthesizer.synthesize(held_note(note))

    assert np.allclose(seen["f0"][seen["f0"] > 0], expected, rtol=1e-9)
    assert np.array_equal(seen["f0"], result.params.f0)
    assert seen["f0"].max() > trained_model.spec.f0_ceil
    assert seen["f0"].min() == 0.0                   # only the sil frames


def test_a_mixed_range_score_keeps_every_note_where_it_was_asked_for(
        trained_model):
    """In-range and out-of-range notes in one render, none of them moved."""
    spec = trained_model.spec
    notes = [60.0, 100.0, 33.0, 76.0]      # inside, above, below, inside
    assert labels_module.midi_to_hz(100.0) > spec.f0_ceil
    assert labels_module.midi_to_hz(33.0) < spec.f0_floor
    segments = [Segment("sil", 0.0, 0.1)]
    for index, note in enumerate(notes):
        segments.append(Segment("a", 0.1 + 0.3 * index,
                                0.1 + 0.3 * (index + 1), note=note))
    segments.append(Segment("sil", 0.1 + 0.3 * len(notes),
                            0.2 + 0.3 * len(notes)))

    result = render(trained_model, Score([Utterance("mixed", segments)]))

    f0 = result.params.f0
    for note in notes:
        here = f0[result.notes == note]
        here = here[here > 0]
        assert here.size > 20, f"MIDI {note} lost its frames"
        assert np.allclose(here, labels_module.midi_to_hz(note), rtol=1e-9)
    assert f0.max() > spec.f0_ceil
    assert f0[f0 > 0].min() < spec.f0_floor
    assert not any("valid MIDI range" in message
                   for message in result.diagnostics)


# --------------------------------------------------------------------------
# 9./10. the acoustics around an out-of-range F0 stay usable
# --------------------------------------------------------------------------
def test_out_of_range_notes_reuse_finite_boundary_acoustics(trained_model):
    """9.: the envelope is the model's, not the requested note's, and finite."""
    in_range = render(trained_model, held_note(69.0))         # ~440 Hz
    high = render(trained_model, held_note(100.0))           # ~2637 Hz
    low = render(trained_model, held_note(24.0))             # ~32.7 Hz

    for result in (high, low):
        assert np.isfinite(result.params.sp).all()
        assert np.isfinite(result.params.ap).all()
        assert (result.params.sp > 0).all()
        assert ((result.params.ap >= 0) & (result.params.ap <= 1)).all()
        assert np.isfinite(result.audio).all()
        assert np.sqrt((result.audio ** 2).mean()) > 0.01
    # only the F0 changed: the acoustic parameters are the same vowel either
    # way, which is precisely the "nearest trained region" rule
    assert np.allclose(high.params.sp, in_range.params.sp, rtol=1e-9)
    assert np.allclose(low.params.ap, in_range.params.ap, rtol=1e-9)
    assert not np.allclose(high.params.f0, in_range.params.f0)


def test_the_range_check_is_a_no_op_for_an_in_range_trajectory(
        trained_model):
    """11.: the OOV path must not perturb the in-range path at all."""
    score = held_note(60.0)
    synthesizer = Synthesizer(trained_model, SynthesisConfig(vibrato=False,
                                                            seed=0))
    _phones, _states, _segments, notes, _frames, diagnostics = \
        synthesizer.plan(score)
    f0_semitones = 12.0 * np.log2(labels_module.midi_to_hz(notes) / F_REF)

    assert synthesizer._out_of_range_pitch_diagnostics(f0_semitones) == []
    assert diagnostics == []


# --------------------------------------------------------------------------
# 12. external-F0 validation is unchanged
# --------------------------------------------------------------------------
@pytest.mark.parametrize("hz", [2000.0, 12.0, 40.0, 12543.85],
                         ids=["above", "below-floor", "far-below", "midi127"])
def test_valid_out_of_range_external_f0_is_not_a_validation_error(
        trained_model, hz):
    """Out of range is not malformed: it must render, not raise."""
    score = held_note(60.0)
    n_frames = len(render(trained_model, score).params.f0)
    result = render(trained_model, score, f0=np.full(n_frames, hz))
    assert np.allclose(result.params.f0, hz, rtol=1e-9)


@pytest.mark.parametrize("bad,message", [
    (np.nan, "must be finite"),
    (np.inf, "must be finite"),
    (-1.0, "non-negative"),
])
def test_malformed_external_f0_is_still_rejected_out_of_range(
        trained_model, bad, message):
    """12.: a valid trajectory with one malformed frame is still an error."""
    score = held_note(60.0)
    n_frames = len(render(trained_model, score).params.f0)
    trajectory = np.full(n_frames, 2000.0)        # out of range throughout
    trajectory[5] = bad

    with pytest.raises(ValueError, match=message) as info:
        render(trained_model, score, f0=trajectory)
    assert "frame 5" in str(info.value)


def test_malformed_external_f0_shape_and_length_are_unchanged(trained_model):
    """12.: the checks that predate OOV handling still fire."""
    score = held_note(60.0)
    n_frames = len(render(trained_model, score).params.f0)

    with pytest.raises(ValueError, match=r"external F0 has \d+ frame\(s\) but"):
        render(trained_model, score, f0=np.full(n_frames + 3, 3000.0))
    with pytest.raises(ValueError, match="one value per frame"):
        render(trained_model, score, f0=np.full((n_frames, 1), 3000.0))
    with pytest.raises(ValueError, match="empty"):
        render(trained_model, score, f0=np.zeros(0))
    with pytest.raises(ValueError, match="real numbers"):
        render(trained_model, score, f0=["a", "b", "c"])
    with pytest.raises(ValueError, match="one value per frame"):
        render(trained_model, score, f0=3000.0)


def test_the_conversion_helper_keeps_out_of_range_values(trained_model):
    """`external_f0_to_semitones` itself must not range-check."""
    spec = trained_model.spec
    values = [0.0, 40.0, 261.6255653005986, 4000.0]
    semitones = external_f0_to_semitones(values, len(values), spec)

    assert np.isnan(semitones[0])                    # 0 Hz is the unvoiced mark
    assert np.isfinite(semitones[1:]).all()
    for value, semi in zip(values[1:], semitones[1:]):
        assert semi == pytest.approx(12.0 * np.log2(value / F_REF))


# --------------------------------------------------------------------------
# where an unexpected measured F0 comes from (the 261.6 -> 816.7 report)
# --------------------------------------------------------------------------
def test_a_c4_render_really_asks_the_vocoder_for_c4(trained_model):
    """The internal and vocoder F0 for the reported case, exactly.

    A build reported "requested ~261.6 Hz, measured ~816.7 Hz" for this note.
    HMS's own trajectory is unambiguous: 261.6256 Hz, which is what the vocoder
    is given.  See the next test for where 816.7 comes from.
    """
    result = render(trained_model, held_note(60.0))

    expected = float(labels_module.midi_to_hz(60.0))
    assert expected == pytest.approx(261.6256, abs=1e-4)
    assert np.allclose(voiced_params(result), expected, rtol=1e-9)
    assert result.params.f0.max() == pytest.approx(expected, rel=1e-9)


def test_816_hz_is_the_estimators_search_boundary_not_a_generated_f0():
    """816.7 Hz is `fs / int(fs / f0_ceil)` -- the shortest lag it can report.

    The numpy autocorrelation estimator searches lags
    ``[int(fs / f0_ceil), int(fs / f0_floor)]``.  At 22.05 kHz -- the rate the
    test fixtures and many real corpora use -- with the default 800 Hz ceiling
    the first lag is ``int(22050 / 800) = 27`` samples, and ``22050 / 27 =
    816.667 Hz``: the reported value exactly, to the digit.  A tracker that
    locks onto that lag is reporting its own search boundary (a sub-harmonic
    of C4's ~84-sample period), not an F0 HMS produced.  That is why the
    waveform measurement must never be used as evidence about what HMS asked
    the vocoder for.
    """
    fs, ceiling = 22050, 800.0
    lag_min = max(1, int(fs / ceiling))
    assert lag_min == 27
    assert fs / lag_min == pytest.approx(816.667, abs=1e-3)

    # the estimator really does emit that value when the fundamental is
    # missing: harmonics 3/5/7 of C4 leave the CMNFD nothing at the true
    # period, and the frame at the signal's edge falls onto the shortest lag
    t = np.arange(fs // 2) / fs
    tone = sum(np.sin(2 * np.pi * 261.6255653 * k * t) / k
               for k in (3, 5, 7))
    track = autocorrelation_f0(tone, fs, 4 * int(fs * 0.005),
                               int(fs * 0.005), 71.0, ceiling)
    voiced = track[track > 0]
    assert np.isclose(voiced, fs / lag_min, atol=0.5).any()
    # ... while the frames that do carry a period are read correctly
    assert np.median(voiced) == pytest.approx(262.5, rel=0.02)


def test_a_pitch_estimate_is_not_used_to_judge_the_synthesis(trained_model):
    """Sanity check on the diagnostic: the estimate can be an octave out.

    HMS's F0 is exact; the estimator's is not.  This test documents the gap so
    that a future "the rendered pitch is wrong" report is investigated in the
    estimator first, and changes the synthesis F0 second (or never).
    """
    result = render(trained_model, held_note(60.0))
    expected = float(labels_module.midi_to_hz(60.0))
    hop = int(result.params.fs * result.params.frame_period / 1000.0)
    estimate = autocorrelation_f0(result.audio, result.params.fs, 4 * hop,
                                  hop, trained_model.spec.f0_floor,
                                  trained_model.spec.f0_ceil)
    estimate = estimate[estimate > 0]

    assert np.allclose(voiced_params(result), expected, rtol=1e-9)
    # whatever the estimator says, the render is what the vocoder was asked for
    assert estimate.size > 0
    assert abs(float(np.median(estimate)) / expected - 1.0) < 0.25
