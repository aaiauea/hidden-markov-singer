"""Score/label parsing, MIDI conversion and the phoneme inventory."""

from __future__ import annotations

import numpy as np
import pytest

from hms.core import labels as labels_module
from hms.core.phonemes import PhonemeSet

SCORE_TEXT = """\
# utt_id  onset   offset  phone   note
phrase    0.000   0.100   sil     -
phrase    0.100   0.160   m       60
phrase    0.160   0.560   a       60
phrase    0.560   0.620   n       60
phrase    0.620   1.020   a       62
phrase    1.020   1.120   sil     -
"""


def sample_score():
    return labels_module.parse(SCORE_TEXT)


def test_parse_score():
    score = sample_score()
    assert len(score) == 1
    utterance = score["phrase"]
    assert utterance.name == "phrase"
    assert [s.phone for s in utterance.segments] == [
        "sil", "m", "a", "n", "a", "sil"]
    assert [s.note for s in utterance.segments] == [None, 60.0, 60.0, 60.0,
                                                    62.0, None]
    assert utterance.segments[2].note == pytest.approx(60.0)
    assert utterance.segments[0].note is None
    assert utterance.start == pytest.approx(0.0)
    assert utterance.end == pytest.approx(1.12)
    assert score.total_duration == pytest.approx(1.12)
    assert score.phone_count["a"] == 2
    assert score.diagnostics == []


def test_score_roundtrip_through_lines(tmp_path):
    score = sample_score()
    path = tmp_path / "roundtrip.tsv"
    score.save(path)
    reloaded = labels_module.load(path)
    assert [s.phone for s in reloaded["phrase"].segments] == \
        [s.phone for s in score["phrase"].segments]
    assert np.allclose([s.start for s in reloaded["phrase"].segments],
                       [s.start for s in score["phrase"].segments])


def test_parser_reports_problems_instead_of_crashing():
    text = "\n".join([
        "phrase 0.0 0.1 a 60",
        "phrase 0.1 0.05 b 60",          # non-positive duration -> dropped
        "bad_line",                       # too few columns
        "phrase nope 0.2 c 60",           # unparseable time
    ])
    score = labels_module.parse(text)
    assert len(score.diagnostics) == 3
    assert [s.phone for s in score["phrase"].segments] == ["a"]


def test_reversed_segment_is_reported_with_line_number():
    text = "u 0.0 0.1 a 60\nu 0.5 0.2 b 60\n"
    score = labels_module.parse(text)
    assert [s.phone for s in score["u"].segments] == ["a"]
    assert len(score.diagnostics) == 1
    message = score.diagnostics[0]
    assert "line 2" in message
    assert "before onset" in message


def test_zero_duration_segment_is_reported_with_line_number():
    text = "u 0.0 0.1 a 60\nu 0.4 0.4 b 60\n"
    score = labels_module.parse(text)
    assert [s.phone for s in score["u"].segments] == ["a"]
    assert len(score.diagnostics) == 1
    message = score.diagnostics[0]
    assert "line 2" in message
    assert "zero-duration" in message


def test_negative_times_are_reported_and_dropped():
    score = labels_module.parse("u -0.1 0.5 a 60\n")
    assert len(score) == 0
    assert len(score.diagnostics) == 1
    message = score.diagnostics[0]
    assert "line 1" in message
    assert "non-negative" in message


def test_out_of_midi_range_note_is_reported_with_line_number():
    text = "u 0.0 0.5 a 60\nu 0.5 1.0 a 200\n"
    score = labels_module.parse(text)
    assert [s.note for s in score["u"].segments] == [60.0]
    message = score.diagnostics[0]
    assert "line 2" in message
    assert "[0, 127]" in message
    assert "200" in message


def test_gap_between_segments_is_reported_but_segments_kept():
    score = labels_module.parse("u 0.0 0.2 a 60\nu 0.5 0.7 a 64\n")
    # the gap is tolerated -- documented behaviour -- but no longer silent
    assert [s.phone for s in score["u"].segments] == ["a", "a"]
    assert len(score.diagnostics) == 1
    message = score.diagnostics[0]
    assert message.startswith("u:")
    assert "gap" in message
    assert "0.300" in message
    # and the fill behaviour itself is unchanged: carried note, first phone
    phones, notes, _ = score["u"].frame_labels(10.0, default_note=60.0)
    assert phones[30] == "a"
    assert notes[30] == pytest.approx(60.0)


def test_contiguous_and_adjacent_labels_produce_no_timing_diagnostics():
    score = sample_score()
    assert score.diagnostics == []
    # exact adjacency (shared boundary) is not an overlap or a gap
    score = labels_module.parse("u 0.0 0.2 a 60\nu 0.2 0.4 b 60\n")
    assert score.diagnostics == []


def test_garbage_extra_column_is_reported_and_row_kept():
    score = labels_module.parse("u 0.0 0.5 a 60 garbage\n")
    assert [s.phone for s in score["u"].segments] == ["a"]
    assert score["u"].segments[0].note == pytest.approx(60.0)
    assert len(score.diagnostics) == 1
    message = score.diagnostics[0]
    assert "line 1" in message
    assert "garbage" in message


def test_malformed_context_column_is_reported_and_valid_context_kept():
    score = labels_module.parse("u 0.0 0.5 a 60 =x stress=1\n")
    assert score["u"].segments[0].context == {"stress": "1"}
    assert len(score.diagnostics) == 1
    message = score.diagnostics[0]
    assert "line 1" in message
    assert "=x" in message


def test_valid_midi_boundary_and_fractional_notes_are_accepted():
    score = labels_module.parse("u 0.0 0.1 a 0\n"
                                "u 0.1 0.2 a 127\n"
                                "u 0.2 0.3 a 60.5\n"
                                "u 0.3 0.4 sil -\n")
    assert score.diagnostics == []
    assert [s.note for s in score["u"].segments] == [0.0, 127.0, 60.5, None]


def test_frame_time_unit_converts_to_seconds_using_the_analysis_period():
    score = labels_module.parse("u\t0\t20\ta\t60\n",
                                time_unit="frames", frame_period=5.0)
    segment = score["u"].segments[0]
    assert segment.start == pytest.approx(0.0)
    assert segment.end == pytest.approx(0.1)
    spans = labels_module.segment_boundaries(score["u"], frame_period=5.0)
    assert spans == [("a", 0, 20, 60.0)]


def test_nonfinite_times_and_malformed_notes_are_diagnostics():
    text = "\n".join([
        "u 0.0 0.1 a not-a-note",
        "u nan 0.2 a 60",
        "u 0.0 inf a 60",
        "u 0.0 0.1 a inf",
        "u 0.0 0.1 a 128",
    ])
    score = labels_module.parse(text)
    assert len(score) == 0
    assert len(score.diagnostics) == 5
    assert any("bad MIDI note" in item for item in score.diagnostics)
    assert any("finite" in item for item in score.diagnostics)
    assert any("[0, 127]" in item for item in score.diagnostics)


def test_invalid_frame_period_is_rejected_for_frame_labels():
    with pytest.raises(ValueError, match="frame_period"):
        labels_module.parse("u 0 10 a 60\n", time_unit="frames",
                            frame_period=0.0)


def test_context_columns_are_captured():
    score = labels_module.parse("u 0.0 0.5 a 60 stress=1 word=la\n")
    assert score["u"].segments[0].context == {"stress": "1", "word": "la"}


def test_gaps_carry_the_first_phoneme_and_the_previous_note():
    """A hole between two labels must not shift every later frame."""
    score = labels_module.parse("u 0.0 0.2 a 60\nu 0.4 0.6 a 64\n")
    phones, notes, start = score["u"].frame_labels(10.0, default_note=60.0)
    assert len(phones) == 60
    assert phones[10] == "a"
    assert phones[30] == "a"                 # the gap is still 'a' at 0.3 s
    assert notes[30] == pytest.approx(60.0)  # previous note carried across
    assert phones[-1] == "a"


def test_frame_labels_match_the_timeline():
    utterance = sample_score()["phrase"]
    phones, notes, start = utterance.frame_labels(5.0, default_note=60.0)
    assert start == pytest.approx(0.0)
    assert len(phones) == int(round(1.12 / 0.005)) == 224
    assert len(notes) == len(phones)
    assert phones[0] == "sil" and phones[-1] == "sil"
    assert phones[int(round(0.30 / 0.005))] == "a"
    assert notes[int(round(0.30 / 0.005))] == pytest.approx(60.0)
    assert notes[int(round(0.80 / 0.005))] == pytest.approx(62.0)
    # unvoiced segments fall back to the default note rather than NaN
    assert notes[0] == pytest.approx(60.0)


def test_frame_labels_respect_a_custom_frame_count():
    utterance = sample_score()["phrase"]
    phones, notes, _ = utterance.frame_labels(5.0, 60.0, n_frames=10)
    assert len(phones) == 10 and len(notes) == 10


def test_note_sequence_and_segment_boundaries():
    utterance = sample_score()["phrase"]
    phones, notes = labels_module.note_sequence(utterance, 5.0, 60.0)
    assert phones[100] == "a" and notes[100] == pytest.approx(60.0)
    spans = labels_module.segment_boundaries(utterance, 5.0)
    assert len(spans) == len(utterance.segments)
    for (phone, lo, hi, note), segment in zip(spans, utterance.segments):
        assert phone == segment.phone
        assert hi > lo
    # a 120 ms silence is 24 frames at 5 ms
    assert spans[0][2] - spans[0][1] == 20


def test_midi_helpers():
    assert labels_module.midi_to_hz(60) == pytest.approx(261.6255653, rel=1e-8)
    assert labels_module.midi_to_hz(69) == pytest.approx(440.0)
    assert float(labels_module.hz_to_midi(440.0)) == pytest.approx(69.0)
    assert float(labels_module.hz_to_midi(261.6255653)) == pytest.approx(60.0)
    assert np.allclose(labels_module.midi_to_hz([60, 72]), [261.6255653,
                                                            523.2511306],
                       rtol=1e-7)


def test_phoneme_inventory(phoneme_set):
    assert "a" in phoneme_set
    assert len(phoneme_set) == 18
    vowel = phoneme_set.resolve("a")
    assert vowel is not None and vowel.type == "vowel"
    assert vowel.n_states >= 3 and vowel.n_components >= 1
    assert vowel.voiced and vowel.can_hold_note
    # aliases point at real phonemes
    assert phoneme_set.resolve("pau") is phoneme_set.resolve("sil")
    assert phoneme_set.canonical("_") == "sil"
    assert phoneme_set.canonical("A") == "a"
    # unknown symbols are reported, not invented
    assert phoneme_set.resolve("zzz") is None
    assert phoneme_set.canonical("zzz") == "zzz"
    assert phoneme_set.unknown(["a", "zzz", "pau"]) == ["zzz"]
    # defaults for unknown symbols stay conservative (unvoiced, 2 states)
    assert phoneme_set.n_states("zzz") == 2
    assert phoneme_set.is_voiced("zzz") is False


def test_phoneme_classes_and_summary(phoneme_set):
    vowels = phoneme_set.of_type("vowel")
    assert {p.symbol for p in vowels} == {"a", "e", "i", "o", "u"}
    assert all(p.n_states == 5 for p in vowels)
    assert phoneme_set.of_type("silence")[0].symbol == "sil"
    assert set(phoneme_set.symbols) == {p.symbol for p in phoneme_set}
    summary = "\n".join(phoneme_set.summary())
    assert "vowel" in summary and "sil" in summary


def test_phoneme_set_roundtrip_is_lossless(phoneme_set):
    restored = PhonemeSet.from_dict(phoneme_set.to_dict())
    assert restored.symbols == phoneme_set.symbols
    assert restored.silence == phoneme_set.silence
    for symbol in phoneme_set.symbols:
        original, copy = phoneme_set.resolve(symbol), restored.resolve(symbol)
        assert original.n_states == copy.n_states
        assert original.n_components == copy.n_components
        assert original.voiced == copy.voiced
        assert original.type == copy.type


def test_new_phonemes_can_be_added_without_touching_the_engine(phoneme_set):
    data = phoneme_set.to_dict()
    data["phonemes"]["q"] = {"type": "unvoiced_consonant", "n_states": 2,
                             "n_components": 1, "voiced": False}
    extended = PhonemeSet.from_dict(data)
    assert len(extended) == len(phoneme_set) + 1
    assert extended.resolve("q").n_states == 2


def test_broken_inventory_is_rejected():
    with pytest.raises(ValueError):
        PhonemeSet.from_dict({"phonemes": {}})
    with pytest.raises(ValueError, match="unknown type"):
        PhonemeSet.from_dict({"phonemes": {"a": {"type": "vocalic"}}})
    with pytest.raises(ValueError, match="must be positive"):
        PhonemeSet.from_dict({"phonemes": {
            "a": {"type": "vowel", "n_states": 0},
        }})
