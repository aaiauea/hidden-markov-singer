"""Optional pitch-conditioned acoustic model selection (experimental).

Covers the deterministic pitch-bin calculation and its validation, the policy
deciding which segments carry a condition at all (scored notes only -- never
silence, rests or measured F0), the trainer's grouping/selection/training of
``(unit, pitch bin)`` buckets, the resolution hierarchy and its fallbacks, the
format-4 serialisation (with format-2/3 compatibility), the unchanged synthesis
behaviour when the feature is off, the interaction with sparse contexts,
evaluation reporting, determinism, and the ``hms train --pitch-conditioning``
CLI surface.

Nothing here asserts that pitch conditioning sounds better: it asserts that the
mechanism is deterministic, that it selects what it was trained on, and that
missing data falls back instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hms.cli.main import main
from hms.config import load_parameters, training_config_from_parameters
from hms.core import labels as labels_module
from hms.core.context import (KIND_LEFT, KIND_RIGHT, KIND_TRIPHONE,
                              context_wildcard, left_diphone_key,
                              right_diphone_key)
from hms.core.evaluate import check_model_compatibility, evaluate_models
from hms.core.features import FeatureSpec
from hms.core.hmm import LeftToRightHMM
from hms.core.labels import MIDI_NOTE_MAX, MIDI_NOTE_MIN, Score, Segment, Utterance
from hms.core.model import MODEL_FORMAT_VERSION, HMSModel
from hms.core.phonemes import PhonemeSet
from hms.core.pitch_condition import (DEFAULT_BIN_SIZE, KIND_PHONE,
                                      MAX_BIN_SIZE, PITCH_TIER_SUFFIX,
                                      PitchConditioning, bin_note_bounds,
                                      effective_note, is_pitch_tier, is_silence,
                                      n_pitch_bins, pitch_bin, pitch_tier,
                                      segment_pitch_bin, validate_bin_size)
from hms.core.synthesizer import SynthesisConfig, Synthesizer
from hms.core.trainer import Trainer, TrainingConfig, UtteranceData
from hms.data import wavio
from hms.data.demo_singer import DemoSinger, SegmentSpec, SingerConfig


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

#: Two pitch regions far enough apart to land in different bins of 6 semitones.
LOW_NOTE = 48            # bin 8  (MIDI 48-53)
HIGH_NOTE = 72           # bin 12 (MIDI 72-77)
MIDDLE_NOTE = 60         # bin 10 (MIDI 60-65): deliberately never trained
LOW_BIN = pitch_bin(LOW_NOTE, DEFAULT_BIN_SIZE)
HIGH_BIN = pitch_bin(HIGH_NOTE, DEFAULT_BIN_SIZE)
MIDDLE_BIN = pitch_bin(MIDDLE_NOTE, DEFAULT_BIN_SIZE)

TEST_FS = 22050
TEST_FFT = 1024


def pitch_phoneme_set() -> PhonemeSet:
    return PhonemeSet.from_dict({
        "defaults": {
            "vowel": {"n_states": 3, "n_components": 1, "voiced": True},
            "unvoiced_consonant": {"n_states": 2, "n_components": 1,
                                   "voiced": False},
            "silence": {"n_states": 1, "n_components": 1, "voiced": False},
        },
        "phonemes": {
            "sil": {"type": "silence"},
            "a": {"type": "vowel"},
            "i": {"type": "vowel"},
            # in the inventory but never given a dedicated HMM in these tests:
            # it exercises the phone-class backoff rung
            "rare": {"type": "vowel"},
            "s": {"type": "unvoiced_consonant"},
        },
        # `_` is an alias here exactly as in the bundled inventory, so the
        # context wildcard falls through to `?` -- the same path a real model
        # takes.
        "aliases": {"A": "a", "pau": "sil", "_": "sil"},
    })


def tiny_spec() -> FeatureSpec:
    return FeatureSpec(fs=TEST_FS, frame_period=5.0, fft_size=TEST_FFT,
                       n_mcep=2, n_band=2, use_delta=False, use_delta2=False)


def make_utterance(name, spans, notes=None, value=0.0, voiced=True) -> UtteranceData:
    """A synthetic analysed utterance; only features/voiced/spans/notes matter.

    ``spans`` are ``(phone, lo, hi)``; ``notes`` is the matching list of scored
    notes (``None`` per span means "no note", exactly as in a label file).
    """
    total = max(hi for _phone, _lo, hi in spans)
    dim = tiny_spec().dim
    span_notes = list(notes) if notes is not None else [None] * len(spans)
    assert len(span_notes) == len(spans)
    return UtteranceData(
        name=name, f0=np.zeros(total), voiced=np.full(total, bool(voiced)),
        relative_pitch=np.zeros(total),
        features=np.full((total, dim), value), phoneme_spans=list(spans),
        span_notes=span_notes)


def pitch_trainer(**overrides) -> Trainer:
    config = TrainingConfig(normalize_scale=False, n_iterations=1,
                            pitch_conditioning_enabled=True, **overrides)
    trainer = Trainer(config, pitch_phoneme_set())
    trainer.spec = tiny_spec()
    return trainer


def simple_hmm(n_states=1) -> LeftToRightHMM:
    return LeftToRightHMM(n_states=n_states)


def make_pitch_model(bins=(LOW_BIN, HIGH_BIN), contexts=False,
                     pitch=True, with_phone_model=True) -> HMSModel:
    """A hand-built model with a dedicated phone HMM and per-bin phone HMMs."""
    phoneme_set = pitch_phoneme_set()
    hmms = {"a": simple_hmm(), "i": simple_hmm()} if with_phone_model else {}
    wildcard = context_wildcard(phoneme_set.phonemes)
    left_key = left_diphone_key("sil", "a", wildcard)
    context_hmms = {}
    context_index = {}
    if contexts:
        context_hmms = {"sil^a^sil": simple_hmm(), left_key: simple_hmm()}
        context_index = {
            "sil^a^sil": {"kind": KIND_TRIPHONE, "frames": 300,
                          "occurrences": 5},
            left_key: {"kind": KIND_LEFT, "frames": 200, "occurrences": 5},
        }
    pitch_models = {}
    pitch_index = {}
    if pitch:
        for index in bins:
            pitch_models[("a", index)] = simple_hmm()
            pitch_index[("a", index)] = {
                "kind": KIND_PHONE, "unit": "a", "curr": "a",
                "pitch_bin": index, "frames": 120, "occurrences": 2}
    return HMSModel(
        name="pitch", spec=tiny_spec(), phoneme_set=phoneme_set, hmms=hmms,
        backoff={"vowel": simple_hmm()}, contexts=context_hmms,
        context_index=context_index, global_backoff=simple_hmm(),
        pitch_models=pitch_models, pitch_index=pitch_index,
        pitch_conditioning=PitchConditioning(enabled=pitch,
                                             bin_size=DEFAULT_BIN_SIZE))


def condition_unit(model: HMSModel, unit: str, kind: str, pitch_bin_index: int,
                   frames: int = 90) -> None:
    """Add one pitch-conditioned model to a hand-built model."""
    model.pitch_models[(unit, pitch_bin_index)] = simple_hmm()
    model.pitch_index[(unit, pitch_bin_index)] = {
        "kind": kind, "unit": unit, "curr": "a",
        "pitch_bin": pitch_bin_index, "frames": frames, "occurrences": 3}


def two_note_corpus(directory: Path, notes=(LOW_NOTE, HIGH_NOTE),
                    with_scored_silence: bool = False) -> dict:
    """A tiny synthetic corpus singing one phone in two pitch regions.

    Rendered with the bundled demo singer (the same source the other fixtures
    use), so the two notes really do produce different spectral material while
    staying a few hundred kilobytes of audio.
    """
    wav_dir = directory / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    singer = DemoSinger(SingerConfig(fs=TEST_FS, seed=5))
    lines = ["# utt_id\tonset\toffset\tphone\tnote"]
    for note in notes:
        script = [SegmentSpec("sil", 120, note if with_scored_silence else None),
                  SegmentSpec("a", 600, note),
                  SegmentSpec("sil", 100, note if with_scored_silence else None),
                  SegmentSpec("a", 600, note + 1),
                  SegmentSpec("sil", 140, note if with_scored_silence else None)]
        name = f"note_{note}"
        wavio.write_wav(wav_dir / f"{name}.wav", singer.render(script), TEST_FS)
        lines.extend(singer.label_lines(name, script))
    label_path = directory / "labels.tsv"
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"labels": str(label_path), "wav_dir": str(wav_dir)}


def training_config(corpus: dict, **overrides) -> TrainingConfig:
    values = dict(label_file=corpus["labels"], wav_dir=corpus["wav_dir"],
                  fs=TEST_FS, fft_size=TEST_FFT, n_mcep=8, n_band=3,
                  n_iterations=2, min_phoneme_frames=10, seed=0,
                  vocoder="builtin", pitch_conditioning_enabled=True,
                  pitch_conditioning_bin_size=DEFAULT_BIN_SIZE)
    values.update(overrides)
    return TrainingConfig(**values)


def note_score(note: float, phone: str = "a") -> Score:
    """A one-note score: silence, the phone on ``note``, silence."""
    return Score([Utterance("probe", [
        Segment("sil", 0.0, 0.1),
        Segment(phone, 0.1, 0.6, note=float(note)),
        Segment("sil", 0.6, 0.7)])])


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict:
    return two_note_corpus(tmp_path_factory.mktemp("pitch_corpus"))


@pytest.fixture(scope="module")
def pitch_trained(corpus, tmp_path_factory):
    """A model trained on the two-note corpus *with* pitch conditioning."""
    config = training_config(corpus)
    return Trainer(config, pitch_phoneme_set()).train()


@pytest.fixture(scope="module")
def plain_trained(corpus):
    """The same corpus and budget, with pitch conditioning left off."""
    config = training_config(corpus, pitch_conditioning_enabled=False)
    return Trainer(config, pitch_phoneme_set()).train()


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_pitch_conditioning_defaults_off_and_is_validated():
    config = TrainingConfig()
    assert config.pitch_conditioning_enabled is False
    assert config.pitch_conditioning_bin_size == DEFAULT_BIN_SIZE == 6

    # a whole number of semitones is normalised, not silently truncated
    assert TrainingConfig(pitch_conditioning_bin_size=12
                          ).pitch_conditioning_bin_size == 12
    for bad in (0, -1, 6.5, MAX_BIN_SIZE + 1, "wide", None):
        with pytest.raises(ValueError, match="bin_size"):
            TrainingConfig(pitch_conditioning_bin_size=bad)


def test_bundled_parameters_keep_pitch_conditioning_disabled():
    config = training_config_from_parameters(load_parameters())
    assert config.pitch_conditioning_enabled is False
    assert config.pitch_conditioning_bin_size == DEFAULT_BIN_SIZE


def test_pitch_conditioning_section_of_parameters_yaml_is_loaded():
    config = training_config_from_parameters(
        {"pitch_conditioning": {"enabled": True, "bin_size": 3}})
    assert config.pitch_conditioning_enabled is True
    assert config.pitch_conditioning_bin_size == 3

    # a null value keeps the default instead of disabling validation
    config = training_config_from_parameters(
        {"pitch_conditioning": {"enabled": True, "bin_size": None}})
    assert config.pitch_conditioning_bin_size == DEFAULT_BIN_SIZE


def test_unknown_pitch_conditioning_option_is_rejected():
    with pytest.raises(ValueError, match="unknown pitch_conditioning option"):
        training_config_from_parameters({"pitch_conditioning": {"bogus": 1}})
    with pytest.raises(ValueError, match="must be a mapping"):
        training_config_from_parameters({"pitch_conditioning": [1, 2]})


# --------------------------------------------------------------------------
# the bin calculation
# --------------------------------------------------------------------------


def test_pitch_bin_boundaries_are_exact_and_documented():
    assert LOW_BIN == 8 and HIGH_BIN == 12 and MIDDLE_BIN == 10
    # the low edge of a bin and the note below it
    assert pitch_bin(60, 6) == 10 and pitch_bin(59, 6) == 9
    assert pitch_bin(65, 6) == 10 and pitch_bin(66, 6) == 11
    assert pitch_bin(0, 6) == 0 and pitch_bin(5, 6) == 0 and pitch_bin(6, 6) == 1
    # the whole MIDI range stays supported, including both ends
    assert pitch_bin(MIDI_NOTE_MIN, 6) == 0
    assert pitch_bin(MIDI_NOTE_MAX, 6) == n_pitch_bins(6) - 1 == 21
    # equal notes always map to the same bin, however they are spelled
    assert pitch_bin(60, 6) == pitch_bin(60.0, 6) == pitch_bin(np.int64(60), 6)
    assert pitch_bin(60, 6) == pitch_bin(np.float64(60.0), 6)


def test_pitch_bin_follows_the_configured_bin_size():
    assert n_pitch_bins(1) == 128 and n_pitch_bins(12) == 11
    assert n_pitch_bins(6) == 22 and n_pitch_bins(MAX_BIN_SIZE) == 1
    for note in (0, 30, 60, 61, 72, 127):
        assert pitch_bin(note, 1) == note            # one bin per semitone
        assert pitch_bin(note, 12) == note // 12     # one bin per octave
        assert pitch_bin(note, 3) == note // 3
        assert pitch_bin(note, MAX_BIN_SIZE) == 0    # a single bin
    # bin_size is not hard-coded: a different width moves the boundary
    assert pitch_bin(63, 6) == 10 and pitch_bin(63, 12) == 5


def test_pitch_bin_quantises_fractional_detuning_deterministically():
    assert pitch_bin(60.5, 6) == 10
    assert pitch_bin(65.99, 6) == 10        # still a cent below the edge
    assert pitch_bin(66.0, 6) == 11
    assert pitch_bin(66.4, 6) == 11
    # the same input always gives the same output, and nearby notes agree
    for _ in range(3):
        assert pitch_bin(65.99, 6) == 10
    assert len({pitch_bin(note, 6) for note in (60.0, 60.2, 60.5)}) == 1


def test_pitch_bin_returns_no_condition_for_invalid_notes():
    """Silence, rests and garbage must not invent a pitch condition."""
    for invalid in (None, float("nan"), float("inf"), float("-inf"),
                    -1, -0.5, MIDI_NOTE_MAX + 1, 128, 1000, "60", "", []):
        assert pitch_bin(invalid, 6) is None, invalid


def test_validate_bin_size_and_bounds():
    assert validate_bin_size(6) == 6 and validate_bin_size(6.0) == 6
    assert validate_bin_size(1) == 1 and validate_bin_size(MAX_BIN_SIZE) == 128
    for bad in (0, -6, 2.5, MAX_BIN_SIZE + 1, "six", None, float("nan")):
        with pytest.raises(ValueError):
            validate_bin_size(bad)

    assert bin_note_bounds(0, 6) == (0, 5)
    assert bin_note_bounds(10, 6) == (60, 65)
    assert bin_note_bounds(21, 6) == (126, 127)      # clipped to MIDI 127
    # the bins partition the MIDI range: every note falls in its own bin's span
    for bin_size in (1, 3, 6, 12, 128):
        for note in range(int(MIDI_NOTE_MIN), int(MIDI_NOTE_MAX) + 1):
            index = pitch_bin(note, bin_size)
            low, high = bin_note_bounds(index, bin_size)
            assert low <= note <= high


def test_effective_note_matches_the_rendered_pitch():
    assert effective_note(60) == 60.0
    assert effective_note(60, transpose=5) == 65.0
    assert effective_note(60, transpose=-60) == MIDI_NOTE_MIN    # clipped
    assert effective_note(120, transpose=20) == MIDI_NOTE_MAX    # clipped
    assert effective_note(None, transpose=5) is None             # still no note


def test_segment_pitch_condition_policy():
    """The single rule both training and synthesis use."""
    phonemes = pitch_phoneme_set()
    assert segment_pitch_bin("a", 60, phonemes, 6) == 10
    assert segment_pitch_bin("A", 60, phonemes, 6) == 10        # alias
    # silence is never conditioned, even when a label gives it a note
    assert segment_pitch_bin("sil", 60, phonemes, 6) is None
    assert segment_pitch_bin("pau", 60, phonemes, 6) is None
    assert is_silence(phonemes, "sil") and is_silence(phonemes, "pau")
    assert not is_silence(phonemes, "a")
    # an unnoted segment has no condition: default_note is not substituted
    assert segment_pitch_bin("a", None, phonemes, 6) is None
    # a segment with an unusable note has no condition either
    assert segment_pitch_bin("a", float("nan"), phonemes, 6) is None
    assert segment_pitch_bin("a", -3, phonemes, 6) is None


def test_pitch_tier_names_are_distinguishable():
    assert pitch_tier(KIND_PHONE) == "phone+pitch"
    assert pitch_tier(KIND_TRIPHONE) == "triphone+pitch"
    assert PITCH_TIER_SUFFIX == "+pitch"
    assert is_pitch_tier("phone+pitch") and is_pitch_tier("left+pitch")
    assert not is_pitch_tier("phone") and not is_pitch_tier("class")
    assert not is_pitch_tier(None)


# --------------------------------------------------------------------------
# grouping observations (training side)
# --------------------------------------------------------------------------


def demo_spans():
    """One utterance: an unvoiced fricative, then the same vowel on two notes."""
    return [make_utterance(
        "u1",
        [("sil", 0, 4), ("s", 4, 8), ("a", 8, 20), ("a", 20, 32),
         ("sil", 32, 36)],
        notes=[None, None, LOW_NOTE, HIGH_NOTE, None])]


def test_collect_pitch_data_separates_one_phone_by_note():
    trainer = pitch_trainer()
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    occurrences = trainer.collect_pitch_data(demo_spans(), offset, scale)

    assert set(occurrences) == {("a", LOW_BIN), ("a", HIGH_BIN)}
    low = occurrences[("a", LOW_BIN)]
    assert low["kind"] == KIND_PHONE and low["unit"] == "a"
    assert low["curr"] == "a" and low["pitch_bin"] == LOW_BIN
    assert len(low["features"]) == 1 and len(low["features"][0]) == 12
    assert len(low["voiced"][0]) == 12
    high = occurrences[("a", HIGH_BIN)]
    assert len(high["features"][0]) == 12
    # the two buckets hold different frames of the same phone
    assert low["features"] is not high["features"]


def test_collect_pitch_data_ignores_silence_and_unnoted_segments():
    trainer = pitch_trainer()
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    spans = [make_utterance(
        "u1", [("sil", 0, 10), ("s", 10, 20), ("a", 20, 30), ("sil", 30, 40)],
        notes=[60.0, None, None, 72.0])]
    occurrences = trainer.collect_pitch_data(spans, offset, scale)
    # scored silence, unvoiced segments and unnoted vowels carry no condition
    assert occurrences == {}


def test_unvoiced_frames_inside_a_noted_segment_keep_the_note_condition():
    """The condition is the musical note, not the instantaneous F0."""
    trainer = pitch_trainer()
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    voiced = make_utterance("u1", [("a", 0, 20)], notes=[LOW_NOTE], voiced=True)
    unvoiced = make_utterance("u2", [("a", 0, 20)], notes=[LOW_NOTE],
                              voiced=False)
    occurrences = trainer.collect_pitch_data([voiced, unvoiced], offset, scale)
    assert set(occurrences) == {("a", LOW_BIN)}
    assert len(occurrences[("a", LOW_BIN)]["features"]) == 2
    # voicing travels with the frames: one all-voiced and one all-unvoiced
    # occurrence of the same note share one condition
    flags = occurrences[("a", LOW_BIN)]["voiced"]
    assert flags[0].all() and not flags[1].any()
    assert flags[0].dtype == bool and flags[1].dtype == bool


def test_collect_pitch_data_uses_context_units_that_exist_only():
    trainer = pitch_trainer()
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    wildcard = context_wildcard(trainer.phoneme_set.phonemes)
    left_key = left_diphone_key("s", "a", wildcard)
    triphone = "s^a^a"
    # only the left diphone of 'a' earned a context model in this run
    occurrences = trainer.collect_pitch_data(demo_spans(), offset, scale,
                                             context_units=(left_key,))
    assert set(occurrences) == {("a", LOW_BIN), ("a", HIGH_BIN),
                                (left_key, LOW_BIN)}
    entry = occurrences[(left_key, LOW_BIN)]
    assert entry["kind"] == KIND_LEFT and entry["curr"] == "a"
    assert entry["pitch_bin"] == LOW_BIN
    # the exact triphone was observed in the corpus but has no context model,
    # so it is not conditioned either
    assert (triphone, LOW_BIN) not in occurrences


def test_collect_pitch_data_without_notes_contributes_nothing():
    """Older callers that carry no span notes must not invent conditions."""
    trainer = pitch_trainer()
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    data = make_utterance("u1", [("a", 0, 20)], notes=[LOW_NOTE])
    data.span_notes = None
    assert trainer.collect_pitch_data([data], offset, scale) == {}


def test_select_pitch_models_thresholds_and_determinism():
    trainer = pitch_trainer(min_phoneme_frames=10, context_min_frames=20,
                            context_min_occurrences=2)
    left_key = left_diphone_key(
        "s", "a", context_wildcard(trainer.phoneme_set.phonemes))

    def entry(kind, unit, frames, count):
        return {"kind": kind, "unit": unit, "curr": "a", "pitch_bin": 1,
                "features": [np.zeros((frames // count, 2))] * count,
                "voiced": [np.zeros(frames // count, dtype=bool)] * count}

    occurrences = {
        ("a", 10): entry(KIND_PHONE, "a", 40, 2),      # clears min_phoneme_frames
        ("a", 11): entry(KIND_PHONE, "a", 8, 2),       # too thin: not a model
        ("a", 12): entry(KIND_PHONE, "a", 20, 1),      # clears, fewer frames
        (left_key, 10): entry(KIND_LEFT, left_key, 30, 3),
        (left_key, 11): entry(KIND_LEFT, left_key, 30, 1),  # too few occurrences
    }
    selected = trainer.select_pitch_models(occurrences)
    # frames descending, occurrences descending, unit ascending, bin ascending
    assert selected == [("a", 10), (left_key, 10), ("a", 12)]
    assert trainer.select_pitch_models(occurrences) == \
        trainer.select_pitch_models(dict(reversed(list(occurrences.items()))))

    # a phone bucket needs the phone threshold; a context bucket the context one
    trainer.config.min_phoneme_frames = 100
    assert trainer.select_pitch_models(occurrences) == [(left_key, 10)]
    trainer.config.min_phoneme_frames = 1
    trainer.config.context_min_frames = 1000
    assert trainer.select_pitch_models(occurrences) == [("a", 10), ("a", 12),
                                                        ("a", 11)]


def test_train_pitch_models_reuses_the_phone_budget_and_records_support():
    trainer = pitch_trainer(min_phoneme_frames=10)
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    occurrences = trainer.collect_pitch_data(demo_spans(), offset, scale)
    selected = trainer.select_pitch_models(occurrences)
    models, index = trainer.train_pitch_models(occurrences, selected,
                                               seed_base=7)

    assert set(models) == {("a", LOW_BIN), ("a", HIGH_BIN)}
    for key, hmm in models.items():
        assert hmm.is_trained()
        # 'a' is a vowel: three states in the test inventory, like the phone tier
        assert hmm.n_states == 3
        assert index[key]["kind"] == KIND_PHONE
        assert index[key]["unit"] == key[0] and index[key]["pitch_bin"] == key[1]
        assert index[key]["curr"] == "a"
        assert index[key]["frames"] == 12
        assert index[key]["occurrences"] == 1
        assert index[key]["n_free_params"] == hmm.n_free_params
    # two different buckets are two different models, not one shared object
    assert models[("a", LOW_BIN)] is not models[("a", HIGH_BIN)]


# --------------------------------------------------------------------------
# resolution hierarchy (model side)
# --------------------------------------------------------------------------


def test_resolve_unit_prefers_the_pitch_conditioned_model():
    model = make_pitch_model()
    key, hmm, tier = model.resolve_unit("sil", "a", "sil", pitch_bin=LOW_BIN)
    assert key == ("a", LOW_BIN) and hmm is model.pitch_models[("a", LOW_BIN)]
    assert tier == "phone+pitch" and is_pitch_tier(tier)

    key, hmm, tier = model.resolve_unit("sil", "a", "sil", pitch_bin=HIGH_BIN)
    assert key == ("a", HIGH_BIN) and hmm is model.pitch_models[("a", HIGH_BIN)]
    assert tier == "phone+pitch"


def test_resolve_unit_falls_back_when_the_bin_was_never_trained():
    """A missing bin must never borrow another bin's model."""
    model = make_pitch_model(bins=(LOW_BIN,))
    key, hmm, tier = model.resolve_unit("sil", "a", "sil", pitch_bin=HIGH_BIN)
    assert key is None and tier == "phone"
    assert hmm is model.hmms["a"]
    assert hmm is not model.pitch_models[("a", LOW_BIN)]

    # ... and without a dedicated phone model it keeps walking down
    model_no_phone = make_pitch_model(bins=(LOW_BIN,), with_phone_model=False)
    key, hmm, tier = model_no_phone.resolve_unit("sil", "a", "sil",
                                                 pitch_bin=HIGH_BIN)
    assert (key, tier) == (None, "class") and hmm is model_no_phone.backoff["vowel"]


def test_resolve_unit_without_a_condition_is_unchanged():
    model = make_pitch_model()
    for probe in [("sil", "a", "sil"), ("s", "i", "sil"), ("sil", "unknown", "sil")]:
        conditioned = model.resolve_unit(*probe)
        plain = model.resolve_unit(*probe, pitch_bin=None)
        assert conditioned == plain
        assert not is_pitch_tier(plain[2])
    # no pitch bin at all -> the ordinary hierarchy, exactly as before
    key, hmm, tier = model.resolve_unit("sil", "a", "sil")
    assert (key, tier) == (None, "phone") and hmm is model.hmms["a"]


def test_resolve_unit_full_ladder_with_contexts_and_pitch():
    """exact context+pitch -> partial context+pitch -> phone+pitch -> context
    -> phone -> class backoff -> global backoff."""
    model = make_pitch_model(contexts=True)
    wildcard = context_wildcard(model.phoneme_set.phonemes)
    left_key = left_diphone_key("sil", "a", wildcard)

    # 1. only the phone is conditioned at this bin, so it outranks every
    #    unconditioned tier
    key, hmm, tier = model.resolve_unit("sil", "a", "sil", pitch_bin=LOW_BIN)
    assert (key, tier) == (("a", LOW_BIN), "phone+pitch")
    assert hmm is model.pitch_models[("a", LOW_BIN)]

    # 2. conditioning the exact context outranks the conditioned phone
    condition_unit(model, "sil^a^sil", KIND_TRIPHONE, LOW_BIN)
    key, hmm, tier = model.resolve_unit("sil", "a", "sil", pitch_bin=LOW_BIN)
    assert (key, tier) == (("sil^a^sil", LOW_BIN), "triphone+pitch")

    # 3. a neighbourhood whose triphone is not conditioned falls to the
    #    conditioned one-sided diphone, and only then to the conditioned phone
    condition_unit(model, left_key, KIND_LEFT, LOW_BIN, frames=80)
    key, hmm, tier = model.resolve_unit("sil", "a", "i", pitch_bin=LOW_BIN)
    assert (key, tier) == ((left_key, LOW_BIN), "left+pitch")
    assert hmm is model.pitch_models[(left_key, LOW_BIN)]

    # 4. nothing conditioned at this bin: the unconditioned context tier
    key, hmm, tier = model.resolve_unit("sil", "a", "sil", pitch_bin=MIDDLE_BIN)
    assert (key, tier) == ("sil^a^sil", "triphone")
    assert hmm is model.contexts["sil^a^sil"]

    # 5. ... then the dedicated phone, the class backoff, the global backoff
    key, hmm, tier = model.resolve_unit("s", "i", "sil", pitch_bin=MIDDLE_BIN)
    assert (key, tier) == (None, "phone") and hmm is model.hmms["i"]
    key, hmm, tier = model.resolve_unit("sil", "rare", "sil",
                                        pitch_bin=MIDDLE_BIN)
    assert (key, tier) == (None, "class") and hmm is model.backoff["vowel"]
    key, hmm, tier = model.resolve_unit("sil", "zz", "sil", pitch_bin=MIDDLE_BIN)
    assert (key, tier) == (None, "global") and hmm is model.global_backoff


def test_pitch_diphone_ties_favour_the_left_context():
    model = make_pitch_model(contexts=True)
    wildcard = context_wildcard(model.phoneme_set.phonemes)
    left_key = left_diphone_key("s", "a", wildcard)
    right_key = right_diphone_key("a", "i", wildcard)
    assert left_key != right_key
    for unit, kind in ((left_key, KIND_LEFT), (right_key, KIND_RIGHT)):
        condition_unit(model, unit, kind, LOW_BIN, frames=50)

    key, hmm, tier = model.resolve_unit("s", "a", "i", pitch_bin=LOW_BIN)
    assert (key, tier) == ((left_key, LOW_BIN), "left+pitch")

    # and the better-supported side wins outright when they differ
    condition_unit(model, right_key, KIND_RIGHT, LOW_BIN, frames=400)
    key, hmm, tier = model.resolve_unit("s", "a", "i", pitch_bin=LOW_BIN)
    assert (key, tier) == ((right_key, LOW_BIN), "right+pitch")


def test_model_without_pitch_conditioning_ignores_a_pitch_bin():
    model = make_pitch_model(pitch=False)
    assert model.pitch_models == {}
    assert model.pitch_conditioning.enabled is False
    assert model.segment_pitch_bin("a", LOW_NOTE) is None
    key, hmm, tier = model.resolve_unit("sil", "a", "sil", pitch_bin=LOW_BIN)
    assert (key, tier) == (None, "phone") and hmm is model.hmms["a"]


def test_segment_pitch_bin_uses_the_models_own_bin_size():
    model = make_pitch_model(bins=(pitch_bin(60, 3),))
    model.pitch_conditioning = PitchConditioning(enabled=True, bin_size=3)
    assert model.segment_pitch_bin("a", 60) == 20
    assert model.segment_pitch_bin("a", 62) == 20
    assert model.segment_pitch_bin("a", 63) == 21
    assert model.segment_pitch_bin("sil", 60) is None
    assert model.segment_pitch_bin("a", None) is None
    assert model.pitch_support("a", 20) == 120
    assert model.pitch_support("a", 21) is None


def test_pitch_conditioning_dataclass_roundtrip_and_validation():
    conditioning = PitchConditioning(enabled=True, bin_size=12)
    document = conditioning.to_dict()
    assert document == {"enabled": True, "bin_size": 12,
                        "bin_unit": "semitones", "n_bins": 11}
    assert PitchConditioning.from_dict(document) == conditioning
    assert PitchConditioning.from_dict({}) == PitchConditioning()
    assert PitchConditioning().active is False
    assert PitchConditioning(enabled=True).bin_of(60) == 10
    assert PitchConditioning(enabled=True).bin_of(None) is None
    assert "disabled" in PitchConditioning().describe()
    assert "12 semitone" in conditioning.describe()
    with pytest.raises(ValueError, match="bin_size"):
        PitchConditioning(bin_size=0)
    with pytest.raises(ValueError, match="unknown pitch conditioning"):
        PitchConditioning.from_dict({"bogus": 1})


# --------------------------------------------------------------------------
# training end to end
# --------------------------------------------------------------------------


def test_training_creates_one_model_per_pitch_region(corpus):
    messages = []
    model = Trainer(training_config(corpus), pitch_phoneme_set(),
                    log=messages.append).train()
    assert model.pitch_conditioning.enabled is True
    assert model.pitch_conditioning.bin_size == DEFAULT_BIN_SIZE
    assert sorted(model.pitch_models) == [("a", LOW_BIN), ("a", HIGH_BIN)]
    assert sorted(model.pitch_index) == sorted(model.pitch_models)
    # the phoneme inventory is untouched: no synthetic "a@48" phonemes anywhere
    assert sorted(model.hmms) == ["a", "sil"]
    assert all("^" not in unit and "@" not in unit
               for unit, _bin in model.pitch_models)
    assert model.duration_model.stats.keys() <= {"a", "sil"}
    assert set(model.pitch_model.stats) <= {"a", "sil"}

    joined = "\n".join(messages)
    assert "pitch conditioning" in joined
    assert f"bin {LOW_BIN}" in joined and f"bin {HIGH_BIN}" in joined

    # the two bins really are different acoustic models, and both differ from
    # the pooled phone model that averages across the two pitch regions
    low = model.pitch_models[("a", LOW_BIN)]
    high = model.pitch_models[("a", HIGH_BIN)]
    pooled = model.hmms["a"]
    assert not np.allclose(low.states[0].gmm.means, high.states[0].gmm.means)
    assert not np.allclose(low.states[0].gmm.means, pooled.states[0].gmm.means)
    assert model.pitch_n_free_params == low.n_free_params + high.n_free_params
    assert model.n_free_params == (model.phoneme_n_free_params
                                   + model.context_n_free_params
                                   + model.backoff_n_free_params
                                   + model.global_backoff_n_free_params
                                   + model.pitch_n_free_params)


def test_training_records_the_bin_definition_in_the_model(corpus):
    model = Trainer(training_config(corpus), pitch_phoneme_set()).train()
    for (unit, index), info in model.pitch_index.items():
        low, high = bin_note_bounds(index, DEFAULT_BIN_SIZE)
        assert info["unit"] == unit and info["pitch_bin"] == index
        assert info["kind"] == KIND_PHONE and info["curr"] == "a"
        assert info["frames"] > 0 and info["occurrences"] > 0
        assert (info["note_min"], info["note_max"]) == (low, high)
        assert low <= (LOW_NOTE if index == LOW_BIN else HIGH_NOTE) <= high


def test_disabled_training_creates_no_pitch_models(corpus, plain_trained):
    model = plain_trained
    assert model.pitch_conditioning.enabled is False
    assert model.pitch_models == {} and model.pitch_index == {}
    assert model.pitch_n_free_params == 0
    assert model.hmms, "the ordinary tiers are still trained"
    # the unconditioned model still averages the two pitch regions
    assert model.n_free_params == (model.phoneme_n_free_params
                                   + model.backoff_n_free_params)


def test_thin_buckets_are_not_forced_into_models(corpus):
    """A bucket below its tier's threshold stays unmodelled (and says so)."""
    messages = []
    config = training_config(corpus, min_phoneme_frames=100_000)
    model = Trainer(config, pitch_phoneme_set(), log=messages.append).train()
    assert model.pitch_models == {}
    assert model.pitch_conditioning.enabled is True
    assert model.hmms == {}              # the phone tier falls short too
    assert model.backoff, "the class backoff still covers everything"
    joined = "\n".join(messages)
    assert "0 pitch-conditioned models" in joined
    assert "phone/context/backoff hierarchy" in joined

    # ... and such a model still synthesises, through the ordinary hierarchy
    result = Synthesizer(model, SynthesisConfig(seed=0, vocoder="builtin")
                         ).synthesize(note_score(LOW_NOTE))
    assert np.isfinite(result.audio).all() and len(result.audio) > 0
    diagnostics = "\n".join(result.diagnostics)
    assert "carries no pitch-conditioned models" in diagnostics
    assert "unconditioned model hierarchy" in diagnostics


def test_pitch_conditioning_does_not_disturb_the_other_tiers(corpus):
    """Same corpus, same seed: the unconditioned tiers are identical."""
    conditioned = Trainer(training_config(corpus), pitch_phoneme_set()).train()
    plain = Trainer(training_config(corpus, pitch_conditioning_enabled=False),
                    pitch_phoneme_set()).train()
    assert sorted(conditioned.hmms) == sorted(plain.hmms)
    assert sorted(conditioned.backoff) == sorted(plain.backoff)
    for phone, hmm in plain.hmms.items():
        other = conditioned.hmms[phone]
        assert other.n_states == hmm.n_states
        assert np.allclose(other.self_loops, hmm.self_loops)
        for state, other_state in zip(hmm.states, other.states):
            assert np.allclose(state.gmm.means, other_state.gmm.means)
            assert np.allclose(state.gmm.variances, other_state.gmm.variances)
    assert np.allclose(conditioned.feature_offset, plain.feature_offset)
    assert np.allclose(conditioned.feature_scale, plain.feature_scale)
    assert conditioned.duration_model.to_dict() == plain.duration_model.to_dict()
    assert conditioned.pitch_model.to_dict() == plain.pitch_model.to_dict()


def test_normalisation_stays_corpus_wide_with_pitch_conditioning(corpus,
                                                                 tmp_path):
    """One shared feature space: no per-bin normalisation statistics."""
    model = Trainer(training_config(corpus), pitch_phoneme_set()).train()
    directory = tmp_path / "model"
    model.save(directory)
    import yaml
    document = yaml.safe_load((directory / "model.yaml").read_text())
    assert set(document["normalization"]) == {"offset", "scale", "units"}
    assert len(document["normalization"]["offset"]) == model.static_dim
    assert len(document["normalization"]["scale"]) == model.static_dim
    # every bin is expressed in that one space, so the models stay comparable
    for hmm in model.pitch_models.values():
        assert hmm.dim == model.feature_dim


def test_repeated_training_is_deterministic(corpus):
    first = Trainer(training_config(corpus), pitch_phoneme_set()).train()
    second = Trainer(training_config(corpus), pitch_phoneme_set()).train()
    assert sorted(first.pitch_models) == sorted(second.pitch_models)
    assert list(first.pitch_models) == list(second.pitch_models)
    assert first.pitch_index == second.pitch_index
    for key, hmm in first.pitch_models.items():
        other = second.pitch_models[key]
        assert other.n_states == hmm.n_states
        for state, other_state in zip(hmm.states, other.states):
            assert np.allclose(state.gmm.means, other_state.gmm.means)
            assert np.allclose(state.gmm.variances, other_state.gmm.variances)


def test_a_different_bin_size_changes_the_inventory(corpus):
    model = Trainer(training_config(corpus, pitch_conditioning_bin_size=12),
                    pitch_phoneme_set()).train()
    assert model.pitch_conditioning.bin_size == 12
    assert sorted(model.pitch_models) == [("a", pitch_bin(LOW_NOTE, 12)),
                                          ("a", pitch_bin(HIGH_NOTE, 12))]
    assert pitch_bin(LOW_NOTE, 12) == 4 and pitch_bin(HIGH_NOTE, 12) == 6


# --------------------------------------------------------------------------
# contexts and pitch conditioning together
# --------------------------------------------------------------------------


def test_context_and_pitch_conditioning_coexist(corpus):
    config = training_config(corpus, context_enabled=True,
                             context_min_frames=10, context_min_occurrences=1,
                             context_max_models=8, context_global_backoff=True)
    messages = []
    model = Trainer(config, pitch_phoneme_set(), log=messages.append).train()
    assert model.contexts, "sanity: this corpus does train contexts"
    assert model.pitch_models, "sanity: and pitch-conditioned models"

    # context buckets are conditioned only for contexts that earned a model
    conditioned_units = {unit for unit, _bin in model.pitch_models}
    assert conditioned_units <= set(model.contexts) | set(model.hmms)
    assert any("^" in unit for unit in conditioned_units), \
        "at least one trained context is also pitch-conditioned"

    # resolution prefers the conditioned context over the conditioned phone
    wildcard = context_wildcard(model.phoneme_set.phonemes)
    probe = None
    for (unit, index) in sorted(model.pitch_models):
        if "^" in unit:
            pre, curr, post = unit.split("^")
            # the unmodelled side of a diphone key carries the wildcard; ask
            # with a phone nothing trained there, so the same key is reached
            probe = (unit, index,
                     "zz" if pre == wildcard else pre, curr,
                     "zz" if post == wildcard else post)
            break
    assert probe is not None
    unit, index, pre, curr, post = probe
    key, hmm, tier = model.resolve_unit(pre, curr, post, pitch_bin=index)
    assert key == (unit, index) and is_pitch_tier(tier)
    assert hmm is model.pitch_models[(unit, index)]

    # a synthesis render with both tiers present still works
    result = Synthesizer(model, SynthesisConfig(seed=0, vocoder="builtin")
                         ).synthesize(note_score(LOW_NOTE))
    assert np.isfinite(result.audio).all() and len(result.audio) > 0


# --------------------------------------------------------------------------
# serialisation (format 4 + older formats)
# --------------------------------------------------------------------------


def test_pitch_model_save_load_roundtrip(pitch_trained, tmp_path):
    model = pitch_trained
    directory = tmp_path / "model"
    model.save(directory)
    assert (directory / "pitch.npz").exists()
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    assert f"format_version: {MODEL_FORMAT_VERSION}" in text
    # top-level sections (the metadata dump also names the training fields)
    assert "\npitch_conditioning:" in text and "\npitch_index:" in text
    assert "bin_size: 6" in text and "bin_unit: semitones" in text
    assert "note_min" in text and "note_max" in text
    # no synthetic phoneme names in the inventory section
    assert "a@48" not in text and "a_48" not in text

    loaded = HMSModel.load(directory)
    assert loaded.loaded_format_version == MODEL_FORMAT_VERSION
    assert loaded.pitch_conditioning == model.pitch_conditioning
    assert loaded.pitch_conditioning.enabled is True
    assert loaded.pitch_conditioning.bin_size == DEFAULT_BIN_SIZE
    assert sorted(loaded.pitch_models) == sorted(model.pitch_models)
    assert loaded.pitch_index == model.pitch_index
    assert loaded.pitch_n_free_params == model.pitch_n_free_params
    assert loaded.n_free_params == model.n_free_params
    for key, hmm in model.pitch_models.items():
        other = loaded.pitch_models[key]
        assert other.n_states == hmm.n_states
        for state, other_state in zip(hmm.states, other.states):
            assert np.allclose(state.gmm.means, other_state.gmm.means)
            assert np.allclose(state.gmm.variances, other_state.gmm.variances)
        assert np.allclose(other.self_loops, hmm.self_loops)
    # selection behaviour survives the round trip, bin by bin
    for probe in (LOW_NOTE, HIGH_NOTE, MIDDLE_NOTE, None):
        expected = model.resolve_unit("sil", "a", "sil",
                                      pitch_bin=model.segment_pitch_bin("a", probe))
        actual = loaded.resolve_unit(
            "sil", "a", "sil",
            pitch_bin=loaded.segment_pitch_bin("a", probe))
        assert expected[0] == actual[0] and expected[2] == actual[2]


def test_model_yaml_is_the_only_source_of_the_bin_definition(pitch_trained,
                                                             tmp_path):
    """Synthesis must not need the training configuration to interpret bins."""
    directory = tmp_path / "model"
    pitch_trained.save(directory)
    loaded = HMSModel.load(directory)
    # the loaded model reproduces the training-time binning on its own
    for note in (LOW_NOTE, HIGH_NOTE, MIDDLE_NOTE, 0, 127, 65.5):
        assert loaded.segment_pitch_bin("a", note) == \
            segment_pitch_bin("a", note, loaded.phoneme_set,
                              DEFAULT_BIN_SIZE)
    assert loaded.pitch_conditioning.n_bins() == n_pitch_bins(DEFAULT_BIN_SIZE)


def test_missing_pitch_npz_is_a_clean_error(pitch_trained, tmp_path):
    directory = tmp_path / "model"
    pitch_trained.save(directory)
    (directory / "pitch.npz").unlink()
    with pytest.raises(ValueError, match="pitch.npz"):
        HMSModel.load(directory)


def test_unconditioned_model_writes_no_pitch_payload(plain_trained, tmp_path):
    directory = tmp_path / "model"
    plain_trained.save(directory)
    assert not (directory / "pitch.npz").exists()
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    assert "\npitch_conditioning:" not in text
    assert "\npitch_index:" not in text


@pytest.mark.parametrize("legacy_version", [2, 3])
def test_older_model_formats_load_with_conditioning_disabled(
        plain_trained, tmp_path, legacy_version):
    """A model trained before this feature needs no migration to load."""
    directory = tmp_path / "model"
    plain_trained.save(directory)
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    (directory / "model.yaml").write_text(
        text.replace(f"format_version: {MODEL_FORMAT_VERSION}",
                     f"format_version: {legacy_version}"), encoding="utf-8")

    loaded = HMSModel.load(directory)
    assert loaded.loaded_format_version == legacy_version
    assert loaded.pitch_conditioning.enabled is False
    assert loaded.pitch_models == {} and loaded.pitch_index == {}
    assert loaded.pitch_n_free_params == 0
    assert loaded.n_free_params == plain_trained.n_free_params
    assert loaded.segment_pitch_bin("a", LOW_NOTE) is None
    key, hmm, tier = loaded.resolve_unit("sil", "a", "sil", pitch_bin=LOW_BIN)
    assert (key, tier) == (None, "phone") and hmm is loaded.get_hmm("a")
    # and it still sings
    result = Synthesizer(loaded, SynthesisConfig(seed=0, vocoder="builtin")
                         ).synthesize(note_score(LOW_NOTE))
    assert np.isfinite(result.audio).all()


def test_future_format_is_still_refused(pitch_trained, tmp_path):
    directory = tmp_path / "model"
    pitch_trained.save(directory)
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    (directory / "model.yaml").write_text(
        text.replace(f"format_version: {MODEL_FORMAT_VERSION}",
                     f"format_version: {MODEL_FORMAT_VERSION + 1}"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="not supported"):
        HMSModel.load(directory)


def test_inspection_reports_the_pitch_tier(pitch_trained, plain_trained):
    report = "\n".join(pitch_trained.parameter_report())
    assert "pitch models" in report and "pitch HMM params" in report
    assert f"bins of {DEFAULT_BIN_SIZE} semitones" in report
    summary = pitch_trained.summary()
    assert "pitch-conditioned HMMs" in summary
    assert "pitch conditioning" in summary
    assert f"bin {LOW_BIN}" in summary and f"bin {HIGH_BIN}" in summary
    low, high = bin_note_bounds(LOW_BIN, DEFAULT_BIN_SIZE)
    assert f"MIDI {low:3d}-{high:3d}" in summary

    plain = "\n".join(plain_trained.parameter_report())
    assert "pitch HMM params  : 0" in plain
    assert "pitch models      : 0" in plain
    assert "pitch-conditioned HMMs" not in plain_trained.summary()


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------


def test_synthesis_selects_the_model_of_the_requested_note(pitch_trained):
    synthesizer = Synthesizer(pitch_trained,
                              SynthesisConfig(seed=0, vocoder="builtin"))
    planned = synthesizer.plan(note_score(LOW_NOTE))
    assert len(planned) == 6              # the plan() API is unchanged
    low_units = set(id(unit) for unit in synthesizer._frame_units)
    assert id(pitch_trained.pitch_models[("a", LOW_BIN)]) in low_units
    assert id(pitch_trained.pitch_models[("a", HIGH_BIN)]) not in low_units

    synthesizer.plan(note_score(HIGH_NOTE))
    high_units = set(id(unit) for unit in synthesizer._frame_units)
    assert id(pitch_trained.pitch_models[("a", HIGH_BIN)]) in high_units
    assert low_units != high_units

    # the same phone, two notes, two different acoustic models -- and both
    # render
    for note in (LOW_NOTE, HIGH_NOTE):
        result = Synthesizer(pitch_trained,
                             SynthesisConfig(seed=0, vocoder="builtin")
                             ).synthesize(note_score(note))
        assert len(result.audio) > 0
        assert np.isfinite(result.audio).all()
        assert result.duration == pytest.approx(0.7, abs=0.05)


def test_synthesis_falls_back_for_an_untrained_bin(pitch_trained):
    synthesizer = Synthesizer(pitch_trained,
                              SynthesisConfig(seed=0, vocoder="builtin"))
    result = synthesizer.synthesize(note_score(MIDDLE_NOTE))
    assert np.isfinite(result.audio).all() and len(result.audio) > 0

    pooled = pitch_trained.hmms["a"]
    units = set(id(unit) for unit in synthesizer._frame_units)
    assert id(pooled) in units
    assert all(id(hmm) not in units for hmm in pitch_trained.pitch_models.values())

    diagnostics = "\n".join(result.diagnostics)
    assert "pitch conditioning" in diagnostics
    assert f"bin {MIDDLE_BIN}" in diagnostics
    assert "fell back to the unconditioned hierarchy" in diagnostics


def test_synthesis_reports_conditioned_frames(pitch_trained):
    result = Synthesizer(pitch_trained, SynthesisConfig(seed=0,
                                                        vocoder="builtin")
                         ).synthesize(note_score(LOW_NOTE))
    diagnostics = "\n".join(result.diagnostics)
    assert "pitch-conditioned model" in diagnostics
    assert "fell back" not in diagnostics


def test_transpose_moves_the_pitch_condition(pitch_trained):
    """The condition follows the note that will actually be rendered."""
    synthesizer = Synthesizer(
        pitch_trained,
        SynthesisConfig(seed=0, vocoder="builtin",
                        transpose=float(HIGH_NOTE - LOW_NOTE)))
    synthesizer.plan(note_score(LOW_NOTE))       # 48 + 24 -> bin of 72
    units = set(id(unit) for unit in synthesizer._frame_units)
    assert id(pitch_trained.pitch_models[("a", HIGH_BIN)]) in units


def test_silence_frames_are_never_pitch_conditioned(pitch_trained):
    synthesizer = Synthesizer(pitch_trained,
                              SynthesisConfig(seed=0, vocoder="builtin"))
    phones = synthesizer.plan(note_score(LOW_NOTE))[0]
    units = synthesizer._frame_units
    conditioned = {id(hmm) for hmm in pitch_trained.pitch_models.values()}
    assert "sil" in phones and conditioned
    for phone, unit in zip(phones, units):
        if phone == "sil":
            assert id(unit) not in conditioned
        else:
            assert id(unit) in conditioned


def test_pitch_conditioning_does_not_change_the_generated_f0(pitch_trained,
                                                             plain_trained):
    """F0 stays score-driven: conditioning is model selection, not a predictor."""
    for model in (pitch_trained, plain_trained):
        result = Synthesizer(model, SynthesisConfig(seed=0, vocoder="builtin")
                             ).synthesize(note_score(LOW_NOTE))
        voiced = np.isfinite(result.f0_semitones)
        assert voiced.any()
        expected = 12.0 * np.log2(
            labels_module.midi_to_hz(LOW_NOTE) / model.spec.f0_ref_hz)
        # the steady middle of the note is the requested pitch, to the cent
        middle = result.f0_semitones[voiced][len(result.f0_semitones[voiced]) // 2]
        assert middle == pytest.approx(expected, abs=0.05)
    # and an external F0 trajectory is still authoritative, condition or not
    frames = len(Synthesizer(pitch_trained, SynthesisConfig(seed=0,
                                                            vocoder="builtin")
                             ).synthesize(note_score(LOW_NOTE)).params.f0)
    trajectory = np.full(frames, 300.0)
    overridden = Synthesizer(pitch_trained, SynthesisConfig(seed=0,
                                                            vocoder="builtin")
                             ).synthesize(note_score(LOW_NOTE), f0=trajectory)
    assert np.allclose(overridden.params.f0[np.isfinite(overridden.f0_semitones)],
                       300.0, atol=1e-6)


def test_unconditioned_synthesis_is_unchanged(plain_trained):
    """A model trained with the feature off takes the classic path."""
    synthesizer = Synthesizer(plain_trained,
                              SynthesisConfig(seed=0, vocoder="builtin"))
    planned = synthesizer.plan(note_score(LOW_NOTE))
    assert len(planned) == 6
    assert synthesizer._frame_units is None       # no per-frame unit resolution
    assert "pitch conditioning" not in "\n".join(planned[5])
    result = synthesizer.synthesize(note_score(LOW_NOTE))
    assert np.isfinite(result.audio).all() and len(result.audio) > 0
    assert not any("pitch conditioning" in d for d in result.diagnostics)


def test_existing_demo_model_and_score_are_unaffected(trained_model,
                                                      short_score):
    """The session fixture's model (trained before/without the feature) works."""
    assert trained_model.pitch_conditioning.enabled is False
    assert trained_model.pitch_models == {}
    assert trained_model.segment_pitch_bin("a", LOW_NOTE) is None
    synthesizer = Synthesizer(trained_model,
                              SynthesisConfig(seed=0, vocoder="builtin"))
    planned = synthesizer.plan(short_score)
    assert len(planned) == 6
    assert synthesizer._frame_units is None
    result = synthesizer.synthesize(short_score)
    assert np.isfinite(result.audio).all()
    assert not any("pitch conditioning" in d for d in result.diagnostics)
    # and an out-of-range note still behaves as before: rendered, not moved
    key, hmm, tier = trained_model.resolve_unit("sil", "a", "sil",
                                                pitch_bin=LOW_BIN)
    assert (key, tier) == (None, "phone") and hmm is trained_model.get_hmm("a")


def test_pitch_conditioned_synthesis_is_deterministic(pitch_trained):
    first = Synthesizer(pitch_trained, SynthesisConfig(seed=0,
                                                       vocoder="builtin")
                        ).synthesize(note_score(LOW_NOTE))
    second = Synthesizer(pitch_trained, SynthesisConfig(seed=0,
                                                        vocoder="builtin")
                         ).synthesize(note_score(LOW_NOTE))
    assert np.array_equal(first.audio, second.audio)
    assert first.diagnostics == second.diagnostics


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def test_evaluate_reports_pitch_conditioned_frames(corpus, pitch_trained,
                                                   plain_trained):
    import copy

    conditioned = copy.deepcopy(pitch_trained)
    conditioned.name = "conditioned"
    baseline = copy.deepcopy(plain_trained)
    baseline.name = "baseline"
    report = evaluate_models([baseline, conditioned], corpus["labels"],
                             corpus["wav_dir"])
    assert set(report["models"]) == {"baseline", "conditioned"}
    baseline_metrics = report["models"]["baseline"]
    conditioned_metrics = report["models"]["conditioned"]
    assert baseline_metrics["pitch_conditioned_frames"] == 0
    assert baseline_metrics["pitch_fallback_frames"] == 0
    assert conditioned_metrics["pitch_conditioned_frames"] > 0
    assert conditioned_metrics["pitch_fallback_frames"] == 0
    assert conditioned_metrics["frames"] == baseline_metrics["frames"]
    for metrics in report["models"].values():
        assert np.isfinite(metrics["total_log_likelihood"])


def test_pitch_conditioning_settings_do_not_warn_in_evaluation(
        pitch_trained, plain_trained):
    import copy

    conditioned = copy.deepcopy(pitch_trained)
    conditioned.name = "conditioned"
    baseline = copy.deepcopy(plain_trained)
    baseline.name = "baseline"
    errors, warnings = check_model_compatibility([baseline, conditioned])
    assert errors == []
    assert not any("pitch_conditioning" in warning for warning in warnings)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_train_pitch_conditioning_flag_and_inspection(corpus, tmp_path,
                                                          capsys):
    parameters = tmp_path / "parameters.yaml"
    parameters.write_text("pitch_conditioning:\n  enabled: false\n"
                          "  bin_size: 6\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    assert main(["train", "--labels", corpus["labels"],
                 "--wav-dir", corpus["wav_dir"], "--out", str(model_dir),
                 "--fs", str(TEST_FS), "--iterations", "1",
                 "--config", str(parameters), "--vocoder", "builtin",
                 "--pitch-conditioning"]) == 0
    assert (model_dir / "pitch.npz").exists()
    text = (model_dir / "model.yaml").read_text(encoding="utf-8")
    assert f"format_version: {MODEL_FORMAT_VERSION}" in text
    # top-level sections (the metadata dump also names the training fields)
    assert "\npitch_conditioning:" in text and "\npitch_index:" in text

    capsys.readouterr()
    assert main(["inspect-model", "--model", str(model_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    conditioning = payload["pitch_conditioning"]
    assert conditioning["enabled"] is True
    assert conditioning["bin_size"] == 6
    assert conditioning["bin_unit"] == "semitones"
    assert conditioning["models"], "conditioned models must be inspected"
    for record in conditioning["models"]:
        assert record["unit"] == "a" and record["pitch_bin"] in (LOW_BIN,
                                                                 HIGH_BIN)
        assert record["note_min"] <= (LOW_NOTE if record["pitch_bin"] == LOW_BIN
                                      else HIGH_NOTE) <= record["note_max"]
    breakdown = payload["parameter_breakdown"]
    assert breakdown["pitch_conditioned"] > 0
    assert sum(breakdown.values()) == payload["parameter_budget"]

    capsys.readouterr()
    assert main(["inspect-model", "--model", str(model_dir)]) == 0
    assert "pitch-conditioned HMMs" in capsys.readouterr().out

    # a conditioned model synthesises through the CLI and says what it used
    score = tmp_path / "score.tsv"
    score.write_text(f"demo\t0.0\t0.1\tsil\t-\n"
                     f"demo\t0.1\t0.7\ta\t{LOW_NOTE}\n"
                     f"demo\t0.7\t0.8\tsil\t-\n", encoding="utf-8")
    capsys.readouterr()
    assert main(["synth", "--model", str(model_dir), "--score", str(score),
                 "--out", str(tmp_path / "out.wav"), "--seed", "0",
                 "--vocoder", "builtin"]) == 0
    output = capsys.readouterr().out
    assert "pitch conditioning" in output
    assert "pitch-conditioned model" in output


def test_cli_pitch_bin_size_flag(corpus, tmp_path):
    model_dir = tmp_path / "model"
    assert main(["train", "--labels", corpus["labels"],
                 "--wav-dir", corpus["wav_dir"], "--out", str(model_dir),
                 "--fs", str(TEST_FS), "--iterations", "1",
                 "--vocoder", "builtin", "--pitch-conditioning",
                 "--pitch-bin-size", "12"]) == 0
    loaded = HMSModel.load(model_dir)
    assert loaded.pitch_conditioning.bin_size == 12
    assert sorted(loaded.pitch_models) == [("a", pitch_bin(LOW_NOTE, 12)),
                                           ("a", pitch_bin(HIGH_NOTE, 12))]


def test_cli_no_pitch_conditioning_flag_overrides_enabled_config(corpus,
                                                                 tmp_path):
    parameters = tmp_path / "parameters.yaml"
    parameters.write_text("pitch_conditioning:\n  enabled: true\n"
                          "  bin_size: 6\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    assert main(["train", "--labels", corpus["labels"],
                 "--wav-dir", corpus["wav_dir"], "--out", str(model_dir),
                 "--fs", str(TEST_FS), "--iterations", "1",
                 "--vocoder", "builtin", "--no-pitch-conditioning"]) == 0
    assert not (model_dir / "pitch.npz").exists()
    text = (model_dir / "model.yaml").read_text(encoding="utf-8")
    assert "\npitch_conditioning:" not in text
    assert "\npitch_index:" not in text
    loaded = HMSModel.load(model_dir)
    assert loaded.pitch_models == {}
    assert loaded.pitch_conditioning.enabled is False


def test_cli_rejects_an_invalid_bin_size(corpus, tmp_path, capsys):
    model_dir = tmp_path / "model"
    assert main(["train", "--labels", corpus["labels"],
                 "--wav-dir", corpus["wav_dir"], "--out", str(model_dir),
                 "--fs", str(TEST_FS), "--vocoder", "builtin",
                 "--pitch-conditioning", "--pitch-bin-size", "0"]) == 2
    assert "bin_size" in capsys.readouterr().err


# --------------------------------------------------------------------------
# scored silence and sparse data (the "never invent a condition" rules)
# --------------------------------------------------------------------------


def test_scored_silence_is_not_pitch_conditioned(tmp_path):
    """A label that gives `sil` a note must still not create a sil/bin model."""
    directory = tmp_path / "corpus"
    corpus = two_note_corpus(directory, notes=(LOW_NOTE,),
                             with_scored_silence=True)
    config = training_config(corpus, min_phoneme_frames=1)
    model = Trainer(config, pitch_phoneme_set()).train()
    assert model.pitch_models, "sanity: the vowel is conditioned"
    assert all(unit != "sil" for unit, _bin in model.pitch_models)
    assert model.segment_pitch_bin("sil", LOW_NOTE) is None


def test_single_note_corpus_conditions_only_that_bin(tmp_path):
    directory = tmp_path / "corpus"
    corpus = two_note_corpus(directory, notes=(LOW_NOTE,))
    model = Trainer(training_config(corpus), pitch_phoneme_set()).train()
    assert sorted(model.pitch_models) == [("a", LOW_BIN)]
    # a score asking for a different note still renders, from the pooled model
    result = Synthesizer(model, SynthesisConfig(seed=0, vocoder="builtin")
                         ).synthesize(note_score(HIGH_NOTE))
    assert np.isfinite(result.audio).all() and len(result.audio) > 0
    assert "fell back to the unconditioned hierarchy" in \
        "\n".join(result.diagnostics)
