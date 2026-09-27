"""Optional sparse phoneme-context modelling.

Covers the context unit keys, the trainer's collection/selection/training of
sparse contexts, the model-side fallback hierarchy, format-3 serialisation
(with format-2 compatibility), the unchanged six-value `plan()` API, and the
`hms train --context` CLI surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hms.cli.main import main
from hms.config import load_parameters, training_config_from_parameters
from hms.core.context import (CONTEXT_SEPARATOR, GLOBAL_KEY, KIND_LEFT,
                              KIND_RIGHT, KIND_TRIPHONE, context_keys,
                              context_wildcard, left_diphone_key,
                              neighbor_contexts, right_diphone_key,
                              triphone_key)
from hms.core.features import FeatureSpec
from hms.core.hmm import LeftToRightHMM
from hms.core.model import MODEL_FORMAT_VERSION, HMSModel
from hms.core.phonemes import PhonemeSet
from hms.core.synthesizer import SynthesisConfig, Synthesizer
from hms.core.trainer import Trainer, TrainingConfig, UtteranceData


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def context_phoneme_set() -> PhonemeSet:
    return PhonemeSet.from_dict({
        "defaults": {
            "vowel": {"n_states": 2, "n_components": 1, "voiced": True},
            "unvoiced_consonant": {"n_states": 2, "n_components": 1},
        },
        "phonemes": {
            "sil": {"type": "silence", "n_states": 1, "n_components": 1},
            "a": {"type": "vowel"},
            "i": {"type": "vowel"},
            "rare": {"type": "vowel"},
            "s": {"type": "unvoiced_consonant"},
        },
        "aliases": {"A": "a"},
    })


def tiny_spec() -> FeatureSpec:
    return FeatureSpec(fs=22050, frame_period=5.0, fft_size=1024, n_mcep=2,
                       n_band=2, use_delta=False, use_delta2=False)


def make_utterance(name, spans, value=0.0, voiced=True) -> UtteranceData:
    """A synthetic analysed utterance; only features/voiced/spans matter."""
    total = max(hi for _phone, _lo, hi in spans)
    dim = tiny_spec().dim
    return UtteranceData(
        name=name, f0=np.zeros(total), sp=np.zeros((total, 3)),
        ap=np.zeros((total, 3)), phones=[], notes=np.zeros(total),
        voiced=np.full(total, bool(voiced)),
        note_semitones=np.zeros(total), relative_pitch=np.zeros(total),
        features=np.full((total, dim), value), phoneme_spans=list(spans))


def context_trainer(**overrides) -> Trainer:
    config = TrainingConfig(normalize_scale=False, n_iterations=1,
                            context_enabled=True, **overrides)
    trainer = Trainer(config, context_phoneme_set())
    trainer.spec = tiny_spec()
    return trainer


def demo_spans():
    """Two utterances exercising boundaries, aliases and repeats."""
    utt1 = make_utterance("u1", [("s", 0, 4), ("A", 4, 12), ("i", 12, 20)])
    utt2 = make_utterance("u2", [("a", 0, 8), ("s", 8, 12)])
    return [utt1, utt2]


def simple_hmm(n_states=1) -> LeftToRightHMM:
    return LeftToRightHMM(n_states=n_states)


def make_resolution_model(with_global=True, contexts=True) -> HMSModel:
    """A hand-built model covering every rung of the fallback hierarchy."""
    phoneme_set = context_phoneme_set()
    hmms = {"a": simple_hmm(), "i": simple_hmm()}
    backoff = {"vowel": simple_hmm()}
    context_hmms = {}
    context_index = {}
    if contexts:
        context_hmms = {
            "s^a^i": simple_hmm(),
            "s^a^_": simple_hmm(),
            "_^a^i": simple_hmm(),
        }
        context_index = {
            "s^a^i": {"kind": KIND_TRIPHONE, "frames": 500, "occurrences": 9},
            "s^a^_": {"kind": KIND_LEFT, "frames": 100, "occurrences": 4},
            "_^a^i": {"kind": KIND_RIGHT, "frames": 50, "occurrences": 2},
        }
    return HMSModel(
        name="resolution", spec=tiny_spec(), phoneme_set=phoneme_set, hmms=hmms,
        backoff=backoff, contexts=context_hmms, context_index=context_index,
        global_backoff=simple_hmm() if with_global else None)


def write_subset_labels(demo_dataset, path: Path,
                        names=("vowel_scale_a",)) -> Path:
    lines = ["# utt_id\tonset\toffset\tphone\tnote"]
    for line in Path(demo_dataset["labels"]).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split()[0] in names:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# keys and configuration
# --------------------------------------------------------------------------


def test_context_keys_and_boundary_neighbours_use_sil_not_bos_eos():
    phones = ["s", "a", "i"]
    triples = list(neighbor_contexts(phones, "sil"))
    assert triples == [("sil", "s", "a"), ("s", "a", "i"), ("a", "i", "sil")]

    keys = context_keys("s", "a", "i", "_", partial=True)
    assert keys == [("s^a^i", KIND_TRIPHONE), ("s^a^_", KIND_LEFT),
                    ("_^a^i", KIND_RIGHT)]
    assert context_keys("s", "a", "i", "_", partial=False) == \
        [("s^a^i", KIND_TRIPHONE)]
    assert triphone_key("s", "a", "i") == "s^a^i"
    assert left_diphone_key("s", "a", "_") == "s^a^_"
    assert right_diphone_key("a", "i", "_") == "_^a^i"
    assert CONTEXT_SEPARATOR == "^"
    # left and right diphone keys of the same bigram never collide
    assert left_diphone_key("s", "a", "_") != right_diphone_key("s", "a", "_")


def test_context_wildcard_avoids_inventory_symbols():
    assert context_wildcard(["a", "sil"]) == "_"
    assert context_wildcard(["a", "_"]) == "?"
    assert context_wildcard(["_", "?", "a"]) == "*"


def test_context_wildcard_exhaustion_raises():
    with pytest.raises(ValueError, match="wildcard"):
        context_wildcard(["_", "?", "*", "~"])


def test_context_config_defaults_off_and_validated():
    config = TrainingConfig()
    assert config.context_enabled is False
    assert config.context_min_frames >= 1
    assert config.context_min_occurrences >= 1
    assert config.context_max_models >= 0
    assert config.context_partial is True
    assert config.context_global_backoff is False

    with pytest.raises(ValueError, match="context_min_frames"):
        TrainingConfig(context_min_frames=0)
    with pytest.raises(ValueError, match="context_min_occurrences"):
        TrainingConfig(context_min_occurrences=0)
    with pytest.raises(ValueError, match="context_max_models"):
        TrainingConfig(context_max_models=-1)


def test_context_section_of_parameters_yaml_is_loaded():
    config = training_config_from_parameters(load_parameters())
    assert config.context_enabled is False       # bundled default stays off

    config = training_config_from_parameters({
        "context": {"enabled": True, "min_frames": 42, "min_occurrences": 7,
                    "max_models": 5, "partial": False,
                    "global_backoff": True}})
    assert config.context_enabled is True
    assert config.context_min_frames == 42
    assert config.context_min_occurrences == 7
    assert config.context_max_models == 5
    assert config.context_partial is False
    assert config.context_global_backoff is True


def test_unknown_context_option_is_rejected():
    with pytest.raises(ValueError, match="unknown context option"):
        training_config_from_parameters({"context": {"bogus": 1}})


# --------------------------------------------------------------------------
# collection, selection and training
# --------------------------------------------------------------------------


def test_collect_context_data_keys_boundaries_and_aliases():
    trainer = context_trainer()
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    occurrences = trainer.collect_context_data(demo_spans(), offset, scale)

    expected = {
        # utterance 1: s a i (the alias A is canonicalised to a)
        "sil^s^a", "sil^s^_", "_^s^a",
        "s^a^i", "s^a^_", "_^a^i",
        "a^i^sil", "a^i^_", "_^i^sil",
        # utterance 2: a s
        "sil^a^s", "sil^a^_", "_^a^s",
        "a^s^sil", "a^s^_", "_^s^sil",
    }
    assert set(occurrences) == expected
    # boundary triphones use the existing sil symbol, not BOS/EOS tokens
    assert occurrences["sil^s^a"]["kind"] == KIND_TRIPHONE
    assert occurrences["a^i^sil"]["post"] == "sil"
    assert occurrences["sil^a^s"]["pre"] == "sil"
    # diphone sides are pooled separately (different frames)
    assert occurrences["s^a^_"]["kind"] == KIND_LEFT
    assert occurrences["s^a^_"]["post"] is None
    assert occurrences["_^s^a"]["kind"] == KIND_RIGHT
    assert occurrences["_^s^a"]["pre"] is None
    # the left context of 'a' in u1 pools exactly the 8 frames of that span
    entry = occurrences["s^a^_"]
    assert len(entry["features"]) == 1
    assert len(entry["features"][0]) == 8
    assert entry["voiced"][0].dtype == bool


def test_collect_context_data_without_partial_diphones():
    trainer = context_trainer(context_partial=False)
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    occurrences = trainer.collect_context_data(demo_spans(), offset, scale)
    assert all(key.count("^") == 2 for key in occurrences)
    assert all(entry["kind"] == KIND_TRIPHONE
               for entry in occurrences.values())


def test_select_context_models_thresholds_cap_and_determinism():
    trainer = context_trainer(context_min_frames=10, context_min_occurrences=2,
                              context_max_models=2)

    def entry(frames, count):
        return {"kind": KIND_TRIPHONE, "pre": "s", "curr": "a", "post": "i",
                "features": [np.zeros((frames // count, 2))] * count,
                "voiced": [np.zeros(frames // count, dtype=bool)] * count}

    occurrences = {
        "big": entry(30, 3),
        "medium": entry(20, 2),
        "small": entry(12, 2),
        "few_frames": entry(6, 3),          # below min_frames
        "few_occurrences": entry(40, 1),    # below min_occurrences
    }
    selected = trainer.select_context_models(occurrences)
    assert selected == ["big", "medium"]   # capped, best-supported first

    trainer.config.context_max_models = 0
    assert trainer.select_context_models(occurrences) == []
    trainer.config.context_max_models = 100
    # the threshold-filtered set, best-supported first; the two
    # under-supported contexts stay excluded at any cap
    assert trainer.select_context_models(occurrences) == \
        ["big", "medium", "small"]
    # deterministic: same input, same answer regardless of insertion order
    assert trainer.select_context_models(occurrences) == \
        trainer.select_context_models(dict(reversed(list(occurrences.items()))))


def test_train_contexts_uses_current_phone_budget_and_records_support():
    trainer = context_trainer()
    offset = np.zeros(trainer.spec.static_dim)
    scale = np.ones(trainer.spec.static_dim)
    occurrences = trainer.collect_context_data(demo_spans(), offset, scale)

    trainer.config.context_min_frames = 1
    trainer.config.context_min_occurrences = 1
    selected = ["s^a^_", "_^s^a"]
    contexts, index = trainer.train_contexts(occurrences, selected,
                                             seed_base=7)
    assert set(contexts) == set(selected)
    # 'a' is a vowel (2 states in the test inventory); 's' an unvoiced
    # consonant -- each context model spends its current phone's budget
    assert contexts["s^a^_"].n_states == 2
    assert contexts["_^s^a"].n_states == 2
    assert index["s^a^_"]["kind"] == KIND_LEFT
    assert index["s^a^_"]["curr"] == "a"
    assert index["s^a^_"]["frames"] == 8
    assert index["s^a^_"]["occurrences"] == 1
    assert index["s^a^_"]["n_free_params"] == contexts["s^a^_"].n_free_params
    assert contexts["s^a^_"].is_trained()


# --------------------------------------------------------------------------
# resolution hierarchy
# --------------------------------------------------------------------------


def test_resolve_unit_fallback_hierarchy():
    model = make_resolution_model()

    key, hmm, tier = model.resolve_unit("s", "a", "i")
    assert (key, hmm, tier) == ("s^a^i", model.contexts["s^a^i"], "triphone")

    # no exact triphone: best-supported partial wins (left: 100 > right: none)
    key, hmm, tier = model.resolve_unit("s", "a", "sil")
    assert (key, tier) == ("s^a^_", "left") and hmm is model.contexts["s^a^_"]

    # only the right diphone exists for this neighbourhood
    key, hmm, tier = model.resolve_unit("t", "a", "i")
    assert (key, tier) == ("_^a^i", "right") and hmm is model.contexts["_^a^i"]

    # tie between left and right supports favours the left context
    model.contexts["_^a^sil"] = simple_hmm()
    model.context_index["_^a^sil"] = {"kind": KIND_RIGHT, "frames": 100,
                                      "occurrences": 4}
    key, hmm, tier = model.resolve_unit("s", "a", "sil")
    assert (key, tier) == ("s^a^_", "left")

    # no context at all: the dedicated current-phone HMM
    key, hmm, tier = model.resolve_unit("s", "i", "sil")
    assert (key, tier) == (None, "phone") and hmm is model.hmms["i"]

    # inventory phone with no dedicated model: phone-class backoff
    key, hmm, tier = model.resolve_unit("s", "rare", "sil")
    assert (key, tier) == (None, "class") and hmm is model.backoff["vowel"]

    # unknown phone, no class backoff either: the global backoff model
    key, hmm, tier = model.resolve_unit("s", "zz", "sil")
    assert (key, tier) == (None, "global") and hmm is model.global_backoff

    # without a global backoff the hierarchy raises, like get_or_backoff
    model.global_backoff = None
    with pytest.raises(KeyError):
        model.resolve_unit("s", "zz", "sil")


def test_resolve_unit_without_contexts_matches_get_or_backoff():
    model = make_resolution_model(contexts=False, with_global=False)
    for pre, curr, post in [("s", "a", "i"), ("s", "rare", "i")]:
        key, hmm, tier = model.resolve_unit(pre, curr, post)
        assert key is None
        assert hmm is model.get_or_backoff(curr)
        assert tier == ("phone" if model.get_hmm(curr) is not None
                        else "class")
    # and it fails exactly like get_or_backoff for an uncoverable phone
    with pytest.raises(KeyError):
        model.resolve_unit("s", "zz", "i")
    with pytest.raises(KeyError):
        model.get_or_backoff("zz")


# --------------------------------------------------------------------------
# serialisation (format 3 + format-2 compatibility)
# --------------------------------------------------------------------------


def test_context_model_save_load_roundtrip(trained_context_model, tmp_path):
    model = trained_context_model
    assert model.contexts, "fixture should train at least one context model"
    assert model.global_backoff is not None

    directory = tmp_path / "model"
    model.save(directory)
    assert (directory / "context.npz").exists()
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    assert f"format_version: {MODEL_FORMAT_VERSION}" in text
    assert "context_index" in text and "global_backoff_index" in text

    loaded = HMSModel.load(directory)
    assert loaded.loaded_format_version == MODEL_FORMAT_VERSION
    assert sorted(loaded.contexts) == sorted(model.contexts)
    assert loaded.context_index == model.context_index
    assert loaded.context_n_free_params == model.context_n_free_params
    assert loaded.global_backoff_n_free_params == \
        model.global_backoff_n_free_params
    assert loaded.n_free_params == model.n_free_params
    for key, hmm in model.contexts.items():
        assert np.allclose(loaded.contexts[key].states[0].gmm.means,
                           hmm.states[0].gmm.means)
    # resolution behaves identically after the round trip
    probes = [("sil", "a", "sil"), ("m", "a", "m"), ("s", "a", "sil"),
              ("a", "i", "u"), ("q", "zz", "q")]
    for probe in probes:
        key_a, _hmm_a, tier_a = model.resolve_unit(*probe)
        key_b, _hmm_b, tier_b = loaded.resolve_unit(*probe)
        assert (key_a, tier_a) == (key_b, tier_b)


def test_model_without_contexts_writes_no_context_payload(
        trained_model, tmp_path):
    directory = tmp_path / "model"
    trained_model.save(directory)
    assert not (directory / "context.npz").exists()
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    assert "context_index" not in text
    assert "global_backoff_index" not in text


def test_missing_context_npz_is_a_clean_error(trained_context_model, tmp_path):
    directory = tmp_path / "model"
    trained_context_model.save(directory)
    (directory / "context.npz").unlink()
    with pytest.raises(ValueError, match="context.npz"):
        HMSModel.load(directory)


def test_global_key_cannot_collide_with_context_keys():
    assert CONTEXT_SEPARATOR not in GLOBAL_KEY


# --------------------------------------------------------------------------
# parameter accounting and inspection
# --------------------------------------------------------------------------


def test_parameter_report_separates_all_four_tiers(trained_context_model):
    model = trained_context_model
    report = "\n".join(model.parameter_report())
    for line in ("phoneme HMM params", "context HMM params",
                 "backoff HMM params", "global HMM params",
                 "total HMM params"):
        assert line in report
    assert model.n_free_params == (model.phoneme_n_free_params
                                   + model.context_n_free_params
                                   + model.backoff_n_free_params
                                   + model.global_backoff_n_free_params)
    assert model.context_n_free_params > 0
    assert model.global_backoff_n_free_params > 0
    assert "context models" in report

    summary = model.summary()
    assert "context HMMs" in summary
    assert next(iter(sorted(model.contexts))) in summary


def test_context_free_model_reports_zero_context_budget(trained_model):
    report = "\n".join(trained_model.parameter_report())
    assert "context HMM params: 0" in report
    assert "global HMM params : 0" in report
    assert "context HMMs" not in trained_model.summary()


# --------------------------------------------------------------------------
# synthesis integration
# --------------------------------------------------------------------------


def test_plan_api_is_unchanged_and_uses_context_units(
        trained_context_model, score):
    model = trained_context_model
    synthesizer = Synthesizer(model, SynthesisConfig(seed=0,
                                                     vocoder="builtin"))
    planned = synthesizer.plan(score)
    assert len(planned) == 6        # the existing six-value plan() API
    frame_phones, state_ids, segment_ids, notes, segment_frames, \
        diagnostics = planned
    assert len(frame_phones) == len(state_ids) == len(segment_ids) \
        == len(notes)
    assert sum(segment_frames) == len(frame_phones)
    assert isinstance(diagnostics, list)

    units = synthesizer._frame_units
    assert units is not None and len(units) == len(frame_phones)
    known = {id(h) for h in list(model.hmms.values())
             + list(model.contexts.values()) + list(model.backoff.values())
             + [model.global_backoff]}
    assert all(id(unit) in known for unit in units)
    context_ids = {id(h) for h in model.contexts.values()}
    assert any(id(unit) in context_ids for unit in units), \
        "at least one frame should be served by a sparse context HMM"


def test_context_synthesis_end_to_end(trained_context_model, short_score):
    synthesizer = Synthesizer(trained_context_model,
                              SynthesisConfig(seed=0, vocoder="builtin"))
    result = synthesizer.synthesize(short_score)
    assert len(result.audio) > 0
    assert np.isfinite(result.audio).all()
    expected = sum(u.end - u.start for u in short_score)
    assert result.duration == pytest.approx(expected, abs=0.05)

    # deterministic for a fixed seed
    again = Synthesizer(trained_context_model,
                        SynthesisConfig(seed=0,
                                        vocoder="builtin")).synthesize(short_score)
    assert np.array_equal(result.audio, again.audio)


def test_context_disabled_keeps_the_classic_path(trained_model, short_score):
    synthesizer = Synthesizer(trained_model, SynthesisConfig(seed=0,
                                                              vocoder="builtin"))
    planned = synthesizer.plan(short_score)
    assert len(planned) == 6
    assert synthesizer._frame_units is None      # no context resolution
    assert trained_model.contexts == {}
    assert trained_model.global_backoff is None
    result = synthesizer.synthesize(short_score)
    assert np.isfinite(result.audio).all()


# --------------------------------------------------------------------------
# CLI: hms train --context / --no-context, inspect-model reporting
# --------------------------------------------------------------------------


def test_cli_train_context_flag_and_inspection(tmp_path, demo_dataset,
                                               capsys):
    labels = write_subset_labels(demo_dataset, tmp_path / "labels.tsv")
    parameters = tmp_path / "parameters.yaml"
    parameters.write_text(
        "context:\n"
        "  enabled: false\n"
        "  min_frames: 10\n"
        "  min_occurrences: 1\n"
        "  max_models: 8\n"
        "  global_backoff: true\n", encoding="utf-8")

    model_dir = tmp_path / "model"
    assert main(["train", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--out", str(model_dir), "--fs", "22050", "--iterations", "1",
                 "--config", str(parameters), "--context"]) == 0
    assert (model_dir / "context.npz").exists()
    text = (model_dir / "model.yaml").read_text(encoding="utf-8")
    assert "context_index" in text and "global_backoff_index" in text
    assert f"format_version: {MODEL_FORMAT_VERSION}" in text

    capsys.readouterr()
    assert main(["inspect-model", "--model", str(model_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_models"], "context models must be inspected"
    assert payload["global_backoff"] is not None
    breakdown = payload["parameter_breakdown"]
    assert set(breakdown) == {"phoneme", "context", "class_backoff",
                              "global_backoff"}
    assert breakdown["context"] > 0 and breakdown["global_backoff"] > 0
    assert sum(breakdown.values()) == payload["parameter_budget"]

    # a context model synthesises through the CLI too
    score = tmp_path / "score.tsv"
    score.write_text("demo\t0.0\t0.1\tsil\t-\n"
                     "demo\t0.1\t0.7\ta\t60\n"
                     "demo\t0.7\t0.8\tsil\t-\n", encoding="utf-8")
    assert main(["synth", "--model", str(model_dir), "--score", str(score),
                 "--out", str(tmp_path / "out.wav"), "--seed", "0",
                 "--vocoder", "builtin"]) == 0


def test_cli_no_context_flag_overrides_enabled_config(tmp_path, demo_dataset):
    labels = write_subset_labels(demo_dataset, tmp_path / "labels.tsv")
    parameters = tmp_path / "parameters.yaml"
    parameters.write_text("context:\n  enabled: true\n  min_frames: 10\n"
                          "  min_occurrences: 1\n", encoding="utf-8")

    model_dir = tmp_path / "model"
    assert main(["train", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--out", str(model_dir), "--fs", "22050", "--iterations", "1",
                 "--config", str(parameters), "--no-context"]) == 0
    assert not (model_dir / "context.npz").exists()
    text = (model_dir / "model.yaml").read_text(encoding="utf-8")
    assert "context_index" not in text
    loaded = HMSModel.load(model_dir)
    assert loaded.contexts == {} and loaded.global_backoff is None
