"""Training pipeline regressions: raw, frame-weighted class backoffs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hms.config import training_config_from_parameters
from hms.core import labels as labels_module
from hms.core.duration import DurationModel, DurationStats
from hms.core.labels import Score, Segment, Utterance
from hms.core.phonemes import PhonemeSet
from hms.core.synthesizer import SynthesisConfig, Synthesizer
from hms.core.trainer import Trainer, TrainingConfig


def simple_vowel_set() -> PhonemeSet:
    return PhonemeSet.from_dict({
        "defaults": {
            "vowel": {"n_states": 1, "n_components": 2, "voiced": True},
        },
        "phonemes": {
            "rare": {"type": "vowel", "n_states": 1, "n_components": 1},
            "common": {"type": "vowel", "n_states": 1, "n_components": 1},
        },
    })


def test_class_backoff_fits_pooled_frames_with_frame_weighting():
    """Rare/common phones form real mixture regions; their models are not averaged."""
    rng = np.random.default_rng(17)
    rare = rng.normal(-8.0, 0.15, size=(20, 2))
    common = rng.normal(8.0, 0.15, size=(2000, 2))
    features = {"rare": [rare], "common": [common]}
    voiced = {
        "rare": [np.ones(len(rare), dtype=bool)],
        "common": [np.ones(len(common), dtype=bool)],
    }
    trainer = Trainer(TrainingConfig(n_iterations=3), simple_vowel_set())

    backoff = trainer.build_backoff(features, voiced)["vowel"]
    gmm = backoff.states[0].gmm
    order = np.argsort(gmm.means[:, 0])
    means = gmm.means[order, 0]
    weights = gmm.weights[order]

    # Components are fitted to both pooled acoustic regions, not an average
    # near zero between independently trained phone GMM means.
    assert np.allclose(means, [-8.0, 8.0], atol=0.5)
    # Each frame contributes once: the 20-frame phone gets about 1% influence,
    # not the same half-mixture weight as the 2000-frame phone.
    assert 0.005 < weights[0] < 0.05
    assert weights[1] > 0.95


def test_under_threshold_phones_are_pooled_and_can_train_a_backoff_only_model():
    """No rare frame is dropped just because no individual phone passes threshold."""
    trainer = Trainer(
        TrainingConfig(min_phoneme_frames=20, n_iterations=2),
        simple_vowel_set())
    features = {
        "rare": [np.full((5, 2), -1.0)],
        "common": [np.full((6, 2), 1.0)],
    }
    voiced = {
        "rare": [np.ones(5, dtype=bool)],
        "common": [np.ones(6, dtype=bool)],
    }

    dedicated = trainer.train_hmms(features, voiced)
    assert dedicated == {}
    backoff = trainer.build_backoff(features, voiced)
    assert set(backoff) == {"vowel"}
    assert backoff["vowel"].is_trained()
    assert backoff["vowel"].states[0].duration.count > 0


def test_training_configuration_rejects_invalid_covariance_and_threshold():
    with pytest.raises(ValueError, match="covariance_type"):
        TrainingConfig(covariance_type="full")
    with pytest.raises(ValueError, match="min_phoneme_frames"):
        TrainingConfig(min_phoneme_frames=0)


@pytest.mark.parametrize("bad_note", [-1.0, 128.0, 200.0, float("nan")])
def test_training_config_rejects_an_absurd_default_note(bad_note):
    # nan is caught by the generic finiteness check, the rest by the
    # MIDI-range check; either way it is a ValueError, never silent
    with pytest.raises(ValueError, match="default_note|finite"):
        TrainingConfig(default_note=bad_note)


@pytest.mark.parametrize("good_note", [0.0, 60.0, 127.0])
def test_training_config_accepts_a_valid_default_note(good_note):
    assert TrainingConfig(default_note=good_note).default_note == good_note


def _subset_labels(demo_dataset, path, utterance="vowel_scale_a"):
    """A one-utterance subset of the demo labels."""
    lines = ["# utt_id\tonset\toffset\tphone\tnote"]
    for line in Path(demo_dataset["labels"]).read_text(
            encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.split("\t")[0] == utterance:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _small_train_config(labels, wav_dir, **overrides):
    values = dict(label_file=str(labels), wav_dir=str(wav_dir),
                  fs=22050, fft_size=1024, n_mcep=10, n_band=4,
                  n_iterations=1, min_phoneme_frames=10, vocoder="builtin")
    values.update(overrides)
    return TrainingConfig(**values)


def test_notes_outside_the_analysis_f0_range_are_warned(demo_dataset,
                                                        phoneme_set,
                                                        tmp_path):
    """Label notes the extractor can never reach are reported, not silent."""
    labels = _subset_labels(demo_dataset, tmp_path / "labels.tsv")
    lines = []
    for line in labels.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
            continue
        parts = line.split("\t")
        if parts[-1] not in ("-",):
            parts[-1] = "127"          # ~12.5 kHz, far above f0_ceil
        lines.append("\t".join(parts))
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")

    messages = []
    model = Trainer(_small_train_config(labels, demo_dataset["wav_dir"]),
                    phoneme_set, log=messages.append).train()
    assert model.hmms or model.backoff, "the model still trains"
    joined = "\n".join(messages)
    assert "outside the analysis F0 range" in joined
    assert "127" in joined


def test_notes_inside_the_analysis_f0_range_produce_no_pitch_warning(
        demo_dataset, phoneme_set, tmp_path):
    labels = _subset_labels(demo_dataset, tmp_path / "labels.tsv")
    messages = []
    Trainer(_small_train_config(labels, demo_dataset["wav_dir"]),
            phoneme_set, log=messages.append).train()
    assert not any("analysis F0 range" in message for message in messages)


def test_note_range_warnings_flag_only_the_notes_the_extractor_cannot_reach(
        phoneme_set):
    trainer = Trainer(_small_train_config("unused.tsv", "unused-dir"),
                      phoneme_set)
    in_range = labels_module.parse("u\t0.0\t0.5\ta\t60\n"
                                   "u\t0.5\t1.0\ta\t75\n")
    assert trainer.note_range_warnings(in_range) == []
    out_of_range = labels_module.parse("u\t0.0\t0.5\ta\t127\n"
                                       "u\t0.5\t1.0\tsil\t-\n")
    warnings = trainer.note_range_warnings(out_of_range)
    assert len(warnings) == 1
    assert warnings[0].startswith("u:")
    assert "127" in warnings[0]
    assert "12544" in warnings[0]


@pytest.mark.parametrize("value", [-0.1, float("nan"), "invalid"])
def test_training_config_rejects_invalid_duration_variance_scale(value):
    with pytest.raises(ValueError, match="duration_variance_scale|finite|numeric"):
        TrainingConfig(duration_variance_scale=value)


def test_duration_variation_setting_is_loaded_and_round_trips():
    config = training_config_from_parameters({
        "duration": {"variance_scale": 0.25},
    })
    trainer = Trainer(config, simple_vowel_set())
    model = trainer.build_duration_model({"rare": [10.0, 12.0]})
    assert model.variance_scale == pytest.approx(0.25)
    restored = DurationModel.from_dict(model.to_dict())
    assert restored.variance_scale == pytest.approx(0.25)


def test_model_duration_mode_samples_variation_reproducibly(trained_model):
    import copy

    model = copy.deepcopy(trained_model)
    stats = {
        "a": DurationStats(mean=float(np.log(40.0)), variance=0.8, count=3),
    }
    score = Score([Utterance("timing", [
        Segment("a", 0.0, 0.5, note=60.0),
    ])])

    def frames(seed, variance_scale=1.0):
        model.duration_model = DurationModel(
            stats=stats, variance_scale=variance_scale)
        synthesizer = Synthesizer(
            model, SynthesisConfig(duration_mode="model", seed=seed))
        return synthesizer.plan(score)[4][0]

    assert frames(0) == frames(0)              # same seed, same sampled timing
    assert frames(0) != frames(1)              # a different seed changes it
    assert frames(0, 0.0) == frames(1, 0.0)    # zero variance is deterministic
