"""Model serialisation and the human-readable model directory."""

from __future__ import annotations


import numpy as np
import pytest

from hms.core.model import MODEL_FORMAT_VERSION, HMSModel, load_model


def test_geometry_and_reports(trained_model):
    model = trained_model
    assert model.feature_dim == model.spec.dim
    assert model.static_dim == model.spec.static_dim
    assert model.static_dim == 1 + model.spec.n_mcep + model.spec.n_band
    assert model.n_free_params > 0
    assert model.stats.frames > 0
    assert model.stats.duration_seconds > 0
    report = "\n".join(model.parameter_report())
    assert "params per frame" in report
    summary = model.summary()
    assert model.name in summary
    assert "a" in summary and "sil" in summary


def test_normalisation_roundtrip(trained_model):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(7, trained_model.static_dim))
    assert np.allclose(trained_model.denormalize(
        trained_model.normalize(values)), values, atol=1e-9)


def test_unfitted_model_still_normalises(trained_model):
    """A model with no normalisation must behave like the identity."""
    naked = HMSModel(name="bare", spec=trained_model.spec,
                     phoneme_set=trained_model.phoneme_set, hmms={})
    values = np.arange(2 * naked.static_dim, dtype=float).reshape(2, -1)
    assert np.allclose(naked.normalize(values), values)
    assert np.allclose(naked.denormalize(values), values)


def test_phoneme_lookup_falls_back_to_a_class_model(trained_model):
    model = trained_model
    assert model.get_hmm("a") is not None
    assert model.get_hmm("zzz") is None
    backoff = model.get_or_backoff("zzz")
    assert backoff.n_states >= 1
    assert backoff.states[0].gmm.n_components >= 1
    # known phonemes are returned as-is, not via backoff
    assert model.get_or_backoff("a") is model.get_hmm("a")
    assert model.backoff, "training should have pooled at least one class"


def test_save_load_roundtrip(trained_model, tmp_path):
    model = trained_model
    directory = tmp_path / "model"
    model.save(directory)
    assert (directory / "model.yaml").exists()
    assert (directory / "hmm.npz").exists()

    loaded = HMSModel.load(directory)
    assert loaded.name == model.name
    assert loaded.spec.to_dict() == model.spec.to_dict()
    assert loaded.phoneme_set.symbols == model.phoneme_set.symbols
    assert sorted(loaded.hmms) == sorted(model.hmms)
    assert loaded.n_free_params == model.n_free_params
    assert loaded.stats.frames == model.stats.frames
    assert np.allclose(loaded.feature_offset, model.feature_offset)
    assert np.allclose(loaded.feature_scale, model.feature_scale)
    assert sorted(loaded.backoff) == sorted(model.backoff)
    # ... and it behaves identically
    rng = np.random.default_rng(1)
    values = rng.normal(size=(5, model.static_dim))
    assert np.allclose(loaded.normalize(values), model.normalize(values))
    for phoneme in sorted(model.hmms):
        original, copy = model.hmms[phoneme], loaded.hmms[phoneme]
        assert copy.n_states == original.n_states
        assert np.allclose(copy.self_loops, original.self_loops)
        assert np.allclose(copy.states[0].gmm.means,
                           original.states[0].gmm.means)
    # the pitch and duration models survive too
    assert loaded.duration_model.mean_frames("a") == pytest.approx(
        model.duration_model.mean_frames("a"))
    assert loaded.pitch_model.voiced_prior == model.pitch_model.voiced_prior


def test_load_model_helper_and_errors(trained_model, tmp_path):
    directory = tmp_path / "model"
    trained_model.save(directory)
    assert load_model(directory).name == trained_model.name
    assert trained_model.exists(directory)
    with pytest.raises(FileNotFoundError):
        HMSModel.load(tmp_path / "definitely-not-a-model")


def test_model_yaml_is_human_readable(trained_model, tmp_path):
    directory = tmp_path / "model"
    trained_model.save(directory)
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    assert f"format_version: {MODEL_FORMAT_VERSION}" in text
    assert "feature_spec" in text and "hmm_index" in text
    # no pickle / opaque blobs: it must be plain YAML
    assert "\x00" not in text
    assert "!!python" not in text


def test_model_stats_roundtrip():
    from hms.core.model import ModelStats

    stats = ModelStats(utterances=3, frames=100, phoneme_occurrences=9,
                       duration_seconds=0.5, n_iterations=2)
    again = ModelStats.from_dict(stats.to_dict())
    assert again.utterances == 3 and again.frames == 100
    assert again.duration_seconds == pytest.approx(0.5)
