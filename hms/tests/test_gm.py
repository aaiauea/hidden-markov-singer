"""Experimental Global Variance Matching (GM): closed-form variance matching.

GM is a lightweight post-MLPG variance rescaling, distinct from the iterative,
likelihood-aware GV optimizer in `hms.core.gv.optimize_global_variance`.
Tests cover the unit function, end-to-end synthesis, CLI flags, backwards
compatibility and the GV/GM mutual-exclusivity rule.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from hms.cli.main import main
from hms.core.generation import mlpg
from hms.core.gv import (
    GlobalVarianceStats,
    GV_ACTIVE_VARIANCE,
    match_global_variance,
    trajectory_variance,
)
from hms.core.labels import Score
from hms.core.model import HMSModel, MODEL_FORMAT_VERSION
from hms.core.synthesizer import SynthesisConfig, Synthesizer


def _step_problem(dim=2, delta2=False):
    """Frame means with sharp states but small delta variances: MLPG smooths."""
    n = 70
    static = np.zeros((n, dim))
    static[20:45, 0] = 0.8
    static[45:, 0] = -0.6
    if dim > 1:
        static[25:55, 1] = -0.3
        static[55:, 1] = 0.9
    if dim > 2:
        static[10:45, 2] = 0.5
    n_streams = 3 if delta2 else 2
    means = np.concatenate([static] + [np.zeros_like(static)] * (n_streams - 1))
    variances = np.concatenate([np.full_like(static, 0.15)] +
                               [np.full_like(static, 0.008)] * (n_streams - 1))
    sizes = (dim,) * n_streams
    return mlpg(means, variances, sizes), variances, sizes, static


# --------------------------------------------------------------------------
# Unit: match_global_variance
# --------------------------------------------------------------------------


def test_strength_zero_returns_trajectory_unchanged():
    rng = np.random.default_rng(7)
    c = rng.normal(size=(60, 3))
    c[:, 2] = 4.0  # near-constant feature
    target = np.array([2.0, 0.5, 3.0])
    out = match_global_variance(c, target, strength=0.0)
    assert np.array_equal(out, c)


def test_strength_one_matches_target_variance_for_eligible_dims():
    rng = np.random.default_rng(3)
    c = rng.normal(size=(80, 3)) * 0.4
    c[:, 2] = 7.0  # constant-ish -> skipped
    target = np.array([2.0, 0.25, 5.0])
    out = match_global_variance(c, target, strength=1.0)
    after = trajectory_variance(out)
    before = trajectory_variance(c)
    for d in range(3):
        if before[d] > GV_ACTIVE_VARIANCE:
            assert abs(after[d] - target[d]) < 1e-6, (d, after[d], target[d])
        else:
            assert after[d] == before[d], "constant dimension must stay unchanged"


def test_strength_one_on_trajectory_already_at_target_is_identity():
    rng = np.random.default_rng(5)
    c = rng.normal(size=(40, 2))
    target = trajectory_variance(c)
    out = match_global_variance(c, target, strength=1.0)
    # rescaling by ratio == 1 leaves every eligible feature exactly as-is
    assert np.allclose(out, c, atol=1e-12)


def test_partial_strength_interpolates_toward_target():
    rng = np.random.default_rng(11)
    c = rng.normal(size=(90, 2)) * 0.3
    target = np.array([2.0, 0.05])
    before = trajectory_variance(c)
    full = match_global_variance(c, target, strength=1.0)
    half = match_global_variance(c, target, strength=0.5)
    after_full = trajectory_variance(full)
    after_half = trajectory_variance(half)
    for d in range(2):
        assert min(before[d], after_full[d]) <= after_half[d] <= max(before[d], after_full[d])


def test_constant_or_near_zero_variance_dimensions_are_skipped():
    c = np.zeros((30, 3))
    c[:, 0] = np.arange(30) * 0.1  # real variance
    c[:, 1] = 3.0  # exactly constant
    c[:, 2] = 3.0 + 1e-13 * np.arange(30)  # near-zero variance
    target = np.array([5.0, 5.0, 5.0])
    out = match_global_variance(c, target, strength=1.0)
    assert np.array_equal(out[:, 1], c[:, 1])  # constant untouched
    assert np.array_equal(out[:, 2], c[:, 2])  # near-constant untouched
    assert not np.array_equal(out[:, 0], c[:, 0])  # eligible feature changed


def test_short_sequence_returns_unchanged_copy():
    for n in (0, 1):
        c = np.zeros((n, 2), dtype=float) if n == 0 else np.array([[0.5, -1.0]])
        target = np.array([1.0, 2.0])
        out = match_global_variance(c, target, strength=1.0)
        assert np.array_equal(out, c)


def test_inputs_are_never_mutated():
    rng = np.random.default_rng(13)
    c = rng.normal(size=(45, 2))
    target = np.array([1.5, 0.4])
    c_before, target_before = c.copy(), target.copy()
    match_global_variance(c, target, strength=1.0)
    assert np.array_equal(c, c_before)
    assert np.array_equal(target, target_before)


@pytest.mark.parametrize("bad", [
    {"trajectory": np.array([[1.0], [np.nan]]), "target_variance": np.array([1.0]), "strength": 1.0},
    {"trajectory": np.ones((3, 1)), "target_variance": np.array([1.0, 2.0]), "strength": 1.0},
    {"trajectory": np.ones((3, 1)), "target_variance": np.array([-1.0]), "strength": 1.0},
    {"trajectory": np.ones((3, 1)), "target_variance": np.array([np.inf]), "strength": 1.0},
    {"trajectory": np.ones((3, 1)), "target_variance": np.array([1.0]), "strength": -0.1},
    {"trajectory": np.ones((3, 1)), "target_variance": np.array([1.0]), "strength": 1.1},
    {"trajectory": np.ones((3, 1)), "target_variance": np.array([1.0]), "strength": float("nan")},
    {"trajectory": np.ones((3, 1)), "target_variance": np.array([1.0]), "strength": float("inf")},
])
def test_match_global_variance_rejects_invalid_inputs(bad):
    kwargs = dict(trajectory=np.array([[1.0], [2.0]]),
                  target_variance=np.array([1.0]), strength=1.0)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        match_global_variance(**kwargs)


# --------------------------------------------------------------------------
# Integration: end-to-end synthesis
# --------------------------------------------------------------------------


def test_gm_enabled_changes_spectral_static_features(trained_model, short_score):
    small = Score(short_score.utterances[:1])
    config = dict(seed=0, vibrato=False, vocoder="builtin")
    baseline = Synthesizer(trained_model, SynthesisConfig(**config)).synthesize(small)
    enabled = Synthesizer(trained_model, SynthesisConfig(
        **config, gm_enabled=True, gm_strength=1.0)).synthesize(small)
    assert len(enabled.audio) == len(baseline.audio)
    assert np.isfinite(enabled.audio).all()
    assert np.isfinite(enabled.params.sp).all()
    assert np.isfinite(enabled.params.ap).all()
    # GM alters the static trajectory, so the spectral envelope changes.
    assert np.max(np.abs(enabled.params.sp - baseline.params.sp)) > 1e-6
    # F0 comes from the score in score mode; untouched by GM.
    assert np.array_equal(enabled.params.f0, baseline.params.f0)


def test_gm_requires_model_statistics(trained_model, short_score):
    small = Score(short_score.utterances[:1])
    # strip the GV statistics to emulate a legacy/pre-GV model
    legacy = copy.deepcopy(trained_model)
    legacy.gv_stats = None
    with pytest.raises(ValueError, match="no global_variance"):
        Synthesizer(legacy, SynthesisConfig(
            seed=0, vibrato=False, vocoder="builtin", gm_enabled=True
        )).synthesize(small)


def test_gm_disabled_is_bit_identical_to_default(trained_model, short_score,
                                                monkeypatch):
    small = Score(short_score.utterances[:1])
    config = dict(seed=0, vibrato=False, vocoder="builtin")
    default = Synthesizer(trained_model, SynthesisConfig(**config)).synthesize(small)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("GM matcher ran when disabled")

    monkeypatch.setattr("hms.core.synthesizer.match_global_variance", must_not_run)
    disabled = Synthesizer(trained_model, SynthesisConfig(
        **config, gm_enabled=False, gm_strength=0.2)).synthesize(small)
    assert np.array_equal(default.audio, disabled.audio)
    assert np.array_equal(default.params.sp, disabled.params.sp)
    assert np.array_equal(default.params.ap, disabled.params.ap)
    assert np.array_equal(default.params.f0, disabled.params.f0)


def test_gm_runs_after_mlpg_and_before_denormalisation(trained_model, short_score,
                                                       monkeypatch):
    """Pin the pipeline order: mlpg -> GM -> denormalize."""
    import hms.core.synthesizer as synthesizer_module

    small = Score(short_score.utterances[:1])
    config = dict(seed=0, vibrato=False, vocoder="builtin", gm_enabled=True)
    seen = {}
    real_match = synthesizer_module.match_global_variance

    def spy_match(trajectory, target_variance, strength):
        seen["input"] = np.array(trajectory, copy=True)
        seen["target"] = np.array(target_variance, copy=True)
        seen["strength"] = strength
        out = real_match(trajectory, target_variance, strength)
        seen["output"] = np.array(out, copy=True)
        return out

    monkeypatch.setattr(synthesizer_module, "match_global_variance", spy_match)
    denormalized = []
    real_denormalize = trained_model.denormalize

    def spy_denormalize(static):
        denormalized.append(np.array(static, copy=True))
        return real_denormalize(static)

    monkeypatch.setattr(trained_model, "denormalize", spy_denormalize)
    Synthesizer(trained_model, SynthesisConfig(**config)).synthesize(small)

    # GM received exactly the raw (still normalized) MLPG trajectory ...
    replicator = Synthesizer(trained_model, SynthesisConfig(**config))
    phones, state_ids, *_ = replicator.plan(small)
    means, variances = replicator.frame_statistics(phones, state_ids)
    raw = mlpg(means, variances, trained_model.spec.stream_sizes,
               window=trained_model.spec.delta_window,
               variance_scale=1.0, smooth=True)
    assert seen["input"].shape == raw.shape
    assert np.array_equal(seen["input"], raw)
    assert np.array_equal(seen["target"], trained_model.gv_stats.target_variance)
    assert seen["strength"] == 1.0
    # ... and denormalisation received GM's output, i.e. GM ran in between.
    assert denormalized and np.array_equal(denormalized[0], seen["output"])
    assert not np.array_equal(seen["output"], seen["input"])


def test_gm_and_gv_are_mutually_exclusive(trained_model, short_score):
    small = Score(short_score.utterances[:1])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Synthesizer(trained_model, SynthesisConfig(
            seed=0, vibrato=False, vocoder="builtin",
            gv_enabled=True, gm_enabled=True)).synthesize(small)


def test_gm_strength_validation_rejected():
    with pytest.raises(ValueError, match="gm_strength"):
        SynthesisConfig(gm_strength=-0.5)
    with pytest.raises(ValueError, match="gm_strength"):
        SynthesisConfig(gm_strength=1.5)
    with pytest.raises(ValueError, match="gm_strength"):
        SynthesisConfig(gm_strength=float("nan"))
    # boundaries accepted
    SynthesisConfig(gm_strength=0.0)
    SynthesisConfig(gm_strength=1.0)


def test_gm_enabled_validation_rejected():
    with pytest.raises(ValueError, match="gm_enabled"):
        SynthesisConfig(gm_enabled="yes")


# --------------------------------------------------------------------------
# GV statistics round-trip (GM reuses them unchanged)
# --------------------------------------------------------------------------


def test_gm_reuses_existing_gv_statistics(trained_model):
    spec = trained_model.spec
    assert trained_model.gv_stats is not None
    assert trained_model.gv_stats.target_variance.shape == (spec.static_dim,)
    assert np.isfinite(trained_model.gv_stats.target_variance).all()
    assert (trained_model.gv_stats.target_variance >= 0).all()


def test_gm_section_roundtrips_without_format_bump(trained_model, tmp_path):
    directory = trained_model.save(tmp_path / "with-gm")
    document = yaml.safe_load((directory / "model.yaml").read_text())
    assert document["format_version"] == MODEL_FORMAT_VERSION
    assert document["global_variance"] == trained_model.gv_stats.to_dict()
    loaded = HMSModel.load(directory)
    assert loaded.gv_stats is not None
    assert np.array_equal(loaded.gv_stats.target_variance,
                          trained_model.gv_stats.target_variance)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write_cli_inputs(tmp_path, trained_model):
    directory = trained_model.save(tmp_path / "model")
    score = tmp_path / "score.tsv"
    score.write_text("song\t0.0\t0.1\tsil\t-\n"
                     "song\t0.1\t0.8\ta\t60\n"
                     "song\t0.8\t0.9\tsil\t-\n")
    parameters = tmp_path / "params.yaml"
    parameters.write_text("synthesis:\n  vocoder: builtin\n  seed: 0\n")
    return directory, score, parameters


def _cli_params(tmp_path, name, directory, score, parameters, *overrides):
    params = tmp_path / f"{name}.npz"
    args = ["synth", "--model", str(directory), "--score", str(score),
            "--config", str(parameters), "--save-params", str(params),
            "--out", str(tmp_path / f"{name}.wav"), *overrides]
    assert main(args) == 0
    with np.load(params) as arrays:
        return {key: arrays[key] for key in ("f0", "sp", "ap")}


def test_gm_cli_flag_changes_envelope(trained_model, tmp_path):
    directory, score, parameters = _write_cli_inputs(tmp_path, trained_model)
    off = _cli_params(tmp_path, "off", directory, score, parameters, "--no-gm")
    on = _cli_params(tmp_path, "on", directory, score, parameters, "--gm")
    assert np.array_equal(on["f0"], off["f0"])
    assert not np.array_equal(on["sp"], off["sp"])


def test_gm_cli_strength_zero_equals_off(trained_model, tmp_path):
    directory, score, parameters = _write_cli_inputs(tmp_path, trained_model)
    off = _cli_params(tmp_path, "off", directory, score, parameters, "--no-gm")
    zero = _cli_params(tmp_path, "zero", directory, score, parameters,
                       "--gm", "--gm-strength", "0")
    assert np.array_equal(off["sp"], zero["sp"])
    assert np.array_equal(off["ap"], zero["ap"])


def test_gm_cli_yaml_and_flag_overrides(trained_model, tmp_path):
    directory, score, parameters = _write_cli_inputs(tmp_path, trained_model)
    parameters.write_text("synthesis:\n  vocoder: builtin\n  seed: 0\n"
                          "  gm_enabled: true\n  gm_strength: 1.0\n")
    on = _cli_params(tmp_path, "on", directory, score, parameters)
    off = _cli_params(tmp_path, "off", directory, score, parameters, "--no-gm")
    assert np.array_equal(on["f0"], off["f0"])
    assert not np.array_equal(on["sp"], off["sp"])


def test_gm_cli_requires_model_statistics(trained_model, tmp_path, capsys):
    legacy = copy.deepcopy(trained_model)
    legacy.gv_stats = None
    legacy_dir = legacy.save(tmp_path / "legacy")
    score = tmp_path / "score.tsv"
    score.write_text("song\t0.0\t0.1\tsil\t-\n"
                     "song\t0.1\t0.8\ta\t60\n"
                     "song\t0.8\t0.9\tsil\t-\n")
    parameters = tmp_path / "params.yaml"
    parameters.write_text("synthesis:\n  vocoder: builtin\n  seed: 0\n"
                          "  gm_enabled: true\n")
    assert main(["synth", "--model", str(legacy_dir), "--score", str(score),
                 "--out", str(tmp_path / "legacy.wav"), "--config",
                 str(parameters)]) == 2
    assert "no global_variance" in capsys.readouterr().err
