"""Experimental GV: statistics, gradient, MLPG geometry, safety, compatibility."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from hms.cli.main import main
from hms.core.features import add_dynamic_features
from hms.core.generation import mlpg, stack_streams, window_matrix
from hms.core.gv import (GlobalVarianceStats, _StaticWindows,
                         estimate_global_variance, optimize_global_variance,
                         trajectory_variance, variance_gradient)
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


def test_global_variance_uses_each_trajectories_own_mean():
    series = np.array([2.0, 4.0, 8.0, 10.0])
    assert trajectory_variance(series) == pytest.approx(10.0)
    c = np.column_stack([series, series * 0.5 - 30.0])
    assert np.allclose(trajectory_variance(c), [10.0, 2.5])
    assert np.array_equal(trajectory_variance(c + [1000.0, -200.0]),
                          trajectory_variance(c))
    assert trajectory_variance(np.array([3.0])) == 0.0
    assert trajectory_variance(np.zeros((0, 2))).tolist() == [0.0, 0.0]


def test_variance_gradient_matches_formula_and_finite_difference():
    c = np.array([[1.0, 50.0], [3.0, 52.0], [7.0, 54.0], [9.0, 70.0]])
    gradient = variance_gradient(c)
    assert np.allclose(gradient, 2 * (c - c.mean(axis=0)) / len(c))
    assert np.allclose(gradient.sum(axis=0), 0.0, atol=1e-14)
    eps = 1e-5
    for t in range(len(c)):
        for d in range(c.shape[1]):
            positive, negative = c.copy(), c.copy()
            positive[t, d] += eps
            negative[t, d] -= eps
            numeric = (trajectory_variance(positive)[d]
                       - trajectory_variance(negative)[d]) / (2 * eps)
            assert gradient[t, d] == pytest.approx(numeric, rel=1e-8, abs=1e-9)
    assert np.array_equal(variance_gradient(c[:, 0]), gradient[:, 0])
    assert np.array_equal(variance_gradient(np.zeros((0, 2))),
                          np.zeros((0, 2)))
    assert np.array_equal(variance_gradient(np.array([[7.0, 9.0]])),
                          np.zeros((1, 2)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_variance_and_gradient_reject_nonfinite_values(bad):
    with pytest.raises(ValueError, match="finite"):
        trajectory_variance(np.array([1.0, bad]))
    with pytest.raises(ValueError, match="finite"):
        variance_gradient(np.array([[1.0], [bad]]))


def test_training_targets_average_utterance_variances_not_corpus_variance():
    a = np.array([[0.0, -10.0, 100.0, 300.0],
                  [2.0, -10.0, 100.0, 300.0]])
    b = np.array([[100.0, 2.0, -1000.0, -400.0],
                  [102.0, 4.0, -1000.0, -400.0],
                  [104.0, 6.0, -1000.0, -400.0]])
    # The last two columns are DELTAS: large/small values there must not
    # become targets, even if the feature matrices carry more than one stream.
    stats = estimate_global_variance(iter((a, b, b[:1])), static_dim=2)
    assert stats.utterances == 2
    assert np.allclose(stats.target_variance, [0.5 * (1 + 8 / 3), 4 / 3])
    assert np.var(np.concatenate([a[:, :2], b[:, :2]])[:, 0]) > 1000
    assert estimate_global_variance([a[:1]], static_dim=2) is None
    assert estimate_global_variance([], static_dim=2) is None


@pytest.mark.parametrize("n_frames", [1, 2, 3, 13])
@pytest.mark.parametrize("n_streams", [1, 2, 3])
def test_gv_window_operator_matches_mlpg_geometry_at_boundaries(
        n_frames, n_streams):
    """W, W^T and preconditioner, including double clipping of delta-delta."""
    dim = 2
    rng = np.random.default_rng(n_frames * 10 + n_streams)
    static = rng.normal(size=(n_frames, dim))
    stacked = rng.normal(size=(n_frames * n_streams, dim))
    precision = rng.uniform(0.05, 2.0, size=stacked.shape)
    sizes = (dim,) * n_streams
    matrix = window_matrix(n_frames, sizes, window=2)
    op = _StaticWindows(n_frames, n_streams, window=2)
    assert np.allclose(op.apply(static).reshape(-1), matrix @ static.reshape(-1))
    assert np.allclose(op.transpose(stacked).reshape(-1),
                       matrix.T @ stacked.reshape(-1))
    for d in range(dim):
        w = matrix[d::dim, d::dim]
        expected = np.diag(w.T @ np.diag(precision[:, d]) @ w)
        assert np.allclose(op.diagonal(precision)[:, d], expected)
    if n_streams > 1:
        expected = stack_streams(add_dynamic_features(
            static, use_delta=True, use_delta2=n_streams == 3), sizes)
        assert np.allclose(op.apply(static), expected)


def test_gv_objective_gradient_matches_finite_difference():
    """Check BOTH the MLPG likelihood increase and the variance penalty."""
    rng = np.random.default_rng(32)
    n, dim, streams = 7, 2, 3
    initial = rng.normal(size=(n, dim))
    current = initial + rng.normal(scale=0.1, size=initial.shape)
    precisions = rng.uniform(0.3, 2, size=(n * streams, dim))
    target = np.array([0.8, 0.3])
    weight = 2.0
    scale = np.maximum.reduce((trajectory_variance(initial), target,
                               np.full(dim, 1e-6)))
    op = _StaticWindows(n, streams, window=2)

    def objective(c):
        delta = op.apply(c - initial)
        gv_error = (trajectory_variance(c) - target) / scale
        return (0.5 / n * np.sum(precisions * delta ** 2, axis=0)
                + 0.5 * weight * gv_error ** 2)

    displaced = op.apply(current - initial)
    gradient = (op.transpose(precisions * displaced) / n
                + weight * (trajectory_variance(current) - target)
                / (scale ** 2) * variance_gradient(current))
    eps = 1e-6
    for t in range(n):
        for d in range(dim):
            plus, minus = current.copy(), current.copy()
            plus[t, d] += eps
            minus[t, d] -= eps
            numeric = (objective(plus)[d] - objective(minus)[d]) / (2 * eps)
            assert gradient[t, d] == pytest.approx(numeric, abs=1e-9)


def test_oversmoothed_trajectory_moves_toward_target_not_just_rescaled():
    initial, variances, sizes, source_means = _step_problem(dim=1)
    baseline = trajectory_variance(initial)[0]
    assert baseline < trajectory_variance(source_means)[0]  # genuine smoothing
    target = np.array([2.0 * baseline])
    optimized, steps = optimize_global_variance(
        initial, variances, sizes, target, weight=3.0, iterations=20,
        return_steps=True)
    assert 1 <= steps <= 20
    after = trajectory_variance(optimized)[0]
    assert baseline < after < target[0]
    assert np.isfinite(optimized).all()
    assert np.max(np.abs(optimized - initial)) < 1.0
    # With W^T P W as the anchor, the update is NOT a uniform multiplier on
    # (c - mean(c)): local acoustic/dynamic constraints matter.
    a = initial[:, 0] - initial[:, 0].mean()
    b = optimized[:, 0] - optimized[:, 0].mean()
    multiplier = np.dot(a, b) / np.dot(a, a)
    assert np.max(np.abs(b - multiplier * a)) > 1e-3


@pytest.mark.parametrize("target_factor,direction", [(2.0, 1), (0.25, -1)])
def test_gv_increases_and_decreases_as_requested(target_factor, direction):
    initial, variances, sizes, _ = _step_problem(dim=1, delta2=True)
    start = trajectory_variance(initial)[0]
    target = np.array([target_factor * start])
    result = optimize_global_variance(initial, variances, sizes, target,
                                      weight=8.0, iterations=24)
    actual = trajectory_variance(result)[0]
    assert (actual - start) * direction > 0.02 * start
    assert abs(actual - target[0]) < abs(start - target[0])


def test_multiple_static_features_have_independent_targets():
    initial, variances, sizes, _ = _step_problem(dim=3)
    before = trajectory_variance(initial)
    target = before * [2.0, 0.25, 1.0]
    result = optimize_global_variance(initial, variances, sizes, target,
                                      weight=5.0, iterations=20)
    after = trajectory_variance(result)
    assert after[0] > before[0]
    assert after[1] < before[1]
    assert np.array_equal(result[:, 2], initial[:, 2])
    # Changing only the first target must not change the second trajectory.
    other = optimize_global_variance(initial, variances, sizes,
                                     before * [3.0, 0.25, 1.0],
                                     weight=5.0, iterations=20)
    assert np.array_equal(result[:, 1], other[:, 1])


def test_accepted_updates_decrease_the_full_objective_without_mutating_inputs():
    initial, variances, sizes, _ = _step_problem(dim=2)
    n = len(initial)
    variances[:n, 0] *= np.linspace(0.5, 2.0, n)
    saved_initial, saved_variances = initial.copy(), variances.copy()
    target = trajectory_variance(initial) * [1.8, 0.4]
    scale = np.maximum.reduce((trajectory_variance(initial), target,
                               np.full(2, 1e-6)))
    weight = 4.0
    result = optimize_global_variance(initial, variances, sizes, target,
                                      variance_scale=2.0, weight=weight,
                                      iterations=20)
    precision = 1 / np.maximum(variances, 1e-8)
    precision[n:] /= 2.0
    displacement = _StaticWindows(n, len(sizes), 2).apply(result - initial)
    before = 0.5 * weight * ((trajectory_variance(initial) - target) / scale) ** 2
    after = (0.5 / n * np.sum(precision * displacement ** 2, axis=0)
             + 0.5 * weight * ((trajectory_variance(result) - target) / scale) ** 2)
    assert (after < before).all()
    assert np.array_equal(initial, saved_initial)
    assert np.array_equal(variances, saved_variances)


def test_zero_variance_target_and_constant_input_are_safe():
    initial, variances, sizes, _ = _step_problem(dim=1)
    result = optimize_global_variance(initial, variances, sizes, np.array([0.0]),
                                      weight=3.0, iterations=24)
    assert np.isfinite(result).all()
    assert 0 < trajectory_variance(result)[0] < trajectory_variance(initial)[0]
    constant = np.full((25, 1), 4.0)
    for target in (0.0, 2.0):
        out = optimize_global_variance(constant, np.ones((50, 1)), sizes,
                                       np.array([target]), iterations=20)
        assert np.array_equal(out, constant)  # no arbitrary invented noise


def test_zero_model_variances_are_floored_and_extreme_weight_is_bounded():
    initial, variances, sizes, _ = _step_problem(dim=1)
    target = 2 * trajectory_variance(initial)
    floored = optimize_global_variance(initial, np.zeros_like(variances), sizes,
                                       target, weight=5.0, iterations=10)
    assert np.isfinite(floored).all()
    assert np.max(np.abs(floored - initial)) < 1.0
    extreme = optimize_global_variance(initial, variances, sizes, target,
                                       weight=1e10, iterations=40)
    assert np.isfinite(extreme).all()
    assert 0 < trajectory_variance(extreme)[0] <= 2 * target[0]


def test_short_sequences_and_zero_strength_leave_trajectory_unchanged():
    for n in (0, 1, 2):
        initial = np.arange(n, dtype=float)[:, None]
        variances = np.ones((2 * n, 1))
        result = optimize_global_variance(initial, variances, (1, 1),
                                          np.array([0.0]), iterations=20)
        assert np.isfinite(result).all()
        if n < 2:
            assert np.array_equal(result, initial)
            unchanged, steps = optimize_global_variance(
                initial, variances, (1, 1), np.array([1.0]),
                return_steps=True)
            assert steps == 0 and np.array_equal(unchanged, initial)
        else:
            assert trajectory_variance(result)[0] < trajectory_variance(initial)[0]
        for weight, iterations in ((0.0, 20), (1.0, 0)):
            unchanged = optimize_global_variance(
                initial, variances, (1, 1), np.array([1.0]),
                weight=weight, iterations=iterations)
            assert np.array_equal(unchanged, initial)


@pytest.mark.parametrize("change,match", [
    ({"trajectory": np.array([[np.nan], [2.0]])}, "trajectory"),
    ({"trajectory": np.array([[np.inf], [2.0]])}, "trajectory"),
    ({"variances": np.array([[1.0], [1.0], [np.inf], [1.0]])}, "variances"),
    ({"variances": np.ones((3, 1))}, "variances"),
    ({"target_variance": np.array([np.nan])}, "target"),
    ({"target_variance": np.array([np.inf])}, "target"),
    ({"target_variance": np.array([-1.0])}, "target"),
    ({"target_variance": np.array([1.0, 2.0])}, "target"),
    ({"weight": -1.0}, "weight"),
    ({"weight": np.inf}, "weight"),
    ({"iterations": -1}, "iterations"),
    ({"iterations": 1.5}, "iterations"),
    ({"variance_scale": 0.0}, "variance_scale"),
    ({"window": 0}, "window"),
    ({"stream_sizes": (2, 2)}, "stream sizes"),
])
def test_optimizer_rejects_invalid_inputs(change, match):
    kwargs = dict(trajectory=np.array([[1.0], [2.0]]),
                  variances=np.ones((4, 1)), stream_sizes=(1, 1),
                  target_variance=np.array([1.0]))
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        optimize_global_variance(**kwargs)


def test_training_populates_only_static_targets(trained_model):
    spec = trained_model.spec
    assert spec.stream_sizes == (spec.static_dim, spec.static_dim)
    assert trained_model.gv_stats is not None
    assert trained_model.gv_stats.target_variance.shape == (spec.static_dim,)
    assert trained_model.gv_stats.utterances == trained_model.stats.utterances
    assert np.isfinite(trained_model.gv_stats.target_variance).all()
    assert (trained_model.gv_stats.target_variance >= 0).all()


def test_optional_gv_section_roundtrips_without_format_bump(trained_model,
                                                               tmp_path):
    directory = trained_model.save(tmp_path / "with-gv")
    document = yaml.safe_load((directory / "model.yaml").read_text())
    assert document["format_version"] == MODEL_FORMAT_VERSION
    assert document["global_variance"] == trained_model.gv_stats.to_dict()
    loaded = HMSModel.load(directory)
    assert loaded.loaded_format_version == MODEL_FORMAT_VERSION
    assert loaded.gv_stats.utterances == trained_model.gv_stats.utterances
    assert np.array_equal(loaded.gv_stats.target_variance,
                          trained_model.gv_stats.target_variance)
    assert "GV targets" in loaded.summary()

    # A corrupt target length must be rejected, not silently applied to the
    # wrong static coefficients or (worse) the delta streams.
    document["global_variance"]["target_variance"].append(1.0)
    (directory / "model.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="static_dim"):
        HMSModel.load(directory)


@pytest.mark.parametrize("bad", [[-1.0], [float("nan")], [float("inf")], []])
def test_bad_gv_statistics_are_rejected(bad):
    with pytest.raises(ValueError, match="GV targets"):
        GlobalVarianceStats(bad, utterances=1)


def test_old_models_have_no_section_and_only_synthesize_without_gv(
        trained_model, short_score, tmp_path):
    old = copy.deepcopy(trained_model)
    old.gv_stats = None
    directory = old.save(tmp_path / "legacy")
    document = yaml.safe_load((directory / "model.yaml").read_text())
    assert "global_variance" not in document
    loaded = HMSModel.load(directory)
    assert loaded.gv_stats is None
    small = Score(short_score.utterances[:1])
    default = Synthesizer(loaded, SynthesisConfig(seed=0, vibrato=False,
                                                  vocoder="builtin")).synthesize(small)
    assert np.isfinite(default.audio).all()
    with pytest.raises(ValueError, match="no global_variance"):
        Synthesizer(loaded, SynthesisConfig(gv_enabled=True, vocoder="builtin")) \
            .synthesize(small)
    # Even a format-2 model (no context/pitch/GV section) keeps loading.
    document["format_version"] = 2
    (directory / "model.yaml").write_text(yaml.safe_dump(document))
    assert HMSModel.load(directory).gv_stats is None


def test_gv_disabled_is_bit_identical_to_existing_generation_path(
        trained_model, short_score, monkeypatch):
    small = Score(short_score.utterances[:1])
    config = dict(seed=0, vibrato=False, vocoder="builtin")
    default = Synthesizer(trained_model, SynthesisConfig(**config)).synthesize(small)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("GV optimizer ran when disabled")

    monkeypatch.setattr("hms.core.synthesizer.optimize_global_variance",
                        must_not_run)
    disabled = Synthesizer(trained_model, SynthesisConfig(
        **config, gv_enabled=False, gv_weight=50, gv_iterations=40)).synthesize(small)
    assert np.array_equal(default.audio, disabled.audio)
    assert np.array_equal(default.params.f0, disabled.params.f0)
    assert np.array_equal(default.params.sp, disabled.params.sp)
    assert np.array_equal(default.params.ap, disabled.params.ap)


def test_end_to_end_generation_with_gv_changes_spectral_static_features(
        trained_model, short_score):
    small = Score(short_score.utterances[:1])
    config = dict(seed=0, vibrato=False, vocoder="builtin")
    baseline = Synthesizer(trained_model, SynthesisConfig(**config)).synthesize(small)
    enabled = Synthesizer(trained_model, SynthesisConfig(
        **config, gv_enabled=True, gv_weight=6.0, gv_iterations=16)).synthesize(small)
    assert len(enabled.audio) == len(baseline.audio)
    assert np.isfinite(enabled.audio).all()
    assert np.isfinite(enabled.params.sp).all()
    assert np.isfinite(enabled.params.ap).all()
    assert np.max(np.abs(enabled.params.sp - baseline.params.sp)) > 1e-6
    # The score, not the note-relative static feature, supplies default F0.
    assert np.array_equal(enabled.params.f0, baseline.params.f0)


def test_gv_cli_yaml_and_flag_overrides(trained_model, tmp_path, capsys):
    directory = trained_model.save(tmp_path / "model")
    score = tmp_path / "score.tsv"
    score.write_text("song\t0.0\t0.1\tsil\t-\n"
                     "song\t0.1\t0.8\ta\t60\n"
                     "song\t0.8\t0.9\tsil\t-\n")
    parameters = tmp_path / "params.yaml"
    parameters.write_text("synthesis:\n  vocoder: builtin\n  seed: 0\n"
                          "  gv_enabled: true\n  gv_weight: 6.0\n"
                          "  gv_iterations: 12\n")

    def cli_params(name, *overrides):
        params = tmp_path / f"{name}.npz"
        args = ["synth", "--model", str(directory), "--score", str(score),
                "--config", str(parameters), "--save-params", str(params),
                "--out", str(tmp_path / f"{name}.wav"), *overrides]
        assert main(args) == 0
        with np.load(params) as arrays:
            return {key: arrays[key] for key in ("f0", "sp", "ap")}

    on = cli_params("on")             # config YAML enables GV
    off = cli_params("off", "--no-gv")  # CLI flag takes precedence
    zero = cli_params("zero", "--gv", "--gv-weight", "0")
    assert np.array_equal(on["f0"], off["f0"])
    assert not np.array_equal(on["sp"], off["sp"])
    assert np.array_equal(off["sp"], zero["sp"])
    assert np.array_equal(off["ap"], zero["ap"])
    capsys.readouterr()
    assert main(["inspect-model", "--model", str(directory), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["global_variance"] == trained_model.gv_stats.to_dict()

    legacy = copy.deepcopy(trained_model)
    legacy.gv_stats = None
    legacy_dir = legacy.save(tmp_path / "legacy")
    assert main(["synth", "--model", str(legacy_dir), "--score", str(score),
                 "--out", str(tmp_path / "legacy.wav"), "--config",
                 str(parameters), "--gv"]) == 2
    assert "no global_variance" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [
    {"gv_weight": -1.0}, {"gv_weight": float("inf")},
    {"gv_iterations": -1}, {"gv_iterations": 1.5},
    {"gv_enabled": "true"},
])
def test_gv_config_rejects_bad_settings(bad):
    with pytest.raises(ValueError, match="gv_|synthesis settings"):
        SynthesisConfig(**bad)
