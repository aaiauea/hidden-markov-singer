"""Objective model evaluation on an evaluation corpus (`hms evaluate`).

No aggregate quality score: models are compared with separate objective
metrics (held-out log-likelihood, voicing agreement, duration error, backoff
usage), and the comparison itself is guarded by compatibility checks --
feature spec (hard), inventory, training method, seed, corpus paths and every
other non-context training setting (warnings).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hms.cli.main import main
from hms.core.evaluate import check_model_compatibility, evaluate_models
from hms.core.model import HMSModel


def write_subset_labels(demo_dataset, path: Path,
                        names=("vowel_scale_a", "syllables_ma")) -> Path:
    lines = ["# utt_id\tonset\toffset\tphone\tnote"]
    for line in Path(demo_dataset["labels"]).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split()[0] in names:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# compatibility checks
# --------------------------------------------------------------------------


def test_identical_models_are_fully_compatible(trained_model):
    errors, warnings = check_model_compatibility(
        [trained_model, trained_model])
    assert errors == [] and warnings == []


def test_feature_spec_mismatch_is_an_error(trained_model):
    other = copy.deepcopy(trained_model)
    other.name = "other"
    other.spec.n_mcep = trained_model.spec.n_mcep + 1
    errors, _warnings = check_model_compatibility([trained_model, other])
    assert len(errors) == 1 and "feature spec" in errors[0]
    with pytest.raises(ValueError, match="feature spec"):
        evaluate_models([trained_model, other], "labels.tsv", ".")


def test_seed_method_paths_and_inventory_mismatches_are_warnings(
        trained_model):
    other = copy.deepcopy(trained_model)
    other.name = "other"
    config = other.metadata["training_config"]
    config["seed"] = int(config.get("seed", 0)) + 1
    config["training_method"] = ("baum_welch"
                                 if config.get("training_method") == "viterbi"
                                 else "viterbi")
    config["label_file"] = "/elsewhere/labels.tsv"
    other.phoneme_set.phonemes.pop(other.phoneme_set.symbols[-1])

    errors, warnings = check_model_compatibility([trained_model, other])
    assert errors == []
    joined = "\n".join(warnings)
    assert "seed" in joined
    assert "training method" in joined
    assert "label_file" in joined
    assert "inventory" in joined


def test_context_settings_do_not_warn(trained_model):
    """Context on/off is what evaluation compares -- it must not warn."""
    other = copy.deepcopy(trained_model)
    other.name = "other"
    other.metadata["training_config"]["context_enabled"] = True
    errors, warnings = check_model_compatibility([trained_model, other])
    assert errors == [] and warnings == []


# --------------------------------------------------------------------------
# the evaluation itself
# --------------------------------------------------------------------------


def test_evaluate_models_reports_separate_objective_metrics(
        demo_dataset, trained_model, tmp_path):
    labels = write_subset_labels(demo_dataset, tmp_path / "eval.tsv")
    report = evaluate_models([trained_model], labels,
                             demo_dataset["wav_dir"])

    assert report["corpus"]["utterances"] == 2
    assert report["corpus"]["frames"] > 0
    (metrics,) = report["models"].values()
    for key in ("frames", "total_log_likelihood", "log_likelihood_per_frame",
                "voicing_agreement", "duration_mae_frames",
                "backoff_frames", "per_phone_log_likelihood_per_frame"):
        assert key in metrics
    assert np.isfinite(metrics["total_log_likelihood"])
    assert 0.0 <= metrics["voicing_agreement"] <= 1.0
    assert metrics["duration_mae_frames"] >= 0.0
    assert metrics["per_phone_log_likelihood_per_frame"], "per-phone table"
    # objective comparison, not a single fused score: the report carries the
    # metrics separately and has no aggregate quality field
    assert "quality_score" not in metrics and "score" not in metrics


def test_evaluate_models_is_deterministic(demo_dataset, trained_model,
                                          tmp_path):
    labels = write_subset_labels(demo_dataset, tmp_path / "eval.tsv")
    first = evaluate_models([trained_model], labels, demo_dataset["wav_dir"])
    second = evaluate_models([trained_model], labels, demo_dataset["wav_dir"])
    assert first == second


def test_evaluate_models_compares_baseline_and_context_models(
        demo_dataset, trained_model, trained_context_model, tmp_path):
    labels = write_subset_labels(demo_dataset, tmp_path / "eval.tsv")
    baseline = copy.deepcopy(trained_model)
    baseline.name = "baseline"
    context = copy.deepcopy(trained_context_model)
    context.name = "context"
    report = evaluate_models([baseline, context], labels,
                             demo_dataset["wav_dir"])
    assert set(report["models"]) == {"baseline", "context"}
    for metrics in report["models"].values():
        assert metrics["frames"] > 0
        assert np.isfinite(metrics["log_likelihood_per_frame"])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_evaluate_smoke(tmp_path, demo_dataset, trained_model, capsys):
    labels = write_subset_labels(demo_dataset, tmp_path / "eval.tsv")
    model_dir = tmp_path / "model"
    trained_model.save(model_dir)
    out_json = tmp_path / "report.json"

    capsys.readouterr()
    assert main(["evaluate", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--model", str(model_dir), "--json", str(out_json),
                 "--vocoder", "builtin"]) == 0
    text = capsys.readouterr().out
    assert "LL/frame" in text and "voicing" in text
    assert "no aggregate" in text

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["corpus"]["utterances"] == 2
    assert len(payload["models"]) == 1
    assert "warnings" in payload


def test_cli_evaluate_compares_two_models(tmp_path, demo_dataset,
                                          trained_model,
                                          trained_context_model, capsys):
    labels = write_subset_labels(demo_dataset, tmp_path / "eval.tsv")
    baseline_dir = tmp_path / "baseline"
    context_dir = tmp_path / "context"
    baseline = copy.deepcopy(trained_model)
    baseline.name = "baseline"
    baseline.save(baseline_dir)
    context = copy.deepcopy(trained_context_model)
    context.name = "context"
    context.save(context_dir)

    capsys.readouterr()
    assert main(["evaluate", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--model", str(baseline_dir),
                 "--model", str(context_dir),
                 "--vocoder", "builtin"]) == 0
    text = capsys.readouterr().out
    assert baseline_dir.name in text and context_dir.name in text


def test_cli_evaluate_refuses_mismatched_feature_specs(
        tmp_path, demo_dataset, trained_model, capsys):
    labels = write_subset_labels(demo_dataset, tmp_path / "eval.tsv")
    baseline_dir = tmp_path / "baseline"
    trained_model.save(baseline_dir)
    other = copy.deepcopy(trained_model)
    other.name = "other"
    other.spec.n_mcep = trained_model.spec.n_mcep + 1
    other_dir = tmp_path / "other"
    other.save(other_dir)

    capsys.readouterr()
    code = main(["evaluate", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--model", str(baseline_dir), "--model", str(other_dir)])
    assert code == 2
    assert "feature spec" in capsys.readouterr().err
