"""The CLI: one thin shell over the library, so it gets one smoke suite.

Everything here could be done in Python instead -- that is the point of the
design -- but the CLI is the documented entry point, so it must actually run:
`doctor`, `extract`, `train`, `inspect-model` and `synth`, plus the error path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hms.cli.main import main
from hms.data import wavio

F_REF = 261.6255653005986


def write_subset_labels(demo_dataset, path: Path, utterance: str = "vowel_scale_a",
                        names=("vowel_scale_a",)):
    """A small label file covering a subset of the demo corpus."""
    lines = ["# utt_id\tonset\toffset\tphone\tnote"]
    for line in Path(demo_dataset["labels"]).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split()[0] in names:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0


def test_doctor_reports_backends(capsys):
    assert main(["doctor"]) == 0
    text = capsys.readouterr().out
    assert "vocoder backends" in text
    assert "native" in text and "builtin" in text


def test_extract_train_inspect_synth(tmp_path, demo_dataset, capsys):
    """The whole documented workflow, on a one-utterance subset."""
    labels = write_subset_labels(demo_dataset, tmp_path / "labels.tsv")

    # 1. extract: WORLD parameters land on disk
    params_dir = tmp_path / "params"
    assert main(["extract", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--out", str(params_dir), "--fs", "22050",
                 "--features"]) == 0
    assert (params_dir / "vowel_scale_a.npz").exists()
    assert (params_dir / "vowel_scale_a.features.npy").exists()
    summary = json.loads((params_dir / "extract_summary.json").read_text())
    assert summary["vowel_scale_a"]["frames"] > 10
    assert 0.0 < summary["vowel_scale_a"]["voiced_fraction"] <= 1.0
    assert summary["vowel_scale_a"]["f0_median_hz"] > 0

    # 2. train: a small but complete model directory
    model_dir = tmp_path / "model"
    assert main(["train", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--out", str(model_dir), "--fs", "22050",
                 "--iterations", "1", "--name", "cli-smoke"]) == 0
    assert (model_dir / "model.yaml").exists()
    assert (model_dir / "hmm.npz").exists()

    # 3. inspect-model: human readable, and machine readable on request
    capsys.readouterr()
    assert main(["inspect-model", "--model", str(model_dir)]) == 0
    text = capsys.readouterr().out
    assert "cli-smoke" in text
    assert "phonemes modelled" in text
    assert main(["inspect-model", "--model", str(model_dir),
                 "--phoneme", "a"]) == 0
    per_phoneme = capsys.readouterr().out
    assert "a" in per_phoneme
    assert main(["inspect-model", "--model", str(model_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "cli-smoke"
    assert payload["parameter_budget"] > 0

    # 4. synth: a rendered WAV of the requested length
    score = tmp_path / "score.tsv"
    score.write_text("demo\t0.0\t0.1\tsil\t-\n"
                     "demo\t0.1\t0.7\ta\t60\n"
                     "demo\t0.7\t0.8\tsil\t-\n", encoding="utf-8")
    out_wav = tmp_path / "out.wav"
    trace = tmp_path / "trace.tsv"
    params = tmp_path / "synth_params.npz"
    assert main(["synth", "--model", str(model_dir), "--score", str(score),
                 "--out", str(out_wav), "--trace", str(trace),
                 "--save-params", str(params), "--seed", "0"]) == 0
    info = wavio.audio_info(out_wav)
    assert info["duration"] == pytest.approx(0.8, abs=0.05)
    signal, fs = wavio.read_wav(out_wav)
    assert fs == 22050
    assert np.abs(signal).max() > 0.01
    assert trace.exists() and params.exists()

    # the rendered pitch must follow the requested note
    sequence = wavio.load_params(params)
    voiced = sequence.f0 > 0
    assert voiced.any()
    assert np.median(sequence.f0[voiced]) == pytest.approx(
        261.6255653, rel=0.05)

    # the trace is a frame table, as documented
    header = trace.read_text().splitlines()[0].split("\t")
    assert header[:5] == ["frame", "time_s", "phone", "state", "note"]


def test_synth_unknown_utterance_is_reported(tmp_path, demo_dataset, capsys):
    labels = write_subset_labels(demo_dataset, tmp_path / "labels.tsv")
    model_dir = tmp_path / "model"
    assert main(["train", "--labels", str(labels),
                 "--wav-dir", demo_dataset["wav_dir"],
                 "--out", str(model_dir), "--fs", "22050",
                 "--iterations", "1"]) == 0
    capsys.readouterr()
    code = main(["synth", "--model", str(model_dir),
                 "--score", demo_dataset["score"],
                 "--out", str(tmp_path / "out.wav"),
                 "--utterance", "no-such-utterance"])
    assert code == 2
    assert "no-such-utterance" in capsys.readouterr().out


def test_missing_model_is_a_clean_error(tmp_path, capsys):
    code = main(["inspect-model", "--model", str(tmp_path / "nope")])
    assert code == 2
    assert "does not look like an HMS model" in capsys.readouterr().err


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit) as info:
        main(["definitely-not-a-command"])
    assert info.value.code != 0
