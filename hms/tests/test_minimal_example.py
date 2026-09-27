"""Run the minimal example's actual documented commands, without native WORLD."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

from hms.core.model import HMSModel
from hms.data import wavio


ROOT = Path(__file__).resolve().parents[2]


def test_minimal_example(tmp_path):
    example = tmp_path / "examples" / "minimal"
    shutil.copytree(ROOT / "examples" / "minimal", example,
                    ignore=shutil.ignore_patterns("out", "__pycache__"))

    # The single `sh` block is the runnable workflow, not a second set of
    # commands duplicated in the test that could drift away from the docs.
    readme = (example / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```sh\n(.*?)\n```", readme, flags=re.DOTALL)
    assert len(blocks) == 1
    commands = [shlex.split(line) for line in
                blocks[0].replace("\\\n", "").splitlines() if line.strip()]
    assert len(commands) == 3
    env = dict(os.environ, HMS_NO_AUTO_BUILD="1")
    # Support tests run from a source checkout as well as an editable install.
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    def run(command):
        assert command[0] == "python"
        result = subprocess.run(
            [sys.executable, *command[1:]], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run(commands[0])
    corpus_wav = example / "out" / "wav" / "training_a.wav"
    original = corpus_wav.read_bytes()
    run(commands[0])
    assert corpus_wav.read_bytes() == original
    corpus_info = wavio.audio_info(corpus_wav)
    assert corpus_info["duration"] == pytest.approx(0.8)
    assert corpus_info["sample_rate"] == 22050

    run(commands[1])
    model_dir = example / "out" / "model"
    for name in ("model.yaml", "hmm.npz", "backoff.npz"):
        assert (model_dir / name).stat().st_size > 0
    model = HMSModel.load(model_dir)
    assert model.name == "minimal"
    assert set(model.hmms) == {"a", "sil"}
    assert model.stats.utterances == 1
    assert model.metadata["training_config"]["vocoder"] == "builtin"

    # The synth subprocess must load the model from disk, not reuse the
    # in-memory training result. No training audio should be needed anymore.
    shutil.rmtree(example / "out" / "wav")
    output = run(commands[2])
    assert "backend : builtin" in output
    wav = example / "out" / "song.wav"
    info = wavio.audio_info(wav)
    assert info["sample_rate"] == 22050
    assert info["channels"] == 1
    assert info["sample_width"] == 2
    assert info["duration"] == pytest.approx(1.2, abs=0.01)
    audio, _ = wavio.read_wav(wav)
    assert np.isfinite(audio).all()
    # Both requested vowel regions must contain actual audio, not just a WAV
    # header or silence. Avoid judging subjective quality in a smoke test.
    for start, end in ((0.1, 0.55), (0.65, 1.1)):
        assert np.max(np.abs(audio[int(start * 22050):int(end * 22050)])) > 0.01
