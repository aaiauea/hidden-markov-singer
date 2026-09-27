"""Shared fixtures.

The expensive pieces (rendering the demo corpus, training a model on it) are
session scoped, so the whole suite stays in the tens of seconds and remains
usable as a pre-commit check.  A reduced sample rate and FFT size are used
throughout: every code path is the same as at 44.1 kHz, it is just faster.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                     # `pytest` from a checkout
    sys.path.insert(0, str(ROOT))

from hms.core import labels as labels_module  # noqa: E402
from hms.core.phonemes import PhonemeSet  # noqa: E402
from hms.core.trainer import Trainer, TrainingConfig  # noqa: E402
from hms.data.demo_singer import SingerConfig, make_dataset  # noqa: E402

#: Sample rate used by the fixtures (half the default -> ~half the runtime).
TEST_FS = 22050
#: FFT size used by the fixtures.
TEST_FFT = 1024


@pytest.fixture(scope="session")
def phoneme_set() -> PhonemeSet:
    return PhonemeSet.default()


@pytest.fixture(scope="session")
def demo_dataset(tmp_path_factory) -> dict:
    """The example corpus, rendered at `TEST_FS`.

    Training labels are deliberately jittered by 8 ms so the alignment code is
    exercised on imperfect boundaries; ``score.tsv`` stays exact.
    """
    directory = tmp_path_factory.mktemp("demo_corpus")
    return make_dataset(directory, fs=TEST_FS,
                        singer=SingerConfig(fs=TEST_FS, seed=3),
                        label_jitter_ms=8.0, seed=3)


@pytest.fixture(scope="session")
def trained_model(demo_dataset, phoneme_set):
    """A model trained on the demo corpus with a deliberately small budget."""
    config = TrainingConfig(
        label_file=demo_dataset["labels"], wav_dir=demo_dataset["wav_dir"],
        fs=TEST_FS, fft_size=TEST_FFT, n_mcep=20, n_band=5, use_delta=True,
        n_iterations=2, min_phoneme_frames=10, seed=0)
    return Trainer(config, phoneme_set).train()


@pytest.fixture(scope="session")
def trained_context_model(demo_dataset, phoneme_set):
    """A model trained with sparse phoneme-context modelling enabled.

    Same corpus and acoustic budget as `trained_model`, plus context HMMs for
    the observed phone contexts and the optional global backoff -- used to
    exercise the context selection, serialisation and evaluation paths.
    """
    config = TrainingConfig(
        label_file=demo_dataset["labels"], wav_dir=demo_dataset["wav_dir"],
        fs=TEST_FS, fft_size=TEST_FFT, n_mcep=20, n_band=5, use_delta=True,
        n_iterations=1, min_phoneme_frames=10, seed=0,
        context_enabled=True, context_min_frames=100,
        context_min_occurrences=3, context_max_models=24,
        context_global_backoff=True)
    return Trainer(config, phoneme_set).train()


@pytest.fixture(scope="session")
def score(demo_dataset):
    """The whole demo score (18 phrases, ~32 s of audio)."""
    return labels_module.load(demo_dataset["score"])


@pytest.fixture(scope="session")
def short_score(score):
    """The first three phrases: same code paths, ~5x faster to render.

    Tests that do not care about the corpus as a whole use this, so the suite
    stays quick enough to run on every commit.
    """
    return labels_module.Score(score.utterances[:3])


@pytest.fixture(scope="session")
def example_wav_path(demo_dataset) -> Path:
    return sorted(Path(demo_dataset["wav_dir"]).glob("*.wav"))[0]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
