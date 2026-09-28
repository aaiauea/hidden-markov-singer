"""Experimental cross-language voice transfer.

Everything here runs on a *synthetic two-voice, two-language corpus*: voice 1,
whose corpus only ever sings language A, and voice 2, whose corpus sings
language B and contains phones voice 1 has never seen.  Training voice 1 with
voice 2 attached must give a model that can sing language B's phone inventory
without giving up its own units.

The tests cover the configuration surface (the feature is off unless asked
for), the arithmetic of the acoustic map and the MAP blend, the division of
labour between the two corpora, phones the target corpus cannot sing at all,
the pitch-conditioned tier, serialisation (including format-4 compatibility),
the fallback hierarchy, the evaluation diagnostic, determinism, memory safety
and the documented `hms train --transfer-*` interface.

Nothing here claims the transferred voice *sounds like* the target voice; it
asserts that the mechanism is deterministic, that no native unit is ever
touched, that a phone only the auxiliary voice has becomes singable, and that
everything neither voice can cover still falls back instead of failing.
"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import yaml

from hms.cli.main import main
from hms.config import load_parameters, training_config_from_parameters
from hms.core import labels as labels_module
from hms.core.duration import DurationModel, DurationStats
from hms.core.evaluate import evaluate_models
from hms.core.features import FeatureSpec
from hms.core.gmm import DiagGMM
from hms.core.hmm import HMMState, LeftToRightHMM, StateDurationStats
from hms.core.labels import Score, Segment, Utterance
from hms.core.model import MODEL_FORMAT_VERSION, HMSModel
from hms.core.phonemes import PhonemeDef, PhonemeSet
from hms.core.pitch import PitchModel, PitchStats
from hms.core.pitch_condition import PitchConditioning, pitch_bin
from hms.core.synthesizer import SynthesisConfig, Synthesizer
from hms.core.trainer import Trainer, TrainingConfig, _CachedUtterance
from hms.core.transfer import (DEFAULT_MAP_ADAPT_FRAMES,
                               DEFAULT_MAP_PRIOR_STRENGTH, MAX_MEAN_SHIFT,
                               MAX_SPREAD_FACTOR, MIN_SPREAD_FACTOR,
                               MIN_TRANSFER_VARIANCE, TIER_TRANSFERRED,
                               AcousticMap,
                               CrossLanguageTransfer, VoiceSource,
                               accumulate_phone_moments, build_transfer,
                               fit_acoustic_map, map_correction, map_hmm,
                               phone_moments_nbytes, pooled_mean)
from hms.data.demo_singer import (DemoSinger, SegmentSpec, SingerConfig,
                                  make_dataset)
from hms.vocoder.base import AcousticFrameSequence

TEST_FS = 22050
TEST_FFT = 1024
#: Every phone needs this many frames before it earns its own HMM; the corpus
#: below is built so that it splits the two voices' inventories cleanly.
MIN_PHONEME_FRAMES = 40
SEED = 0
BIN_SIZE = 6

#: Language B's extra phones: voice 2 sings them, language A's corpus cannot.
LANGUAGE_B_EXTRA = {
    "y": ((300, 1700, 2200, 3300), (70, 90, 140, 180)),
    "oe": ((450, 1500, 2300, 3300), (80, 100, 140, 180)),
    "x": {"class": "unvoiced_fricative",
          "formants": ((400, 1200, 2200, 3300), (200, 300, 400, 400)),
          "noise": 1.0, "noise_band": (1000.0, 5000.0)},
}

#: Notes chosen so that the two languages' pitch bins interleave: A sings bins
#: 9/10, B sings 9/10/11 (bin size 6), which is what makes the last bucket of
#: the auxiliary voice a genuinely transferred one.
LANGUAGE_A_NOTES = (55.0, 60.0, 65.0)
LANGUAGE_B_NOTES = (57.0, 62.0, 67.0)


# --------------------------------------------------------------------------
# the synthetic corpus
# --------------------------------------------------------------------------


def transfer_phoneme_set() -> PhonemeSet:
    """The inventory both voices are trained with.

    It deliberately contains phones neither corpus sings (``n``, ``t``) and a
    phone only language B's corpus touches (``x``, below its support threshold
    so it never earns an HMM), plus the phones language B adds (``y``, ``oe``).
    """
    return PhonemeSet.from_dict({
        "defaults": {
            "vowel": {"n_states": 3, "n_components": 1, "voiced": True},
            "voiced_consonant": {"n_states": 2, "n_components": 1,
                                 "voiced": True},
            "unvoiced_consonant": {"n_states": 2, "n_components": 1,
                                   "voiced": False},
            "silence": {"n_states": 1, "n_components": 1, "voiced": False},
        },
        "phonemes": {
            "sil": {"type": "silence"},
            **{name: {"type": "vowel"} for name in ("a", "e", "i", "y", "oe")},
            "m": {"type": "voiced_consonant"},
            "n": {"type": "voiced_consonant"},
            "s": {"type": "unvoiced_consonant"},
            "t": {"type": "unvoiced_consonant"},
            "x": {"type": "unvoiced_consonant"},
        },
        "aliases": {"A": "a", "pau": "sil", "Y": "y"},
    })


class TwoLanguageSinger(DemoSinger):
    """The demo singer with a different vocal-tract length and extra phones.

    The two voices differ in *timbre* only -- a longer or shorter tract scales
    every formant by the same factor -- which is exactly the difference the
    acoustic map has to carry across when it moves language B's phones into
    voice 1's space.
    """

    def __init__(self, config: SingerConfig, *, tract_scale: float = 1.0,
                 extra=None) -> None:
        super().__init__(config)
        self.tract_scale = float(tract_scale)
        self.extra = dict(extra or {})

    def _articulation(self, phone: str) -> dict:
        if phone in self.extra:
            entry = self.extra[phone]
            spec = {"class": "vowel", "formants": entry} \
                if isinstance(entry, tuple) else dict(entry)
        else:
            spec = dict(super()._articulation(phone))
        if "formants" in spec:
            formants, bandwidths = spec["formants"]
            spec["formants"] = (
                tuple(f * self.tract_scale for f in formants),
                tuple(b * self.tract_scale for b in bandwidths))
        return spec


def _segment(phone: str, ms: float, note=None) -> SegmentSpec:
    return SegmentSpec(phone, ms, note)


def language_a_phrases():
    """Voice 1's language: shared vowels, two consonants, a glimpse of *y*."""
    phrases = []
    for vowel in ("a", "e", "i"):
        phrases.append((f"a_{vowel}", [
            _segment("sil", 70), _segment(vowel, 250, LANGUAGE_A_NOTES[0]),
            _segment(vowel, 250, LANGUAGE_A_NOTES[1]),
            _segment(vowel, 250, LANGUAGE_A_NOTES[2]),
            _segment("sil", 70)]))
    phrases.append(("a_ma", [
        _segment("sil", 60), _segment("m", 70, 58.0), _segment("a", 250, 58.0),
        _segment("m", 60, 62.0), _segment("a", 240, 62.0), _segment("sil", 70)]))
    phrases.append(("a_sa", [
        _segment("sil", 60), _segment("s", 80), _segment("a", 250, 62.0),
        _segment("s", 70), _segment("a", 240, 65.0), _segment("sil", 70)]))
    # a 30 ms glimpse of a language-B phone: below the threshold, so it earns no
    # native HMM -- but it does give the MAP blend target-side frames to use
    phrases.append(("a_y_glimpse", [
        _segment("sil", 60), _segment("y", 30, 60.0), _segment("a", 250, 60.0),
        _segment("sil", 70)]))
    return phrases


def language_b_phrases():
    """Voice 2's language: the shared vowels plus phones language A lacks."""
    phrases = []
    for vowel in ("a", "e", "i", "y", "oe"):
        phrases.append((f"b_{vowel}", [
            _segment("sil", 70), _segment(vowel, 250, LANGUAGE_B_NOTES[0]),
            _segment(vowel, 250, LANGUAGE_B_NOTES[1]),
            _segment(vowel, 250, LANGUAGE_B_NOTES[2]),
            _segment("sil", 70)]))
    # below its own support threshold: language B owns the phone but never
    # trains a unit for it, so it must not appear as a transferred unit either
    phrases.append(("b_xa", [
        _segment("sil", 60), _segment("x", 60), _segment("a", 250, 62.0),
        _segment("x", 60), _segment("a", 240, 62.0), _segment("sil", 70)]))
    return phrases


def _write_language(directory, phrases, *, tract_scale: float, seed: int) -> dict:
    renderer = TwoLanguageSinger(SingerConfig(fs=TEST_FS, seed=seed),
                                 tract_scale=tract_scale,
                                 extra=LANGUAGE_B_EXTRA)
    return make_dataset(directory, fs=TEST_FS, renderer=renderer,
                        phrases=phrases, seed=seed)


def _training_config(corpus: dict, **overrides) -> TrainingConfig:
    values = dict(label_file=corpus["labels"], wav_dir=corpus["wav_dir"],
                  fs=TEST_FS, fft_size=TEST_FFT, n_mcep=20, n_band=5,
                  n_iterations=1, min_phoneme_frames=MIN_PHONEME_FRAMES,
                  seed=SEED)
    values.update(overrides)
    return TrainingConfig(**values)


@pytest.fixture(scope="module")
def phoneme_set() -> PhonemeSet:
    return transfer_phoneme_set()


@pytest.fixture(scope="module")
def corpora(tmp_path_factory) -> dict:
    """Voice 1's corpus (language A), voice 2's (language B) and a B score."""
    root = tmp_path_factory.mktemp("transfer-corpus")
    return {
        "root": root,
        "a": _write_language(root / "langA", language_a_phrases(),
                             tract_scale=0.86, seed=5),
        "b": _write_language(root / "langB", language_b_phrases(),
                             tract_scale=1.18, seed=11),
    }


@pytest.fixture(scope="module")
def auxiliary(corpora, phoneme_set) -> HMSModel:
    """Voice 2: trained on language B with its own pitch bins, saved to disk."""
    config = _training_config(corpora["b"], pitch_conditioning_enabled=True)
    model = Trainer(config, phoneme_set).train()
    model.name = "voice-2"
    model.save(Path(corpora["root"]) / "voice-2-model")
    return model


def _train_transferred(corpora, phoneme_set) -> HMSModel:
    config = _training_config(
        corpora["a"], pitch_conditioning_enabled=True,
        transfer_enabled=True, transfer_target_language="en",
        transfer_target_speaker="voice-1",
        transfer_auxiliary_language="de", transfer_auxiliary_speaker="voice-2",
        transfer_auxiliary_model=str(Path(corpora["root"]) / "voice-2-model"))
    model = Trainer(config, phoneme_set).train()
    model.name = "voice-1"
    return model


@pytest.fixture(scope="module")
def transferred(corpora, auxiliary, phoneme_set) -> HMSModel:
    """Voice 1, trained in language A with voice 2 attached (the feature)."""
    return _train_transferred(corpora, phoneme_set)


@pytest.fixture(scope="module")
def baseline(corpora, phoneme_set) -> HMSModel:
    """The same voice and corpus, trained with the feature off."""
    config = _training_config(corpora["a"], pitch_conditioning_enabled=True)
    return Trainer(config, phoneme_set).train()


@pytest.fixture(scope="module")
def note_score() -> Score:
    """A short language-B score sung on one of B's own pitch regions."""
    return Score([Utterance("probe", [
        Segment("sil", 0.00, 0.06),
        Segment("y", 0.06, 0.31, note=62.0),
        Segment("a", 0.31, 0.56, note=62.0),
        Segment("x", 0.56, 0.68),
        Segment("m", 0.68, 0.78, note=62.0),
        Segment("sil", 0.78, 0.86)])])


def _sing(model: HMSModel, score: Score, **overrides):
    options = dict(seed=0, vocoder="builtin")
    options.update(overrides)
    return Synthesizer(model, SynthesisConfig(**options)).synthesize(score)


# --------------------------------------------------------------------------
# 1. the feature is off unless asked for, and writes nothing when off
# --------------------------------------------------------------------------


def test_transfer_is_disabled_by_default():
    parameters = load_parameters(None)
    assert parameters["transfer"]["enabled"] is False
    config = training_config_from_parameters(parameters)
    assert config.transfer_enabled is False
    assert config.transfer_auxiliary_model is None
    assert config.transfer_auxiliary_labels is None
    assert config.transfer_target_language == ""
    assert config.transfer_auxiliary_language == ""
    assert config.transfer_adapt_pitch_bins is True
    assert config.transfer_map_prior_strength == DEFAULT_MAP_PRIOR_STRENGTH
    assert config.transfer_map_adapt_frames == DEFAULT_MAP_ADAPT_FRAMES


def test_a_voice_without_transfer_carries_no_transfer_artifacts(
        tmp_path, baseline):
    """The disabled path is the path that existed before the feature."""
    assert baseline.transfer.active is False
    assert baseline.transfer_models == {}
    assert baseline.transfer_index == {}
    assert baseline.transfer_map is None
    assert baseline.transfer_n_free_params == 0
    assert baseline.get_transfer_hmm("y") is None
    assert baseline.has_transfer_hmm("y") is False
    assert baseline.n_free_params == baseline.phoneme_n_free_params \
        + baseline.context_n_free_params + baseline.backoff_n_free_params \
        + baseline.global_backoff_n_free_params + baseline.pitch_n_free_params

    directory = baseline.save(tmp_path / "plain")
    assert not (directory / "transfer.npz").exists()
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    assert "\ntransfer:" not in text and "\ntransfer_index:" not in text
    assert "transferred" not in baseline.summary()

    loaded = HMSModel.load(directory)
    assert loaded.transfer.active is False
    assert loaded.transfer_models == {} and loaded.transfer_map is None
    assert loaded.loaded_format_version == MODEL_FORMAT_VERSION


def test_disabled_training_leaves_the_native_units_alone(transferred, baseline):
    """Every unit voice 1 trained on its own is bit-identical either way."""
    assert set(transferred.hmms) == set(baseline.hmms) == {"a", "e", "i", "sil"}
    for phone, hmm in baseline.hmms.items():
        plain, with_transfer = hmm.to_arrays(), \
            transferred.hmms[phone].to_arrays()
        assert set(plain) == set(with_transfer)
        for key, value in plain.items():
            assert np.array_equal(value, with_transfer[key]), (phone, key)
    assert set(baseline.pitch_models) <= set(transferred.pitch_models)


# --------------------------------------------------------------------------
# 2. configuration parsing and validation
# --------------------------------------------------------------------------


def test_transfer_section_parses_into_the_training_config(tmp_path):
    parameters = load_parameters(None)
    parameters["transfer"].update({
        "enabled": True, "target_language": "en",
        "auxiliary_language": "de", "auxiliary_speaker": "voice-2",
        "auxiliary_model": str(tmp_path / "voice-2-model"),
        "adapt_pitch_bins": False, "map_prior_strength": 3.5,
        "map_adapt_frames": 250.0,
    })
    config = training_config_from_parameters(parameters)
    assert config.transfer_enabled is True
    assert config.transfer_target_language == "en"
    assert config.transfer_auxiliary_language == "de"
    assert config.transfer_auxiliary_speaker == "voice-2"
    assert config.transfer_auxiliary_model == str(tmp_path / "voice-2-model")
    assert config.transfer_adapt_pitch_bins is False
    assert config.transfer_map_prior_strength == 3.5
    assert config.transfer_map_adapt_frames == 250.0

    parameters["transfer"]["mystery"] = 1
    with pytest.raises(ValueError, match="unknown transfer option"):
        training_config_from_parameters(parameters)


@pytest.mark.parametrize("overrides, message", [
    ({"transfer_auxiliary_model": "m", "transfer_auxiliary_language": "de"},
     "target language"),
    ({"transfer_target_language": "en", "transfer_auxiliary_model": "m"},
     "auxiliary language"),
    ({"transfer_target_language": "en", "transfer_auxiliary_language": "de"},
     "exactly one auxiliary voice source"),
    ({"transfer_target_language": "en", "transfer_auxiliary_language": "de",
      "transfer_auxiliary_model": "m", "transfer_auxiliary_labels": "l"},
     "exactly one auxiliary voice source"),
    ({"transfer_target_language": "en", "transfer_auxiliary_language": "de",
      "transfer_auxiliary_labels": "l"},
     "auxiliary_wav_dir"),
    ({"transfer_target_language": "en", "transfer_auxiliary_language": "de",
      "transfer_auxiliary_model": "m", "transfer_map_prior_strength": 0.0},
     "map_prior_strength"),
    ({"transfer_target_language": "en", "transfer_auxiliary_language": "de",
      "transfer_auxiliary_model": "m", "transfer_map_adapt_frames": -1.0},
     "map_adapt_frames"),
])
def test_transfer_configuration_is_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(transfer_enabled=True, **overrides)


def test_transfer_record_round_trips_through_its_dict():
    record = CrossLanguageTransfer(
        enabled=True, target_speaker="voice-1", target_language="en",
        auxiliary_speaker="voice-2", auxiliary_language="de",
        auxiliary_source="/models/voice-2", adapt_pitch_bins=False,
        map_prior_strength=2.0, map_adapt_frames=50.0,
        anchor_phones=("i", "a"), anchor_classes=("vowel",),
        transferred_phones=("y",), transferred_pitch_units=3)
    again = CrossLanguageTransfer.from_dict(record.to_dict())
    assert again == record
    assert again.target_label == "voice-1 (en)"
    assert again.auxiliary_label == "voice-2 (de)"
    assert "2 anchor phone(s)" in record.describe()
    with pytest.raises(ValueError, match="unknown cross-language transfer"):
        CrossLanguageTransfer.from_dict({"enabled": True, "slope": 1.0})


# --------------------------------------------------------------------------
# 3. the acoustic map's arithmetic
# --------------------------------------------------------------------------


def _random_affine(rng, dim=4):
    matrix = np.eye(dim) + 0.3 * rng.normal(size=(dim, dim))
    intercept = 0.2 * rng.normal(size=dim)
    return matrix, intercept


def test_fit_acoustic_map_recovers_an_affine_relation():
    rng = np.random.default_rng(0)
    matrix, intercept = _random_affine(rng)
    x = rng.normal(size=(8, 4))
    y = x @ matrix.T + intercept
    fitted = fit_acoustic_map([(x[i], y[i]) for i in range(len(x))], 4,
                              prior_strength=1e-9)
    assert np.allclose(fitted.matrix, matrix, atol=1e-6)
    assert np.allclose(fitted.intercept, intercept, atol=1e-6)
    assert fitted.identity is False
    assert fitted.n_anchors == 0            # names are passed in separately
    assert (fitted.residual_variance <= MIN_TRANSFER_VARIANCE + 1e-9).all()
    assert fitted.linear_deviation > 0


def test_fit_acoustic_map_shrinks_towards_the_identity():
    rng = np.random.default_rng(1)
    matrix, intercept = _random_affine(rng)
    x = rng.normal(size=(6, 4))
    y = x @ matrix.T + intercept
    pairs = [(x[i], y[i]) for i in range(len(x))]

    weak = fit_acoustic_map(pairs, 4, prior_strength=1e-9)
    strong = fit_acoustic_map(pairs, 4, prior_strength=1e6)
    assert strong.identity is False
    assert np.allclose(strong.matrix, np.eye(4), atol=1e-3)
    assert np.allclose(strong.intercept, 0.0, atol=1e-3)
    assert strong.linear_deviation < weak.linear_deviation


def test_no_anchor_is_the_identity_map_with_a_fallback_residual():
    fallback = np.array([0.7, 0.9, 1.1, 1.3])
    fitted = fit_acoustic_map([], 4, prior_strength=1.0,
                              fallback_spread=fallback)
    assert fitted.identity is True
    assert np.allclose(fitted.matrix, np.eye(4))
    assert np.allclose(fitted.intercept, 0.0)
    assert np.allclose(fitted.residual_variance, fallback)
    moved = fitted.apply_means(np.array([[1.0, -1.0, 2.0, 0.0]]))
    assert np.allclose(moved, [[1.0, -1.0, 2.0, 0.0]])


def test_map_moves_means_through_every_stream_and_clips_the_correction():
    matrix = np.array([[2.0, 0.0], [0.0, 0.5]])
    acoustic_map = AcousticMap(matrix=matrix, intercept=np.array([1.0, 0.0]),
                               residual_variance=np.array([0.1, 0.1]),
                               static_dim=2)
    # two streams of the same two dimensions: static block, then its deltas
    means = np.array([[1.0, 2.0, 1.0, 2.0]])
    moved = acoustic_map.apply_means(means)
    assert np.allclose(moved, [[3.0, 1.0, 2.0, 1.0]])

    wild = AcousticMap(matrix=np.eye(2) * 1000.0, intercept=np.zeros(2),
                       residual_variance=np.ones(2), static_dim=2)
    clipped = wild.apply_means(np.array([[1.0, -1.0]]))
    assert np.allclose(clipped, [[1.0 + MAX_MEAN_SHIFT,
                                  -1.0 - MAX_MEAN_SHIFT]])


def test_variance_propagation_plus_residual_and_its_bounds():
    matrix = np.array([[0.5, 0.5], [0.0, 2.0]])
    acoustic_map = AcousticMap(matrix=matrix, intercept=np.zeros(2),
                               residual_variance=np.array([0.25, 0.5]),
                               static_dim=2)
    variances = np.array([[1.0, 1.0]])
    # dim 0: (0.25 + 0.25) * 1 + 0.25 ; dim 1: (0 + 4) * 1 + 0.5
    assert np.allclose(acoustic_map.apply_variances(variances),
                       [[0.75, 4.5]])

    exploding = AcousticMap(matrix=np.eye(2) * 1e6, intercept=np.zeros(2),
                            residual_variance=np.array([0.5, 0.5]),
                            static_dim=2)
    bounded = exploding.apply_variances(variances)
    assert np.allclose(bounded, [[MAX_SPREAD_FACTOR + 0.5,
                                  MAX_SPREAD_FACTOR + 0.5]])

    collapsing = AcousticMap(matrix=np.zeros((2, 2)), intercept=np.zeros(2),
                             residual_variance=np.array([0.25, 0.25]),
                             static_dim=2)
    assert np.allclose(collapsing.apply_variances(variances),
                       [[MIN_SPREAD_FACTOR + 0.25, MIN_SPREAD_FACTOR + 0.25]])


def _unit(means, *, variance=0.5, n_states=1, components=1, voiced_prob=1.0):
    """A hand-built HMM: ``means`` is one (state, component) mean vector."""
    mean = np.atleast_1d(np.asarray(means, dtype=np.float64))
    hmm = LeftToRightHMM(n_states=n_states)
    hmm.dim = int(mean.size)
    for index in range(n_states):
        hmm.states[index] = HMMState(
            gmm=DiagGMM(np.full(components, 1.0 / components),
                        np.tile(mean, (components, 1)),
                        np.full((components, mean.size), float(variance))),
            duration=StateDurationStats(mean=3.0 + index, variance=0.5,
                                        count=7 + index),
            voiced_prob=voiced_prob)
    hmm.self_loops = np.linspace(0.3, 0.8, n_states)
    return hmm


def test_map_hmm_adapts_emissions_and_copies_structure():
    hmm = _unit([1.0, -2.0], n_states=2, components=2, voiced_prob=0.25)
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    acoustic_map = AcousticMap(matrix=matrix, intercept=np.array([0.5, -0.5]),
                               residual_variance=np.array([0.25, 0.5]),
                               static_dim=2)
    mapped = map_hmm(hmm, acoustic_map)

    assert mapped is not hmm
    assert mapped.n_states == hmm.n_states
    assert mapped.allow_skip == hmm.allow_skip
    assert mapped.covariance_type == hmm.covariance_type
    assert np.array_equal(mapped.self_loops, hmm.self_loops)
    for index in range(hmm.n_states):
        original, adapted = hmm.states[index], mapped.states[index]
        assert adapted.voiced_prob == original.voiced_prob
        assert adapted.duration == original.duration
        assert np.array_equal(adapted.gmm.weights, original.gmm.weights)
        assert adapted.gmm.means.shape == original.gmm.means.shape
        expected = original.gmm.means @ matrix.T + np.array([0.5, -0.5])
        assert np.allclose(adapted.gmm.means, expected)
        assert np.allclose(adapted.gmm.variances,
                           original.gmm.variances * np.array([4.0, 9.0])
                           + np.array([0.25, 0.5]))

    # the stage-3 correction is a location shift plus a spread factor
    shifted = map_hmm(hmm, acoustic_map, shift=np.array([0.1, 0.2]),
                      spread=np.array([2.0, 0.5]))
    assert np.allclose(shifted.states[0].gmm.means,
                       mapped.states[0].gmm.means + np.array([0.1, 0.2]))
    assert np.allclose(shifted.states[0].gmm.variances,
                       mapped.states[0].gmm.variances
                       * np.array([2.0, 0.5]))


def test_map_correction_blends_towards_the_target_voices_own_frames():
    unit = _unit([1.0, 1.0], variance=0.25)
    moments = {}
    target_frames = np.tile(np.array([3.0, -1.0]), (4, 1))
    accumulate_phone_moments(moments, "y", target_frames, 2)
    entry = moments["y"]
    assert entry.count == 4
    assert np.allclose(entry.mean, [3.0, -1.0])

    shift, spread, record = map_correction(unit, entry, 2, adapt_frames=100.0)
    weight = 4.0 / (4.0 + 100.0)
    assert record["adapted_frames"] == 4
    assert record["adaptation_weight"] == pytest.approx(weight, abs=1e-6)
    expected_shift = weight * (entry.mean - pooled_mean(unit)[:2])
    assert np.allclose(shift, expected_shift)
    assert (spread > 0).all() and (spread <= 1.0).all()

    # more of the target's own frames means more of the target's own answer
    many = {}
    accumulate_phone_moments(many, "y", np.tile(np.array([3.0, -1.0]),
                                                (400, 1)), 2)
    closer, _spread, record = map_correction(unit, many["y"], 2,
                                             adapt_frames=100.0)
    assert np.linalg.norm(closer) > np.linalg.norm(shift)
    assert record["adaptation_weight"] > 0.5

    # tau = 0 (or no frames at all) is the identity correction
    identity, identity_spread, record = map_correction(unit, entry, 2, 0.0)
    assert np.allclose(identity, 0.0) and np.allclose(identity_spread, 1.0)
    assert record["adapted_frames"] == 0
    none, _spread, _record = map_correction(unit, None, 2, 100.0)
    assert np.allclose(none, 0.0)


def test_accumulated_moments_are_bounded_by_phones_times_dimensions():
    moments = {}
    frames = np.arange(400, dtype=np.float64).reshape(100, 4)
    accumulate_phone_moments(moments, "a", frames, 4)
    accumulate_phone_moments(moments, "a", frames, 4)
    accumulate_phone_moments(moments, "y", frames[:, :4], 4)
    entry = moments["a"]
    assert entry.count == 200
    assert np.allclose(entry.mean, frames.mean(axis=0))
    assert np.allclose(entry.variance, frames.var(axis=0))
    assert entry.sums.size == 4 and entry.sumsquares.size == 4
    assert entry.to_dict()["phone"] == "a"
    # four floats per dimension plus counters -- never one value per frame
    assert phone_moments_nbytes(moments) < 2 * (8 + 2 * 4 * 8 + 128)


# --------------------------------------------------------------------------
# 4. building units from two hand-made voices
# --------------------------------------------------------------------------


TINY_SPEC = FeatureSpec(fs=TEST_FS, frame_period=5.0, fft_size=TEST_FFT,
                        n_mcep=2, n_band=2, use_delta=False,
                        use_delta2=False)
TINY_STATIC = TINY_SPEC.static_dim          # 5


def _tiny_phoneme_set() -> PhonemeSet:
    return PhonemeSet.from_dict({
        "defaults": {
            "vowel": {"n_states": 2, "n_components": 1, "voiced": True},
            "unvoiced_consonant": {"n_states": 2, "n_components": 1,
                                   "voiced": False},
            "silence": {"n_states": 1, "n_components": 1, "voiced": False},
        },
        "phonemes": {
            "sil": {"type": "silence"},
            "a": {"type": "vowel"},
            "e": {"type": "vowel"},
            "i": {"type": "vowel"},
            "o": {"type": "vowel"},
            "u": {"type": "vowel"},
            "y": {"type": "vowel"},
            "Q": {"type": "vowel"},          # the auxiliary inventory only
            "s": {"type": "unvoiced_consonant"},
        },
        "aliases": {"A": "a"},
    })


def _auxiliary_phoneme_set() -> PhonemeSet:
    """The tiny inventory plus a phone only *its* side describes."""
    base = _tiny_phoneme_set()
    phonemes = dict(base.phonemes)
    phonemes["Zq"] = PhonemeDef(symbol="Zq", type="vowel", n_states=2,
                                n_components=1, voiced=True,
                                can_hold_note=True)
    return PhonemeSet(phonemes, aliases=base.aliases, defaults=base.defaults,
                      silence=base.silence)


def _voice_source(name, language, units, phoneme_set, *, spec=TINY_SPEC,
                  pitch_models=None, pitch_index=None,
                  conditioning=None, durations=None, pitched=()) -> VoiceSource:
    """A `VoiceSource` assembled by hand, with the auxiliary's extras."""
    duration_model = DurationModel()
    for phone, hmm in units.items():
        stats = DurationStats(mean=np.log(8.0), variance=0.25, count=25)
        stats.frames = 8.0
        duration_model.stats[phone] = stats
    if durations:
        duration_model.stats.update(durations)
    stats = {phone: [PitchStats(mean=0.5, variance=0.25, count=25)]
             for phone in pitched}
    pitch_model = PitchModel(stats=stats,
                             voiced_prior={phone: 0.75 for phone in pitched})
    return VoiceSource(
        speaker=name, language=language, source=f"{name}.tsv",
        hmms=dict(units), duration_model=duration_model,
        pitch_model=pitch_model, pitch_models=dict(pitch_models or {}),
        pitch_index=dict(pitch_index or {}),
        pitch_conditioning=conditioning or PitchConditioning(),
        phoneme_set=phoneme_set, spec=spec)


def _basis(index: int) -> np.ndarray:
    """A one-hot static mean of the tiny spec's dimension."""
    out = np.zeros(TINY_STATIC)
    out[index] = 1.0
    return out


def _hand_built():
    """Two voices with an exact affine relation between their anchor phones.

    Voice 1's unit means are ``2 * voice 2's + 0.5`` on every dimension, so the
    fitted map has a known answer; ``s`` exists in the target only and must not
    move at all.
    """
    target_set, aux_set = _tiny_phoneme_set(), _auxiliary_phoneme_set()
    # the anchor phones span the whole static block (sil takes dimension 2),
    # so the affine relation is determined in every direction
    base = {"a": _basis(0), "i": _basis(1), "e": _basis(3),
            "o": _basis(4), "u": _basis(0) + _basis(2), "sil": _basis(2),
            "y": _basis(2), "Q": _basis(0) + _basis(1),
            "Zq": _basis(0) + _basis(1)}
    auxiliary = _voice_source(
        "voice-2", "de", {phone: _unit(vector)
                          for phone, vector in base.items()}, aux_set,
        pitched=("Q", "Zq", "y"))
    target_units = {}
    for phone in ("a", "i", "e", "o", "u", "sil", "s"):
        vector = base.get(phone, np.full(TINY_STATIC, 0.25))
        target_units[phone] = _unit(2.0 * vector + 0.5)
    target = _voice_source("voice-1", "en", target_units, target_set)
    return target, auxiliary


def test_build_transfer_adapts_only_the_phones_the_target_lacks():
    target, auxiliary = _hand_built()
    request = CrossLanguageTransfer(enabled=True, target_speaker="voice-1",
                                    target_language="en",
                                    auxiliary_speaker="voice-2",
                                    auxiliary_language="de",
                                    auxiliary_source="voice-2-model",
                                    map_prior_strength=1e-9,
                                    map_adapt_frames=0.0)
    build = build_transfer(target=target, auxiliary=auxiliary, request=request)

    # only phones the target has no unit of its own for, and never `s`
    assert sorted(build.units) == ["Q", "Zq", "y"]
    assert "s" not in build.units
    assert build.record.anchor_phones == ("a", "e", "i", "o", "sil", "u")
    assert build.record.anchor_classes == ()
    assert build.record.transferred_phones == ("Q", "Zq", "y")
    assert build.record.transferred_pitch_units == 0
    assert build.index["y"]["anchors"] == 6
    assert build.index["y"]["class_anchors"] == 0
    assert build.index["y"]["source_speaker"] == "voice-2"
    assert build.index["y"]["source_language"] == "de"
    assert build.index["y"]["kind"] == TIER_TRANSFERRED

    # the map is the affine relation the anchors were built from: the anchor
    # phones span the whole static block, so every dimension is determined
    assert not build.acoustic_map.identity
    assert np.allclose(build.acoustic_map.matrix, 2.0 * np.eye(TINY_STATIC),
                       atol=1e-6)
    assert np.allclose(build.acoustic_map.intercept, 0.5, atol=1e-6)
    # ... and the transferred units land where it says they should
    assert np.allclose(pooled_mean(build.units["y"]),
                       2.0 * pooled_mean(auxiliary.hmms["y"]) + 0.5)
    assert np.allclose(pooled_mean(build.units["Q"]),
                       2.0 * pooled_mean(auxiliary.hmms["Q"]) + 0.5)

    # the auxiliary's own inventory entry is imported for the unknown phone
    assert set(build.phoneme_definitions) == {"Zq"}
    assert build.phoneme_definitions["Zq"].type == "vowel"
    assert "Zq" not in target.phoneme_set.phonemes
    # duration statistics travel with the unit, pitch/voicing statistics too
    assert build.duration_stats["y"].count == 25
    assert build.duration_stats["y"].frames == pytest.approx(8.0)
    assert build.voiced_prior["y"] == 0.75


def test_build_transfer_maps_the_auxiliarys_pitch_statistics():
    target, auxiliary = _hand_built()
    auxiliary.pitch_model.stats["y"] = [PitchStats(mean=1.0, variance=0.25,
                                                   count=25)]
    request = CrossLanguageTransfer(enabled=True, map_prior_strength=1e-9,
                                    map_adapt_frames=0.0)
    build = build_transfer(target=target, auxiliary=auxiliary, request=request)
    scale = float(build.acoustic_map.matrix[0, 0])
    intercept = float(build.acoustic_map.intercept[0])
    residual = float(build.acoustic_map.residual_variance[0])
    stat = build.pitch_stats["y"][0]
    assert stat.mean == pytest.approx(scale * 1.0 + intercept)
    assert stat.variance == pytest.approx(scale ** 2 * 0.25 + residual)
    assert stat.count == 25


def test_build_transfer_falls_back_to_class_anchors_with_no_shared_phone():
    """Two languages that share no *phone* still share phoneme classes."""
    target, auxiliary = _hand_built()
    auxiliary.hmms = {phone: hmm for phone, hmm in auxiliary.hmms.items()
                      if phone in ("y", "Q")}
    build = build_transfer(target=target, auxiliary=auxiliary,
                           request=CrossLanguageTransfer(enabled=True))
    assert sorted(build.units) == ["Q", "y"]
    assert build.record.anchor_phones == ()
    assert build.record.anchor_classes == ("vowel",)
    assert build.acoustic_map.identity is False
    assert build.acoustic_map.n_anchors == 1
    assert any("share no trained phone" in message
               for message in build.diagnostics)


def test_build_transfer_with_no_anchor_at_all_is_the_identity():
    """With no statistics to learn from, the units keep their own geometry."""
    target, auxiliary = _hand_built()
    auxiliary.hmms = {phone: hmm for phone, hmm in auxiliary.hmms.items()
                      if phone in ("y", "Q")}
    target.phoneme_set = None
    auxiliary.phoneme_set = None
    build = build_transfer(target=target, auxiliary=auxiliary,
                           request=CrossLanguageTransfer(enabled=True))
    assert sorted(build.units) == ["Q", "y"]
    assert build.acoustic_map.identity is True
    assert np.allclose(build.acoustic_map.matrix, np.eye(TINY_STATIC))
    assert build.record.anchor_phones == () and \
        build.record.anchor_classes == ()
    assert np.allclose(pooled_mean(build.units["y"]),
                       pooled_mean(auxiliary.hmms["y"]))
    assert build.diagnostics


def test_build_transfer_can_skip_the_auxiliarys_pitch_bins():
    target, auxiliary = _hand_built()
    bins = {("y", 9): _unit(_basis(2))}
    index = {("y", 9): {"kind": "phone", "unit": "y", "pitch_bin": 9,
                        "frames": 50, "occurrences": 1}}
    auxiliary.pitch_models = dict(bins)
    auxiliary.pitch_index = dict(index)
    auxiliary.pitch_conditioning = PitchConditioning(enabled=True,
                                                     bin_size=BIN_SIZE)
    target.pitch_conditioning = PitchConditioning(enabled=True,
                                                  bin_size=BIN_SIZE)
    request = CrossLanguageTransfer(enabled=True, adapt_pitch_bins=True)
    build = build_transfer(target=target, auxiliary=auxiliary, request=request,
                           min_phone_frames=MIN_PHONEME_FRAMES)
    assert ("y", 9) in build.pitch_models
    assert build.pitch_index[("y", 9)]["transferred"] is True
    assert build.pitch_index[("y", 9)]["frames"] == 50
    assert build.record.transferred_pitch_units == 1
    # the pitch condition does not make the *phone* any different
    assert np.allclose(pooled_mean(build.pitch_models[("y", 9)]),
                       pooled_mean(build.units["y"]))

    off = build_transfer(target=target, auxiliary=auxiliary,
                         request=CrossLanguageTransfer(
                             enabled=True, adapt_pitch_bins=False),
                         min_phone_frames=MIN_PHONEME_FRAMES)
    assert off.pitch_models == {}
    assert off.record.transferred_pitch_units == 0

    # a target trained without pitch conditioning cannot use them
    target.pitch_conditioning = PitchConditioning(enabled=False)
    plain = build_transfer(target=target, auxiliary=auxiliary,
                           request=CrossLanguageTransfer(enabled=True),
                           min_phone_frames=MIN_PHONEME_FRAMES)
    assert plain.pitch_models == {}
    assert any("without pitch conditioning" in message
               for message in plain.diagnostics)


def test_build_transfer_refuses_incomparable_voices():
    target, auxiliary = _hand_built()
    other = FeatureSpec(fs=44100, frame_period=10.0, fft_size=TEST_FFT,
                        n_mcep=3, n_band=2, use_delta=True, use_delta2=False)
    auxiliary.spec = other
    with pytest.raises(ValueError, match="feature"):
        build_transfer(target=target, auxiliary=auxiliary,
                       request=CrossLanguageTransfer(enabled=True))

    target, auxiliary = _hand_built()
    auxiliary.spec = target.spec
    target.pitch_conditioning = PitchConditioning(enabled=True, bin_size=6)
    auxiliary.pitch_conditioning = PitchConditioning(enabled=True, bin_size=12)
    auxiliary.pitch_models = {("y", 4): _unit(_basis(2))}
    auxiliary.pitch_index = {("y", 4): {"frames": 50}}
    with pytest.raises(ValueError, match="bin"):
        build_transfer(target=target, auxiliary=auxiliary,
                       request=CrossLanguageTransfer(enabled=True))


def test_build_transfer_with_the_feature_off_is_a_no_op():
    target, auxiliary = _hand_built()
    build = build_transfer(target=target, auxiliary=auxiliary,
                           request=CrossLanguageTransfer())
    assert build.units == {} and build.index == {}
    assert build.acoustic_map is None
    assert build.active is False
    assert build.describe() == ["cross-language transfer: disabled"]


# --------------------------------------------------------------------------
# 5. phones the target corpus cannot sing
# --------------------------------------------------------------------------


def test_the_two_corpora_keep_their_own_inventories(auxiliary, transferred):
    """Language B's phones are covered, but never as the target's own."""
    assert set(auxiliary.hmms) == {"a", "e", "i", "y", "oe", "sil"}
    assert set(transferred.hmms) == {"a", "e", "i", "sil"}
    assert set(transferred.transfer_models) == {"y", "oe"}
    assert transferred.transfer.transferred_phones == ("oe", "y")
    assert set(transferred.transfer.anchor_phones) == {"a", "e", "i", "sil"}


def test_phones_absent_from_the_target_corpus_resolve_to_transferred_units(
        transferred):
    for phone in ("y", "oe"):
        assert transferred.has_native_hmm(phone) is False
        assert transferred.has_transfer_hmm(phone) is True
        key, hmm, tier = transferred.resolve_unit("sil", phone, "sil")
        assert key is None and tier == TIER_TRANSFERRED
        assert hmm is transferred.transfer_models[phone]
        assert transferred.unit_is_transferred(key, tier) is True
    # aliases go through the inventory, like every other lookup
    assert transferred.has_transfer_hmm("Y") is True

    # a phone *in the auxiliary corpus* below its support threshold is not
    # transferred, and neither is a phone no corpus ever sang
    for phone in ("x", "m", "n", "t"):
        _key, _hmm, tier = transferred.resolve_unit("sil", phone, "sil")
        assert tier in ("class", "global"), phone
        assert transferred.has_transfer_hmm(phone) is False

    # the native tier always wins over a transferred unit
    key, hmm, tier = transferred.resolve_unit("sil", "a", "sil")
    assert key is None and tier == "phone"
    assert hmm is transferred.hmms["a"]


def test_transferred_units_are_adapted_not_copied(auxiliary, transferred,
                                                  baseline):
    for phone in ("y", "oe"):
        raw = pooled_mean(auxiliary.hmms[phone])
        adapted = pooled_mean(transferred.transfer_models[phone])
        assert np.isfinite(adapted).all()
        assert not np.allclose(raw, adapted)
        assert np.abs(raw - adapted)[:transferred.static_dim].max() > 1e-3
        index = transferred.transfer_index[phone]
        assert index["kind"] == TIER_TRANSFERRED
        assert index["source_speaker"] == "voice-2"
        assert index["source_language"] == "de"
        assert index["anchors"] == 4 and index["class_anchors"] == 0
    # the acoustic map is auditable from the model itself
    acoustic_map = transferred.transfer_map
    assert acoustic_map is not None and not acoustic_map.identity
    assert acoustic_map.static_dim == transferred.static_dim
    assert acoustic_map.anchor_phones == ("a", "e", "i", "sil")
    assert acoustic_map.anchor_classes == ()
    assert acoustic_map.linear_deviation > 0

    # language B's timing and phonation travel with the phone ...
    for phone in ("y", "oe"):
        source, unit = auxiliary.hmms[phone], \
            transferred.transfer_models[phone]
        assert np.array_equal(unit.self_loops, source.self_loops)
        assert unit.states[0].duration == source.states[0].duration
        assert unit.states[0].voiced_prob == source.states[0].voiced_prob
    # ... and so do language B's duration and voicing statistics, for phones
    # this corpus never observed.  Where it *has* its own observation -- the
    # 30 ms glimpse of `y` -- its own statistic keeps precedence: a transfer
    # only ever fills a gap.
    assert transferred.duration_model.stats["oe"].mean == \
        auxiliary.duration_model.stats["oe"].mean
    assert transferred.duration_model.stats["y"].mean == \
        baseline.duration_model.stats["y"].mean
    assert transferred.duration_model.stats["y"].count == 1
    assert transferred.pitch_model.voiced_prior.get("oe") == \
        auxiliary.pitch_model.voiced_prior.get("oe")


def test_the_targets_own_statistics_are_not_replaced(baseline, transferred):
    """Native duration/pitch/voicing statistics survive the transfer stage."""
    for phone in ("a", "e", "i", "sil"):
        assert baseline.duration_model.stats[phone].count > 0
        assert transferred.duration_model.stats[phone].mean == \
            baseline.duration_model.stats[phone].mean
        assert transferred.duration_model.stats[phone].variance == \
            baseline.duration_model.stats[phone].variance
        assert transferred.pitch_model.voiced_prior.get(phone) == \
            baseline.pitch_model.voiced_prior.get(phone)
        assert len(transferred.pitch_model.stats.get(phone, [])) == \
            len(baseline.pitch_model.stats.get(phone, []))


def test_a_language_b_score_sings_with_the_transferred_voice(
        corpora, transferred):
    """The point of the feature: voice 1 sings language B's phones."""
    score = labels_module.load(corpora["b"]["score"])
    result = _sing(transferred, score)

    assert np.isfinite(result.audio).all()
    assert result.audio.size > 0
    assert "y" in result.phones and "oe" in result.phones
    # both phones are rendered from transferred units, and the report says so
    joined = " ".join(result.diagnostics)
    assert "cross-language transfer" in joined
    assert "frame(s) rendered from transferred units" in joined
    assert "y" in joined
    # ... while the shared phones still come from the target's own models
    assert transferred.has_native_hmm("a")
    assert sum(1 for phone in result.phones if phone == "a") > 0


# --------------------------------------------------------------------------
# 6. the pitch-conditioned tier
# --------------------------------------------------------------------------


def test_transferred_phones_carry_the_auxiliarys_pitch_bins(auxiliary,
                                                            transferred):
    assert transferred.pitch_conditioning.active
    transferred_bins = {(unit, pitch_bin) for (unit, pitch_bin)
                        in transferred.pitch_models
                        if transferred.pitch_index[(unit, pitch_bin)]
                        .get("transferred")}
    assert {unit for unit, _ in transferred_bins} == {"y", "oe"}
    assert {b for _, b in transferred_bins} == {
        pitch_bin(note, BIN_SIZE) for note in LANGUAGE_B_NOTES}
    for (unit, bin_index) in transferred_bins:
        assert unit in auxiliary.hmms
        assert np.isfinite(
            transferred.pitch_models[(unit, bin_index)].states[0]
            .gmm.means).all()

    # a note in one of those bins resolves to the transferred pitch-conditioned
    # unit -- and is recognisable as transferred even at the ordinary tier name
    bin_index = pitch_bin(LANGUAGE_B_NOTES[0], BIN_SIZE)
    key, hmm, tier = transferred.resolve_unit("sil", "y", "sil",
                                              pitch_bin=bin_index)
    assert key == ("y", bin_index) and tier == "phone+pitch"
    assert hmm is transferred.pitch_models[("y", bin_index)]
    assert transferred.unit_is_transferred(key, tier) is True
    # native phones keep their own bins, and their own identity
    native_bin = pitch_bin(LANGUAGE_A_NOTES[0], BIN_SIZE)
    key, hmm, tier = transferred.resolve_unit("sil", "a", "sil",
                                              pitch_bin=native_bin)
    assert key == ("a", native_bin) and tier == "phone+pitch"
    assert transferred.unit_is_transferred(key, tier) is False
    assert hmm is transferred.pitch_models[("a", native_bin)]


def test_a_missing_pitch_bin_falls_back_to_the_transferred_phone(transferred):
    unknown_bin = pitch_bin(LANGUAGE_B_NOTES[2] + 24.0, BIN_SIZE)
    assert ("y", unknown_bin) not in transferred.pitch_models
    key, hmm, tier = transferred.resolve_unit("sil", "y", "sil",
                                              pitch_bin=unknown_bin)
    assert key is None and tier == TIER_TRANSFERRED
    assert hmm is transferred.transfer_models["y"]
    assert transferred.unit_is_transferred(key, tier) is True

    # a native phone whose own bin is missing never borrows the auxiliary's:
    # the target trained no `i` at the highest of language B's notes
    aux_only_bin = pitch_bin(LANGUAGE_B_NOTES[2], BIN_SIZE)
    assert ("i", aux_only_bin) not in transferred.pitch_models
    key, hmm, tier = transferred.resolve_unit("sil", "i", "sil",
                                              pitch_bin=aux_only_bin)
    assert key is None and tier == "phone" and hmm is transferred.hmms["i"]


def test_synthesis_reports_pitch_conditioned_transfer(note_score, transferred):
    result = _sing(transferred, note_score)
    joined = " ".join(result.diagnostics)
    assert "transferred units" in joined
    # `x` and `m` have neither a native nor a transferred unit: they must fall
    # back, and the report must say so rather than hide it
    assert "backoff" in joined
    assert result.audio.size > 0


# --------------------------------------------------------------------------
# 7. serialisation
# --------------------------------------------------------------------------


def test_transfer_survives_a_round_trip(tmp_path, transferred):
    directory = transferred.save(tmp_path / "voice-1-model")
    assert (directory / "transfer.npz").exists()
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    assert "\ntransfer:" in text and "\ntransfer_index:" in text

    loaded = HMSModel.load(directory)
    assert loaded.loaded_format_version == MODEL_FORMAT_VERSION
    assert loaded.transfer == transferred.transfer
    assert set(loaded.transfer_models) == set(transferred.transfer_models)
    for phone, hmm in transferred.transfer_models.items():
        original, restored = hmm.to_arrays(), \
            loaded.transfer_models[phone].to_arrays()
        assert set(original) == set(restored)
        for key, value in original.items():
            assert np.array_equal(value, restored[key]), (phone, key)
    assert loaded.transfer_index == transferred.transfer_index
    assert np.allclose(loaded.transfer_map.matrix,
                       transferred.transfer_map.matrix)
    assert np.allclose(loaded.transfer_map.residual_variance,
                       transferred.transfer_map.residual_variance)
    assert loaded.transfer_map.anchor_phones == \
        transferred.transfer_map.anchor_phones
    assert loaded.transfer_map.anchor_classes == \
        transferred.transfer_map.anchor_classes

    assert set(loaded.pitch_models) == set(transferred.pitch_models)
    assert loaded.pitch_index[("y", pitch_bin(
        LANGUAGE_B_NOTES[0], BIN_SIZE))]["transferred"] is True
    for phone in ("y", "oe"):
        _key, hmm, tier = loaded.resolve_unit("sil", phone, "sil")
        assert hmm is loaded.transfer_models[phone]
        assert tier == TIER_TRANSFERRED
    assert loaded.n_free_params == transferred.n_free_params
    assert "transferred models" in loaded.summary()


def test_missing_transfer_npz_is_a_clean_error(tmp_path, transferred):
    directory = transferred.save(tmp_path / "model")
    (directory / "transfer.npz").unlink()
    with pytest.raises(ValueError, match="transfer.npz"):
        HMSModel.load(directory)


@pytest.mark.parametrize("legacy_version", [2, 3, 4])
def test_older_model_formats_load_with_transfer_disabled(
        tmp_path, transferred, legacy_version):
    """A voice trained before this feature needs no migration to load."""
    directory = tmp_path / "model"
    transferred.save(directory)
    with open(directory / "model.yaml", "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document.pop("transfer", None)
    document.pop("transfer_index", None)
    document["format_version"] = legacy_version

    # A real format-4 file has no transferred pitch bins either: its
    # `pitch_index` only ever described this voice's own models.
    with np.load(directory / "pitch.npz") as handle:
        arrays = {key: handle[key] for key in handle.files}
    dropped = []
    pitch_index = document.get("pitch_index") or {}
    for unit, bins in list(pitch_index.items()):
        for raw_bin, info in list(bins.items()):
            if (info or {}).get("transferred"):
                bins.pop(raw_bin)
                dropped.append(f"{unit}/{raw_bin}/")
        if not bins:
            pitch_index.pop(unit)
    if not pitch_index:
        document.pop("pitch_index", None)
    if dropped:
        arrays = {key: value for key, value in arrays.items()
                  if not any(key.startswith(prefix) for prefix in dropped)}
        np.savez_compressed(directory / "pitch.npz", **arrays)

    with open(directory / "model.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
    (directory / "transfer.npz").unlink()

    loaded = HMSModel.load(directory)
    assert loaded.loaded_format_version == legacy_version
    assert loaded.transfer.active is False
    assert loaded.transfer_models == {} and loaded.transfer_index == {}
    assert loaded.transfer_map is None
    assert loaded.transfer_n_free_params == 0
    _key, _hmm, tier = loaded.resolve_unit("sil", "y", "sil")
    assert tier in ("class", "global")
    assert "transferred" not in loaded.summary()


def test_future_format_is_still_refused(tmp_path, transferred):
    directory = tmp_path / "model"
    transferred.save(directory)
    text = (directory / "model.yaml").read_text(encoding="utf-8")
    (directory / "model.yaml").write_text(
        text.replace(f"format_version: {MODEL_FORMAT_VERSION}",
                     f"format_version: {MODEL_FORMAT_VERSION + 1}"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="not supported"):
        HMSModel.load(directory)


# --------------------------------------------------------------------------
# 8. diagnostics, evaluation and determinism
# --------------------------------------------------------------------------


def test_evaluate_reports_transferred_frames(corpora, baseline, transferred):
    report = evaluate_models([baseline, transferred], corpora["b"]["labels"],
                             corpora["b"]["wav_dir"], vocoder="builtin")
    models = report["models"]
    assert "transferred_frames" in next(iter(models.values()))
    # language B's corpus contains phones the baseline simply cannot sing
    assert models[baseline.name]["transferred_frames"] == 0
    assert models["voice-1"]["transferred_frames"] > 0


def test_training_is_deterministic(corpora, auxiliary, phoneme_set, note_score):
    again = _train_transferred(corpora, phoneme_set)
    first = _train_transferred(corpora, phoneme_set)
    assert set(first.transfer_models) == set(again.transfer_models)
    for phone, hmm in first.transfer_models.items():
        original = hmm.to_arrays()
        restored = again.transfer_models[phone].to_arrays()
        for key, value in original.items():
            assert np.array_equal(value, restored[key]), (phone, key)
    for key, value in first.transfer_map.to_arrays().items():
        assert np.array_equal(value, again.transfer_map.to_arrays()[key])
    assert first.transfer.transferred_phones == \
        again.transfer.transferred_phones
    assert np.array_equal(
        _sing(first, note_score).audio, _sing(again, note_score).audio)


# --------------------------------------------------------------------------
# 9. memory safety
# --------------------------------------------------------------------------


def test_the_moment_pass_reads_the_cache_through_the_mmap(
        tmp_path, monkeypatch, phoneme_set):
    """The transfer stage must not hold feature matrices in memory."""
    spec = FeatureSpec(fs=TEST_FS, frame_period=5.0, fft_size=TEST_FFT,
                       n_mcep=2, n_band=2, use_delta=False, use_delta2=False)
    rng = np.random.default_rng(0)
    features_path = tmp_path / "u0.features.npy"
    np.save(features_path, rng.normal(size=(400, spec.dim)))
    utterance = _CachedUtterance(
        name="u0", features_path=features_path,
        relative_pitch_path=tmp_path / "u0.relative_pitch.npy",
        voiced_path=tmp_path / "u0.voiced.npy",
        phoneme_spans=[("a", 0, 150), ("y", 150, 260), ("y", 260, 400)],
        n_frames=400)

    trainer = Trainer(TrainingConfig(label_file="unused.tsv"), phoneme_set)
    trainer.spec = spec
    seen = []
    real_load = np.load

    def recording_load(path, *args, **kwargs):
        array = real_load(path, *args, **kwargs)
        if "mmap_mode" in kwargs:
            seen.append(weakref.ref(array))
        return array

    monkeypatch.setattr("hms.core.trainer.np.load", recording_load)
    moments = trainer._collect_cached_phone_moments([utterance])
    gc.collect()

    assert seen and all(ref() is None for ref in seen), \
        "the moment pass kept a memory-mapped feature matrix alive"
    assert set(moments) == {"a", "y"}
    assert moments["y"].count == 250 and moments["a"].count == 150
    assert phone_moments_nbytes(moments) < 2 * (8 + 2 * spec.static_dim * 8
                                                + 128)


def test_transfer_training_releases_each_utterance_into_the_cache(
        tmp_path, monkeypatch):
    """The extra pass changes nothing about the disk-backed training discipline.

    Mirrors the memory test of the disk-backed cache: a recording vocoder
    asserts that the previous utterance's waveform and WORLD arrays are gone by
    the time the next one is analysed -- now with the cross-language tier on,
    which adds an auxiliary model load, a moment pass and unit mapping.
    """
    frame_count = 12
    spec = FeatureSpec(fs=8000, frame_period=10.0, fft_size=16, n_mcep=3,
                       n_band=2, use_delta=False, use_delta2=False)

    # an auxiliary voice on disk: one unit language A's corpus will never sing
    auxiliary_set = _tiny_phoneme_set()
    auxiliary = HMSModel(
        name="voice-2", spec=spec, phoneme_set=auxiliary_set,
        hmms={"a": _unit([0.4] * spec.dim),
              "y": _unit([-0.4] * spec.dim)},
        duration_model=DurationModel(),
        pitch_model=PitchModel(voiced_prior={"y": 0.9}))
    auxiliary_dir = tmp_path / "voice-2-model"
    auxiliary.save(auxiliary_dir)

    score = labels_module.Score([
        labels_module.Utterance("first", [labels_module.Segment(
            "a", 0.0, 0.12, note=60.0)]),
        labels_module.Utterance("second", [labels_module.Segment(
            "a", 0.0, 0.12, note=64.0)]),
    ])
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    for name in ("first", "second"):
        (wav_dir / f"{name}.wav").write_bytes(b"test")

    class RecordingVocoder:
        name = "test"

        def __init__(self):
            self.intermediates = []

        def analyze_to_sequence(self, signal, fs, frame_period, **_kwargs):
            gc.collect()
            assert all(ref() is None for refs in self.intermediates
                       for ref in refs), (
                "the previous utterance's waveform/WORLD arrays are still live")
            index = int(signal[0])
            f0 = np.full(frame_count, 220.0 + 20.0 * index)
            sp = np.full((frame_count, spec.n_bins), 1.0 + 0.1 * index)
            ap = np.full((frame_count, spec.n_bins), 0.25)
            self.intermediates.append(tuple(
                weakref.ref(array) for array in (signal, f0, sp, ap)))
            return AcousticFrameSequence(
                f0=f0, sp=sp, ap=ap, frame_period=frame_period,
                fs=fs, fft_size=spec.fft_size)

    def read_wav(path):
        return np.full(4, 0 if Path(path).stem == "first" else 1.0), 8000

    monkeypatch.setattr("hms.core.trainer.wavio.read_wav", read_wav)
    monkeypatch.setattr("hms.core.trainer.labels_module.load",
                        lambda *_args, **_kwargs: score)
    config = TrainingConfig(
        label_file="unused.tsv", wav_dir=str(wav_dir), fs=8000,
        frame_period=10.0, fft_size=16, n_mcep=3, n_band=2,
        use_delta=False, use_delta2=False, n_iterations=1, vocoder="builtin",
        min_phoneme_frames=1, seed=0,
        transfer_enabled=True, transfer_target_language="en",
        transfer_auxiliary_language="de",
        transfer_auxiliary_model=str(auxiliary_dir))
    trainer = Trainer(config, auxiliary_set)
    trainer._vocoder = RecordingVocoder()
    model = trainer.train()

    assert len(trainer._vocoder.intermediates) == 2
    assert set(model.transfer_models) == {"y"}
    assert model.transfer.active is True
    assert model.transfer.target_language == "en"
    assert model.transfer.auxiliary_language == "de"
    assert model.has_native_hmm("a") and model.has_transfer_hmm("y")
    assert "y" not in model.hmms


# --------------------------------------------------------------------------
# 12. the documented interface: `hms train --transfer-*` and `inspect-model`
# --------------------------------------------------------------------------


def _cli_parameters(tmp_path: Path) -> Path:
    """A parameters file matching the corpora the fixtures built."""
    path = tmp_path / "parameters.yaml"
    path.write_text(yaml.safe_dump({
        # no `vocoder` here on purpose: the fixtures train with the default
        # backend, and a run that pinned another one would differ by its
        # analyser rather than by the transfer configuration under test
        "acoustic": {"fs": TEST_FS, "frame_period_ms": 5.0,
                     "fft_size": TEST_FFT, "n_mcep": 20, "n_band": 5,
                     "use_delta": True, "use_delta2": False},
        "training": {"n_iterations": 1,
                     "min_phoneme_frames": MIN_PHONEME_FRAMES, "seed": SEED},
        "pitch_conditioning": {"enabled": True, "bin_size": BIN_SIZE},
        "synthesis": {"vocoder": "builtin"},
    }, sort_keys=False), encoding="utf-8")
    return path


def _cli_phonemes(tmp_path: Path, phoneme_set: PhonemeSet) -> Path:
    path = tmp_path / "phonemes.yaml"
    path.write_text(yaml.safe_dump(phoneme_set.to_dict(), sort_keys=False),
                    encoding="utf-8")
    return path


def test_cli_language_label_alone_does_not_enable_transfer(
        tmp_path, corpora, phoneme_set, capsys):
    """`--language en` labels the corpus; only an auxiliary voice opts in."""
    out = tmp_path / "voice-1-en"
    assert main(["train", "--labels", corpora["a"]["labels"],
                 "--wav-dir", corpora["a"]["wav_dir"],
                 "--config", str(_cli_parameters(tmp_path)),
                 "--phonemes", str(_cli_phonemes(tmp_path, phoneme_set)),
                 "--language", "en",
                 "--name", "voice-1", "--out", str(out)]) == 0
    model = HMSModel.load(out)
    assert model.transfer.active is False
    assert model.transfer_models == {}
    assert "cross-language transfer" not in capsys.readouterr().out


def test_cli_transfers_from_a_model_and_from_a_corpus(
        tmp_path, corpora, auxiliary, phoneme_set, note_score, capsys):
    """Both documented ways of naming the auxiliary voice transfer."""
    phonemes = _cli_phonemes(tmp_path, phoneme_set)
    parameters = _cli_parameters(tmp_path)
    auxiliary_dir = Path(corpora["root"]) / "voice-2-model"

    def train(out, *extra):
        # The auxiliary model the fixtures wrote was analysed with the default
        # backend (`auto`), so the run that retrains voice 2 from its corpus
        # must not pin a different one -- otherwise the two ways of naming the
        # auxiliary voice would differ by their vocoder, not by their data.
        return main(["train", "--labels", corpora["a"]["labels"],
                     "--wav-dir", corpora["a"]["wav_dir"],
                     "--config", str(parameters), "--phonemes", str(phonemes),
                     "--language", "en",
                     "--name", "voice-1", "--out", str(out), *extra])

    # 1. the auxiliary voice is a model directory
    from_model = tmp_path / "from-model"
    assert train(from_model, "--transfer-model", str(auxiliary_dir),
                 "--transfer-language", "de") == 0
    log = capsys.readouterr().out
    assert "cross-language transfer (experimental)" in log
    assert "transferred phones" in log

    # 2. ... or a corpus, trained on the fly with this run's settings
    from_corpus = tmp_path / "from-corpus"
    assert train(from_corpus, "--transfer-labels", corpora["b"]["labels"],
                 "--transfer-wav-dir", corpora["b"]["wav_dir"],
                 "--transfer-language", "de", "--transfer-speaker",
                 "voice-2") == 0
    log = capsys.readouterr().out
    assert "training the auxiliary voice from" in log

    for directory in (from_model, from_corpus):
        model = HMSModel.load(directory)
        assert model.transfer.active is True
        assert model.transfer.target_speaker == "voice-1"
        assert model.transfer.target_language == "en"
        assert model.transfer.auxiliary_language == "de"
        assert model.transfer.auxiliary_speaker == "voice-2"
        assert set(model.transfer_models) == {"y", "oe"}
        assert model.transfer.transferred_pitch_units > 0
        # the language-B score sings through the transferred voice
        result = _sing(model, note_score)
        assert result.audio.size > 0
        assert any("transferred" in message for message in result.diagnostics)
    # the two ways of naming voice 2 produce the same units
    left = HMSModel.load(from_model).transfer_models
    right = HMSModel.load(from_corpus).transfer_models
    assert set(left) == set(right)
    for phone in left:
        for key, value in left[phone].to_arrays().items():
            assert np.allclose(value, right[phone].to_arrays()[key]), (phone, key)


def test_cli_inspect_model_reports_the_transfer_record(
        tmp_path, transferred, capsys):
    """`hms inspect-model --json` explains what the tier contains."""
    directory = transferred.save(tmp_path / "voice-1-in-b")
    assert main(["inspect-model", "--model", str(directory), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    transfer = report["transfer"]
    assert transfer["enabled"] is True
    assert transfer["target_speaker"] == "voice-1"
    assert transfer["auxiliary_speaker"] == "voice-2"
    assert transfer["auxiliary_language"] == "de"
    assert transfer["transferred_phones"] == ["oe", "y"]
    assert transfer["anchor_phones"]
    assert transfer["acoustic_map"]["identity"] is False
    assert transfer["acoustic_map"]["linear_deviation"] > 0
    adapted = {entry["phone"]: entry for entry in transfer["models"]}
    assert set(adapted) == {"oe", "y"}
    for entry in adapted.values():
        assert entry["kind"] == TIER_TRANSFERRED
        assert entry["native"] is False
        assert entry["source_speaker"] == "voice-2"
        assert entry["source_language"] == "de"
        assert entry["anchors"] >= 1
        assert entry["adapted_frames"] >= 0
        assert 0.0 <= entry["adaptation_weight"] <= 1.0
    # the summary lines the trainer prints are the same record
    summary = transferred.summary()
    assert "transferred models" in summary
    assert "voice-2 (de)" in summary
