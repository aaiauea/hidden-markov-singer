"""Focused tests for Phase 2's context/F0-conditioned source HMM/GMM."""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.features import FeatureSpec
from hms.core.labels import Segment, Utterance
from hms.core.phonemes import PhonemeSet
from hms.source import (SourceHMMModel, SourcePCA, SourceSequence,
                        SourceTrainer, SourceTrainingConfig,
                        SourceTrainingExample, align_source_coefficients,
                        f0_condition_features, get_source_model)


def _spec(fs: int = 22000, frame_period: float = 5.0) -> FeatureSpec:
    return FeatureSpec(fs=fs, frame_period=frame_period, fft_size=512,
                       n_mcep=4, n_band=2, use_delta=True,
                       use_delta2=False, delta_window=2,
                       f0_ref_hz=220.0, voiced_threshold=5.0)


def _phonemes() -> PhonemeSet:
    return PhonemeSet.from_dict({
        "silence": "sil",
        "phonemes": {
            "sil": {"type": "silence", "n_states": 1,
                    "n_components": 1, "voiced": False},
            "a": {"type": "vowel", "n_states": 2,
                   "n_components": 1, "voiced": True},
            "i": {"type": "vowel", "n_states": 2,
                   "n_components": 1, "voiced": True},
            "s": {"type": "unvoiced_consonant", "n_states": 1,
                   "n_components": 1, "voiced": False},
        },
    })


def _manual_pca(cycle_length: int = 16) -> SourcePCA:
    x = np.arange(cycle_length, dtype=np.float64)
    mean = np.sqrt(2.0) * np.sin(2.0 * np.pi * x / cycle_length)
    component = np.cos(2.0 * np.pi * x / cycle_length)
    component /= np.linalg.norm(component)
    return SourcePCA(mean=mean, components=component[None, :],
                     eigenvalues=np.array([1.0]), total_variance=1.0,
                     n_samples=100)


def _unit_vector(pca: SourcePCA, f0: float, phone_offset: float = 0.0
                 ) -> np.ndarray:
    """A normalized Phase-1-like vector with an F0-dependent PCA coefficient."""
    semitone = float(12.0 * np.log2(f0 / 220.0))
    coefficient = 0.22 * semitone + float(phone_offset)
    vector = pca.mean + coefficient * pca.components[0]
    rms = float(np.sqrt(np.mean(vector ** 2)))
    return vector / max(rms, 1e-12)


def _residual_example(index: int, spec: FeatureSpec,
                      pca: SourcePCA,
                      phones: tuple[str, ...] = ("a", "i", "a"),
                      frames_per_phone: int = 20) -> SourceTrainingExample:
    hop = max(1, int(round(spec.fs * spec.frame_period / 1000.0)))
    n_frames = len(phones) * frames_per_phone
    base_f0 = np.linspace(170.0 + 12.0 * index, 430.0 + 9.0 * index, n_frames)
    phone_offsets = {"a": 0.0, "i": 0.15, "s": -0.1}
    vectors = np.asarray([
        _unit_vector(pca, f0, phone_offsets.get(phones[t // frames_per_phone], 0.0))
        for t, f0 in enumerate(base_f0)
    ])
    epochs = np.arange(n_frames, dtype=np.int64) * hop
    periods = np.full(n_frames, hop, dtype=np.int64)
    n_samples = n_frames * hop
    source = SourceSequence(
        f0=base_f0, voiced=np.ones(n_frames, dtype=bool),
        noise_level=np.zeros(n_frames), excitation=vectors,
        gains=np.ones(n_frames), epochs=epochs, periods=periods,
        cycle_length=pca.cycle_length, fs=spec.fs,
        frame_period=spec.frame_period, n_samples=n_samples,
        backend="residual")
    segments = [
        Segment(phone, start=i * frames_per_phone * spec.frame_period / 1000.0,
                end=(i + 1) * frames_per_phone * spec.frame_period / 1000.0,
                note=60.0, utterance=f"utt-{index}")
        for i, phone in enumerate(phones)
    ]
    return SourceTrainingExample(Utterance(f"utt-{index}", segments), source,
                                 base_f0)


def _voice_example(index: int, spec: FeatureSpec, pca: SourcePCA,
                   n_frames: int = 80) -> SourceTrainingExample:
    hop = max(1, int(round(spec.fs * spec.frame_period / 1000.0)))
    f0 = np.full(n_frames, 220.0 + index * 8.0)
    n_samples = n_frames * hop
    period = int(round(spec.fs / float(f0[0])))
    epochs = np.arange(0, n_samples - period, period, dtype=np.int64)
    periods = np.full(len(epochs), period, dtype=np.int64)
    vectors = np.asarray([_unit_vector(pca, float(f0[0]), 0.03 * j)
                          for j in range(len(epochs))]).reshape(-1, pca.cycle_length)
    source = SourceSequence(
        f0=f0, voiced=np.ones(n_frames, dtype=bool),
        noise_level=np.zeros(n_frames), excitation=vectors,
        gains=np.ones(len(epochs)), epochs=epochs, periods=periods,
        cycle_length=pca.cycle_length, fs=spec.fs,
        frame_period=spec.frame_period, n_samples=n_samples,
        backend="voice")
    utterance = Utterance(f"voice-{index}", [
        Segment("a", 0.0, n_frames * spec.frame_period / 1000.0,
                note=60.0, utterance=f"voice-{index}")])
    return SourceTrainingExample(utterance, source, f0)


def _trainer(spec: FeatureSpec, context: bool = False) -> SourceTrainer:
    config = SourceTrainingConfig(
        pca_components=3, n_iterations=2, min_phone_frames=3,
        context_enabled=context, context_min_frames=8,
        context_min_occurrences=2, context_max_models=16,
        context_partial=True, source_f0_floor=50.0,
        source_f0_ceil=2000.0)
    return SourceTrainer(spec=spec, phoneme_set=_phonemes(), config=config)


def test_source_alignment_overlap_resampling_matches_phase1_unit_spans():
    spec = _spec(fs=20000)
    hop = 100
    source = SourceSequence(
        f0=np.full(6, 220.0), voiced=np.ones(6, dtype=bool),
        noise_level=np.zeros(6), excitation=np.ones((3, 8)),
        gains=np.ones(3), epochs=np.array([50, 150, 350]),
        periods=np.array([100, 200, 100]), cycle_length=8,
        fs=spec.fs, frame_period=spec.frame_period, n_samples=600,
        backend="voice")
    aligned, valid = align_source_coefficients(
        source, np.array([[1.0], [3.0], [5.0]]),
        f0_hz=np.array([220, 220, 220, 220, 220, 0]),
        voiced_threshold=spec.voiced_threshold)
    assert aligned[:, 0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    assert valid.tolist() == [True, True, True, True, True, False]


def test_source_alignment_requires_exact_f0_frame_count():
    source = SourceSequence(
        f0=np.full(4, 220.0), voiced=np.ones(4, dtype=bool),
        noise_level=np.zeros(4), excitation=np.ones((4, 8)),
        gains=np.ones(4), epochs=np.arange(4) * 100,
        periods=np.full(4, 100), cycle_length=8, fs=20000,
        frame_period=5.0, n_samples=400, backend="voice")
    with pytest.raises(ValueError, match="expected exactly 4"):
        align_source_coefficients(source, np.ones((4, 2)), np.ones(3) * 220)


def test_f0_condition_features_reuse_semitones_deltas_and_voicing():
    spec = _spec()
    f0 = np.array([220.0, 440.0, 440.0, 0.0, 110.0])
    features = f0_condition_features(f0, spec, mean_semitones=0.0,
                                     scale_semitones=1.0, delta_scale=1.0)
    assert features.shape == (5, 3)
    assert features[:, 0] == pytest.approx([0.0, 12.0, 12.0, 0.0, -12.0])
    assert features[:, 2].tolist() == [1.0, 1.0, 1.0, 0.0, 1.0]
    # The voiced run's delta follows the existing HMS regression-window rule;
    # an unvoiced gap is neither logged nor used to create a huge pitch jump.
    assert np.isfinite(features).all()
    assert features[3, :2].tolist() == [0.0, 0.0]
    assert features[1, 1] > 0.0
    assert features[4, 1] == 0.0
    zero_threshold = _spec()
    zero_threshold.voiced_threshold = 0.0
    zero_threshold_features = f0_condition_features(
        np.array([0.0, 220.0]), zero_threshold)
    assert zero_threshold_features[:, 2].tolist() == [0.0, 1.0]


def test_tiny_source_training_uses_phase1_pca_and_existing_hmm_gmm():
    spec = _spec()
    pca = _manual_pca()
    examples = [_residual_example(i, spec, pca) for i in range(4)]
    model = _trainer(spec, context=False).fit(examples, pca=pca, name="tiny")

    assert model.pca is pca                         # supplied Phase-1 basis reused
    assert model.source_backend == "residual"
    assert set(model.hmms) >= {"a", "i"}
    assert model.feature_dim == pca.n_components * len(model.stream_sizes)
    assert model.hmms["a"].states[0].gmm.dim == model.feature_dim
    assert model.condition_feature_dim == 3
    assert model.training["source_units"] == sum(x.source.n_units for x in examples)


def test_sparse_context_keeps_gmm_dimension_in_pca_stream_space():
    spec = _spec()
    pca = _manual_pca()
    examples = [_residual_example(i, spec, pca) for i in range(4)]
    model = _trainer(spec, context=True).fit(examples, pca=pca)

    assert model.contexts
    unit_id, hmm, tier = model.resolve_unit("a", "i", "a")
    assert unit_id == ("context", "a^i^a")
    assert tier == "triphone"
    assert hmm is model.contexts["a^i^a"]
    assert model.context_feature_dim == 3  # previous/current/next symbols
    # Context selects an HMM; it is not concatenated to the PCA observation.
    assert hmm.states[0].gmm.dim == pca.n_components * len(model.stream_sizes)


def test_pitch_changes_within_a_phoneme_condition_generated_coefficients():
    spec = _spec()
    pca = _manual_pca()
    examples = [_residual_example(i, spec, pca, phones=("a",),
                                  frames_per_phone=48) for i in range(6)]
    model = _trainer(spec).fit(examples, pca=pca)
    utterance = examples[0].utterance
    low = model.generate(utterance, np.full(48, 180.0))
    high = model.generate(utterance, np.full(48, 440.0))
    glide = model.generate(utterance, np.linspace(180.0, 440.0, 48))

    assert low.frame_coefficients.shape == (48, pca.n_components)
    assert high.frame_coefficients.shape == low.frame_coefficients.shape
    assert np.mean(np.abs(low.frame_coefficients - high.frame_coefficients)) > 1e-3
    assert np.std(glide.frame_coefficients[:, 0]) > 1e-3
    assert glide.voiced.all() and glide.active.all()


def test_voice_backend_masks_unvoiced_frames_and_does_not_predict_fake_cycles():
    spec = _spec()
    pca = _manual_pca()
    examples = [_voice_example(i, spec, pca) for i in range(4)]
    model = _trainer(spec).fit(examples, pca=pca)
    f0 = np.concatenate([np.full(40, 220.0), np.zeros(40)])
    prediction = model.generate(examples[0].utterance, f0)

    assert prediction.voiced.tolist() == [True] * 40 + [False] * 40
    assert prediction.active.tolist() == [True] * 40 + [False] * 40
    assert np.all(prediction.frame_coefficients[40:] == 0.0)
    assert prediction.sequence.n_units > 0
    assert np.all(prediction.sequence.epochs + prediction.sequence.periods
                   <= 40 * prediction.sequence.hop)
    assert np.all(prediction.sequence.f0[40:] == 0.0)


def test_generic_residual_model_keeps_unvoiced_source_units_explicit():
    spec = _spec()
    pca = _manual_pca()
    examples = [_residual_example(i, spec, pca, phones=("a",),
                                  frames_per_phone=32) for i in range(4)]
    model = _trainer(spec).fit(examples, pca=pca)
    f0 = np.concatenate([np.full(16, 220.0), np.zeros(16)])
    prediction = model.generate(examples[0].utterance, f0)

    assert prediction.voiced.tolist() == [True] * 16 + [False] * 16
    assert prediction.active.all()  # frame-residual source also models unvoiced noise
    assert len(prediction.sequence.f0) == len(f0)
    assert np.all(prediction.sequence.f0[16:] == 0.0)
    assert prediction.sequence.n_units == len(f0)


def test_residual_units_do_not_straddle_labelled_silence():
    spec = _spec()
    pca = _manual_pca()
    raw = _residual_example(0, spec, pca, phones=("a", "sil", "a"),
                            frames_per_phone=16)
    source = raw.source
    unit_frames = 4
    hop = source.hop
    starts = np.arange(0, source.n_frames, unit_frames, dtype=np.int64)
    epochs = starts * hop
    periods = np.minimum(unit_frames * hop, source.n_samples - epochs)
    vectors = source.excitation[np.minimum(starts + unit_frames // 2,
                                           source.n_frames - 1)]
    grouped = SourceSequence(
        f0=source.f0, voiced=source.voiced, noise_level=source.noise_level,
        excitation=vectors, gains=np.ones(len(epochs)), epochs=epochs,
        periods=periods, cycle_length=source.cycle_length, fs=source.fs,
        frame_period=source.frame_period, n_samples=source.n_samples,
        backend="residual")
    example = SourceTrainingExample(raw.utterance, grouped, raw.f0_hz)
    model = _trainer(spec).fit([example], pca=pca)
    assert model.unit_frames == unit_frames

    prediction = model.generate(raw.utterance, np.full(source.n_frames, 220.0))
    silence_start, silence_stop = 16 * hop, 32 * hop
    unit_stops = prediction.sequence.epochs + prediction.sequence.periods
    assert np.all((unit_stops <= silence_start)
                  | (prediction.sequence.epochs >= silence_stop))
    backend = get_source_model(
        "residual", cycle_length=pca.cycle_length, fs=spec.fs,
        frame_period=spec.frame_period, n_mcep=spec.n_mcep)
    rendered = backend.synthesize(prediction.sequence, pca=model.pca,
                                  restore_gain=False)
    assert np.count_nonzero(rendered[silence_start:silence_stop]) == 0


def test_generation_is_deterministic_and_decodes_through_phase1_source_api():
    spec = _spec()
    pca = _manual_pca()
    examples = [_residual_example(i, spec, pca) for i in range(4)]
    model = _trainer(spec, context=True).fit(examples, pca=pca)
    f0 = np.linspace(200.0, 390.0, examples[0].source.n_frames)
    first = model.generate(examples[0].utterance, f0)
    second = model.generate(examples[0].utterance, f0)

    assert first.frame_coefficients.shape == (len(f0), pca.n_components)
    assert first.frame_coefficients == pytest.approx(second.frame_coefficients)
    assert first.sequence.coefficients == pytest.approx(second.sequence.coefficients)
    decoded = pca.decode(first.sequence.coefficients)
    assert decoded == pytest.approx(first.sequence.excitation)

    backend = get_source_model("residual", cycle_length=pca.cycle_length,
                               fs=spec.fs, frame_period=spec.frame_period,
                               n_mcep=spec.n_mcep)
    excitation = backend.synthesize(first.sequence, pca=pca,
                                    restore_gain=False)
    assert excitation.shape == (first.sequence.n_samples,)
    assert np.isfinite(excitation).all()
    assert np.any(np.abs(excitation) > 0.0)


def test_source_model_yaml_npz_roundtrip_preserves_prediction(tmp_path):
    spec = _spec()
    pca = _manual_pca()
    examples = [_residual_example(i, spec, pca) for i in range(4)]
    model = _trainer(spec, context=True).fit(examples, pca=pca, name="singer")
    f0 = np.linspace(190.0, 420.0, examples[0].source.n_frames)
    before = model.generate(examples[0].utterance, f0)

    model.save(tmp_path / "singer-source")
    loaded = SourceHMMModel.load(tmp_path / "singer-source")
    after = loaded.generate(examples[0].utterance, f0)

    assert loaded.name == "singer"
    assert loaded.pca.n_components == pca.n_components
    assert loaded.pca.cycle_length == pca.cycle_length
    assert loaded.context_index == model.context_index
    assert before.frame_coefficients == pytest.approx(after.frame_coefficients)
    assert before.sequence.excitation == pytest.approx(after.sequence.excitation)
    assert (tmp_path / "singer-source" / "source.yaml").exists()
    assert (tmp_path / "singer-source" / "source_hmms.npz").exists()
    assert (tmp_path / "singer-source" / "source_pca.npz").exists()


def test_generation_uses_the_explicit_f0_length_and_rejects_invalid_values():
    spec = _spec()
    pca = _manual_pca()
    examples = [_residual_example(i, spec, pca, phones=("a",),
                                  frames_per_phone=24) for i in range(3)]
    model = _trainer(spec).fit(examples, pca=pca)
    utterance = examples[0].utterance
    truncated = model.generate(utterance, np.full(23, 220.0))
    assert len(truncated.f0_hz) == len(truncated.frame_coefficients) == 23
    bad = np.full(24, 220.0)
    bad[7] = np.nan
    with pytest.raises(ValueError, match="frame 7"):
        model.generate(utterance, bad)
    bad[7] = -1.0
    with pytest.raises(ValueError, match="frame 7 is negative"):
        model.generate(utterance, bad)


def test_phase1_analyzer_to_source_model_generation_end_to_end():
    spec = _spec()
    analyzer = get_source_model(
        "residual", cycle_length=16, fs=spec.fs,
        frame_period=spec.frame_period, fft_size=512,
        n_mcep=spec.n_mcep, unit_frames=1)
    audio = np.random.default_rng(9).normal(0.0, 0.05, 2200)
    f0 = np.linspace(190.0, 330.0, 20)
    utterance = Utterance("audio", [Segment("a", 0.0, 0.1, note=60.0)])
    example = SourceTrainingExample.from_audio(
        utterance, audio, analyzer, fs=spec.fs,
        frame_period=spec.frame_period, f0_hz=f0)
    model = _trainer(spec).fit([example], name="audio-source")
    prediction = model.generate(utterance, f0)
    excitation = analyzer.synthesize(prediction.sequence, pca=model.pca,
                                     restore_gain=False)

    assert prediction.frame_coefficients.shape == (20, model.pca.n_components)
    assert prediction.f0_hz == pytest.approx(f0)
    assert prediction.sequence.excitation.shape[1] == 16
    assert excitation.shape == audio.shape
    assert np.isfinite(excitation).all()


def test_training_can_fit_the_existing_phase1_pca_when_not_supplied():
    spec = _spec()
    examples = [_residual_example(i, spec, _manual_pca()) for i in range(3)]
    config = SourceTrainingConfig(pca_components=5, n_iterations=1,
                                  min_phone_frames=3)
    model = SourceTrainer(spec, _phonemes(), config).fit(examples)
    assert model.pca.n_components == 5
    assert model.pca.cycle_length == examples[0].source.cycle_length
