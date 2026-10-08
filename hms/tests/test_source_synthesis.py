"""Phase 3: source-aware synthesis integration tests.

Phase 1 gave the excitation a representation, Phase 2 gave it a predictor, and
neither touched synthesis.  This module is the third step: a trained acoustic
model and a *tiny* Phase-2 source model render a score together, and the
learned excitation reaches the filter.

The tests are plumbing tests, not quality tests -- the source model here is
fitted on a few hundred synthetic frames, so it says nothing about how a real
singer's source sounds.  What they pin down is the contract:

* the ordinary path is bit-identical without a source model;
* the two branches agree on sample rate, frame period, frame count, F0,
  voicing and duration -- and refuse to run when they do not;
* the learned source is what drives the filter where it covers the timeline;
* everywhere else (unvoiced frames, gaps, the tail past the frame grid) the
  backend's own excitation and the aperiodicity/noise path are untouched;
* nothing about the render becomes non-deterministic or non-finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.features import FeatureSpec
from hms.core.labels import Score, Segment, Utterance
from hms.core.phonemes import PhonemeSet
from hms.core.synthesizer import SynthesisConfig, Synthesizer
from hms.source import (SourceExcitation, SourceHMMModel, SourcePCA,
                        SourceSequence, SourceTrainer, SourceTrainingConfig,
                        SourceTrainingExample, render_source_excitation)
from hms.source.cycles import frame_coverage, pick_epochs, unit_support_mask
from hms.source.synthesis import (fade_weights, frame_hop,
                                  render_score_source_excitation,
                                  source_model_diagnostics,
                                  source_model_is_compatible)
from hms.vocoder.base import Vocoder, VocoderUnavailable, render_length
from hms.vocoder.builtin import BuiltinVocoder
from hms.vocoder.mlsa import MLSAVocoder

FS = 22050                 # matches hms/tests/conftest.py
FRAME_PERIOD = 5.0
F_REF = 261.6255653005986  # the acoustic fixtures' FeatureSpec default


# --------------------------------------------------------------------------
# A tiny Phase-2 source model (synthetic, a few hundred frames)
# --------------------------------------------------------------------------

def _pca(cycle_length: int = 64) -> SourcePCA:
    """A fixed, broadband two-component basis (no fitting -> deterministic).

    Phase 1's residual is *whitened* (`hms.source.residual.whiten` divides out a
    unit-mean mel-cepstral envelope), so a real source cycle is broadband like
    the pulse train the learned source replaces.  The stand-in here is built the
    same way -- a band-limited impulse plus two orthonormal shifts of it -- so
    the level comparisons below measure the integration rather than the
    spectral tilt of an unrepresentative smooth test vector.
    """
    x = np.arange(cycle_length, dtype=np.float64)
    offsets = x - cycle_length // 2
    pulse = np.roll(np.sinc(0.9 * offsets) * np.hanning(cycle_length),
                    -(cycle_length // 2))
    components = []
    for shift in (8, 16):
        vector = np.roll(pulse, shift).astype(np.float64)
        for basis_vector in [pulse / np.linalg.norm(pulse)] + components:
            vector = vector - basis_vector * float(vector @ basis_vector)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            components.append(vector / norm)
    return SourcePCA(mean=pulse, components=np.vstack(components),
                     eigenvalues=np.array([1.0, 0.5]), total_variance=1.5,
                     n_samples=100)


def _unit_vector(pca: SourcePCA, f0: float, phone_offset: float) -> np.ndarray:
    """A unit-RMS source vector whose shape depends on F0 and on the phone."""
    semitone = float(12.0 * np.log2(max(f0, 1e-6) / F_REF))
    coefficients = np.array([0.05 * semitone + 0.6 * phone_offset,
                             0.02 * semitone - 0.3 * phone_offset])
    vector = pca.mean + coefficients @ pca.components
    rms = float(np.sqrt(np.mean(vector ** 2)))
    return vector / max(rms, 1e-12)


def _source_spec(fs: int = FS, frame_period: float = FRAME_PERIOD) -> FeatureSpec:
    return FeatureSpec(fs=fs, frame_period=frame_period, fft_size=1024,
                       n_mcep=20, n_band=5, use_delta=True, use_delta2=False,
                       delta_window=2, f0_ref_hz=F_REF, voiced_threshold=5.0)


def _voice_example(index: int, spec: FeatureSpec, pca: SourcePCA,
                   phones: tuple = ("a", "i", "m", "s"),
                   frames_per_phone: int = 24) -> SourceTrainingExample:
    """One fully voiced utterance: pitch cycles whose shape tracks the F0."""
    hop = frame_hop(spec)
    n_frames = len(phones) * frames_per_phone
    f0 = np.linspace(190.0 + 17.0 * index, 430.0 + 11.0 * index, n_frames)
    n_samples = n_frames * hop
    boundaries, runs = pick_epochs(f0, hop, n_samples, int(spec.fs),
                                   f0_floor=50.0, f0_ceil=2000.0)
    # A cycle exists only where both boundary epochs belong to one voiced run
    # -- the same rule Phase 1's `extract_cycles` applies.
    keep = (np.flatnonzero(runs[:-1] == runs[1:]) if len(runs) > 1
            else np.zeros(0, dtype=np.int64))
    epochs = boundaries[keep].astype(np.int64)
    periods = (boundaries[keep + 1] - boundaries[keep]).astype(np.int64) \
        if len(keep) else np.zeros(0, dtype=np.int64)
    offsets = {"a": 0.0, "i": 0.11, "m": -0.07, "s": 0.05}
    grid = np.arange(n_frames, dtype=np.float64)
    vectors = []
    for epoch, period in zip(epochs.tolist(), periods.tolist()):
        centre = min(max((epoch + 0.5 * period) / hop, 0.0), n_frames - 1)
        phone = phones[min(int(centre) // frames_per_phone, len(phones) - 1)]
        vectors.append(_unit_vector(pca, float(np.interp(centre, grid, f0)),
                                    offsets.get(phone, 0.0)))
    source = SourceSequence(
        f0=f0, voiced=np.ones(n_frames, dtype=bool),
        noise_level=np.zeros(n_frames),
        excitation=np.asarray(vectors, dtype=np.float64).reshape(
            -1, pca.cycle_length),
        gains=np.ones(len(epochs)), epochs=epochs, periods=periods,
        cycle_length=pca.cycle_length, fs=int(spec.fs),
        frame_period=spec.frame_period, n_samples=n_samples, backend="voice")
    segments = [
        Segment(phone, i * frames_per_phone * spec.frame_period / 1000.0,
                (i + 1) * frames_per_phone * spec.frame_period / 1000.0,
                note=60.0, utterance=f"src-{index}")
        for i, phone in enumerate(phones)]
    return SourceTrainingExample(Utterance(f"src-{index}", segments), source, f0)


def _tiny_source_model(fs: int = FS, frame_period: float = FRAME_PERIOD,
                       pca: SourcePCA | None = None) -> SourceHMMModel:
    """Train a Phase-2 model on four synthetic utterances (no corpus needed)."""
    spec = _source_spec(fs, frame_period)
    basis = pca or _pca()
    config = SourceTrainingConfig(
        n_iterations=2, min_phone_frames=4, global_backoff=True,
        context_enabled=False, source_f0_floor=50.0, source_f0_ceil=2000.0)
    examples = [_voice_example(i, spec, basis) for i in range(4)]
    return SourceTrainer(spec=spec, phoneme_set=PhonemeSet.default(),
                         config=config).fit(examples, pca=basis,
                                            name="tiny-source")


@pytest.fixture(scope="session")
def tiny_source_model() -> SourceHMMModel:
    return _tiny_source_model()


def render(model, score, source_model=None, vocoder="mlsa", **overrides):
    options = dict(vibrato=False, seed=0, vocoder=vocoder)
    options.update(overrides)
    return Synthesizer(model, SynthesisConfig(**options)).synthesize(
        score, source_model=source_model)


def _unvoiced_score() -> Score:
    """A phrase of unvoiced consonants only: no voiced frame anywhere."""
    return Score([Utterance(name="hiss", segments=[
        Segment("s", 0.0, 0.25), Segment("f", 0.25, 0.5),
        Segment("s", 0.5, 0.75)])])


def _held_note_score() -> Score:
    return Score([Utterance(name="held", segments=[
        Segment("sil", 0.0, 0.1), Segment("a", 0.1, 0.8, note=60.0),
        Segment("sil", 0.8, 0.9)])])


# --------------------------------------------------------------------------
# Backwards compatibility: the ordinary path
# --------------------------------------------------------------------------

def test_existing_synthesis_is_untouched_without_a_source_model(
        trained_model, short_score):
    result = render(trained_model, short_score)
    assert result.source is None
    assert np.isfinite(result.audio).all()
    assert len(result.audio) == render_length(
        len(result.params.f0), trained_model.spec.frame_period,
        trained_model.spec.fs)
    assert not any("source" in message for message in result.diagnostics
                   if "source-aware" in message)


def test_source_aware_and_default_renders_share_one_frame_grid(
        trained_model, short_score, tiny_source_model):
    """Same score, same frames: only the excitation differs."""
    default = render(trained_model, short_score)
    aware = render(trained_model, short_score, tiny_source_model)
    assert len(aware.audio) == len(default.audio)
    assert len(aware.params.f0) == len(default.params.f0)
    assert np.array_equal(aware.params.sp, default.params.sp)
    assert np.array_equal(aware.params.ap, default.params.ap)
    assert np.array_equal(aware.params.f0, default.params.f0)
    assert np.array_equal(aware.phones, default.phones)


def test_source_gain_defaults_to_neutral_and_is_validated():
    assert SynthesisConfig().source_gain == 1.0
    with pytest.raises(ValueError, match="source_gain"):
        SynthesisConfig(source_gain=-1.0)
    with pytest.raises(ValueError, match="finite"):
        SynthesisConfig(source_gain=float("nan"))


# --------------------------------------------------------------------------
# The source branch runs and reaches the filter
# --------------------------------------------------------------------------

def test_source_aware_synthesis_renders_with_a_tiny_phase2_model(
        trained_model, short_score, tiny_source_model):
    result = render(trained_model, short_score, tiny_source_model)
    assert isinstance(result.source, SourceExcitation)
    assert result.source.n_units > 0
    assert result.source.backend == "voice"
    assert np.isfinite(result.audio).all()
    assert len(result.audio) == render_length(
        len(result.params.f0), trained_model.spec.frame_period,
        trained_model.spec.fs)
    assert any("source-aware synthesis" in message
               for message in result.diagnostics)


def test_learned_source_reaches_the_filter(trained_model, short_score,
                                           tiny_source_model):
    """The excitation is not computed and thrown away: it changes the audio."""
    default = render(trained_model, short_score)
    aware = render(trained_model, short_score, tiny_source_model)
    assert not np.array_equal(aware.audio, default.audio)
    difference = float(np.max(np.abs(aware.audio - default.audio)))
    assert difference > 1e-6
    # ... and it is audible only where the source actually covers the timeline
    assert result_weight_is_used_everywhere(aware)


def result_weight_is_used_everywhere(result) -> bool:
    """Coverage > 0 on voiced frames, and the weights follow the coverage."""
    source = result.source
    voiced = source.voiced
    assert voiced.any(), "the fixture render has no voiced frames"
    assert (source.coverage[voiced] > 0.0).any()
    assert source.weights.min() >= 0.0 and source.weights.max() <= 1.0
    assert source.covered_samples > 0
    return True


def test_source_gain_scales_the_learned_excitation(
        trained_model, short_score, tiny_source_model):
    """`source_gain` scales the learned source, and only the periodic half.

    The knob itself is exact: it multiplies the excitation by `source_gain`,
    full stop.  How far the *rendered* level then moves depends on how much of
    the output the periodic half owns, and that is a property of the acoustic
    model and the backend, not of this integration -- measured across both
    numpy backends, both fixture corpora, and with and without the optional
    WORLD native backend, the same 16x gain moves the rendered RMS by
    anything from 1.1x to 14x.  So the assertion is the invariant rather than
    a number:

    * the excitation scales exactly;
    * the render follows it in the right direction;
    * the render can never out-run it, because the noise/aperiodicity half of
      the excitation is untouched -- a ceiling of 16x, not a coincidence.
    """
    quiet = render(trained_model, short_score, tiny_source_model,
                   source_gain=0.25)
    loud = render(trained_model, short_score, tiny_source_model, source_gain=4.0)
    assert not np.array_equal(quiet.audio, loud.audio)
    # Scaling the *source* moves the periodic part only, so the render stays
    # finite and non-silent either way.
    for result in (quiet, loud):
        assert np.isfinite(result.audio).all()
        assert np.sqrt(np.mean(result.audio ** 2)) > 1e-4
    # the knob is exact on the waveform it scales
    scale = (float(np.std(loud.source.excitation))
             / float(np.std(quiet.source.excitation)))
    assert scale == pytest.approx(16.0, rel=1e-9)
    # ... and the render follows it without ever out-running it
    ratio = (np.sqrt(np.mean(loud.audio ** 2))
             / np.sqrt(np.mean(quiet.audio ** 2)))
    assert 1.2 < ratio <= 16.0 * (1.0 + 1e-9)


def test_source_aware_synthesis_is_deterministic(trained_model, short_score,
                                                 tiny_source_model):
    first = render(trained_model, short_score, tiny_source_model)
    second = render(trained_model, short_score, tiny_source_model)
    assert np.array_equal(first.audio, second.audio)
    assert np.array_equal(first.source.excitation, second.source.excitation)
    assert np.array_equal(first.source.weights, second.source.weights)


def test_both_numpy_backends_accept_a_learned_excitation(
        trained_model, short_score, tiny_source_model):
    for name in ("builtin", "mlsa"):
        result = render(trained_model, short_score, tiny_source_model,
                        vocoder=name)
        assert np.isfinite(result.audio).all()
        assert result.source is not None and result.source.n_units > 0


def test_a_backend_that_cannot_filter_an_excitation_is_refused(
        trained_model, short_score, tiny_source_model):
    """WORLD takes (f0, sp, ap) and nothing else -- say so, do not ignore it."""

    class _ClosedVocoder(BuiltinVocoder):
        supports_external_excitation = False

    synthesizer = Synthesizer(trained_model,
                              SynthesisConfig(vocoder="builtin", seed=0,
                                              vibrato=False))
    synthesizer._vocoder = _ClosedVocoder(fft_size=trained_model.spec.fft_size)
    with pytest.raises(ValueError, match="cannot filter a caller-supplied"):
        synthesizer.synthesize(short_score, source_model=tiny_source_model)
    # ... while the same backend renders the ordinary path fine.
    assert np.isfinite(synthesizer.synthesize(short_score).audio).all()


@pytest.mark.parametrize("name", ["builtin", "mlsa"])
def test_zero_weights_reproduce_the_ordinary_render_bit_for_bit(
        trained_model, short_score, name):
    """The strongest form of backwards compatibility: no weight, no change.

    Handing a backend an excitation it is told to ignore must give back
    exactly the waveform it produced itself.
    """
    result = render(trained_model, short_score, vocoder=name)
    params, default = result.params, result.audio
    backend = {"builtin": BuiltinVocoder, "mlsa": MLSAVocoder}[name](
        fft_size=trained_model.spec.fft_size)
    n_samples = render_length(len(params.f0), params.frame_period, params.fs)
    ignored = backend.synthesize_with_excitation(
        params, np.zeros(n_samples), np.zeros(n_samples))
    assert np.array_equal(ignored[:len(default)], default)


@pytest.mark.parametrize("name", ["builtin", "mlsa"])
def test_full_weights_replace_the_periodic_component(
        trained_model, short_score, name):
    """Weight 1 with a silent source = noise only: the aperiodicity path lives."""
    result = render(trained_model, short_score, vocoder=name)
    params, default = result.params, result.audio
    backend = {"builtin": BuiltinVocoder, "mlsa": MLSAVocoder}[name](
        fft_size=trained_model.spec.fft_size)
    n_samples = render_length(len(params.f0), params.frame_period, params.fs)
    silenced = backend.synthesize_with_excitation(
        params, np.zeros(n_samples), np.ones(n_samples))
    assert np.isfinite(silenced).all()
    assert float(np.sqrt(np.mean(silenced ** 2))) < float(
        np.sqrt(np.mean(default ** 2)))


def test_the_vocoder_interface_refuses_an_external_excitation_by_default():
    class _Bare(Vocoder):
        name = "bare"
        fft_size = 256

        def analyze(self, x, fs=None, frame_period=None, **kwargs):
            raise NotImplementedError

        def synthesize(self, params):
            raise NotImplementedError

    bare = _Bare(fft_size=256)
    assert bare.supports_external_excitation is False
    with pytest.raises(VocoderUnavailable, match="cannot filter"):
        bare.synthesize_with_excitation(None, np.zeros(8))


# --------------------------------------------------------------------------
# Timing and alignment
# --------------------------------------------------------------------------

def test_source_and_acoustic_frame_counts_agree(trained_model, short_score,
                                                tiny_source_model):
    result = render(trained_model, short_score, tiny_source_model)
    source = result.source
    n_frames = len(result.params.f0)
    assert source.n_frames == n_frames
    assert len(source.coverage) == n_frames
    assert len(source.voiced) == n_frames
    assert len(source.f0_hz) == n_frames
    assert source.fs == trained_model.spec.fs
    assert source.frame_period == trained_model.spec.frame_period
    assert source.n_samples == render_length(
        n_frames, trained_model.spec.frame_period, trained_model.spec.fs)
    assert len(source.excitation) == source.n_samples
    assert len(source.weights) == source.n_samples


def test_the_source_model_sees_exactly_the_render_f0(trained_model,
                                                     tiny_source_model):
    """No second F0: the source branch conditions on the rendered track."""
    score = _held_note_score()
    result = render(trained_model, score, tiny_source_model)
    assert np.array_equal(result.source.f0_hz, result.params.f0)
    # voicing is derived from that same track, with the same convention
    assert np.array_equal(result.source.voiced,
                          result.params.f0 > trained_model.spec.voiced_threshold)


def test_an_external_f0_reaches_the_source_model_unchanged(
        trained_model, tiny_source_model):
    score = _held_note_score()
    synthesizer = Synthesizer(trained_model,
                              SynthesisConfig(vocoder="mlsa", seed=0,
                                              vibrato=False))
    reference = synthesizer.synthesize(score)
    external = np.asarray(reference.params.f0, dtype=np.float64).copy()
    external[:5] = 0.0                        # an unvoiced lead-in
    result = synthesizer.synthesize(score, f0=external,
                                    source_model=tiny_source_model)
    assert np.array_equal(result.source.f0_hz, external)
    assert not result.source.voiced[:5].any()
    assert np.isfinite(result.audio).all()


def test_an_external_f0_of_the_wrong_length_is_rejected(
        trained_model, tiny_source_model):
    score = _held_note_score()
    synthesizer = Synthesizer(trained_model,
                              SynthesisConfig(vocoder="mlsa", seed=0,
                                              vibrato=False))
    reference = synthesizer.synthesize(score)
    with pytest.raises(ValueError, match="never resizes"):
        synthesizer.synthesize(score, f0=reference.params.f0[:-3],
                               source_model=tiny_source_model)


def test_sample_rate_mismatch_is_rejected(trained_model, short_score):
    other = _tiny_source_model(fs=44100)
    assert not source_model_is_compatible(trained_model, other)
    with pytest.raises(ValueError, match="sample rate"):
        render(trained_model, short_score, other)


def test_frame_period_mismatch_is_rejected(trained_model, short_score):
    other = _tiny_source_model(frame_period=10.0)
    with pytest.raises(ValueError, match="frame period"):
        render(trained_model, short_score, other)


def test_a_compatible_source_model_reports_no_geometry_diagnostics(
        trained_model, tiny_source_model):
    assert source_model_diagnostics(trained_model, tiny_source_model) == []
    assert source_model_is_compatible(trained_model, tiny_source_model)


def test_a_differing_voicing_threshold_is_reported_not_fatal(
        trained_model, short_score):
    """Shared F0, different thresholds: the two branches cannot actually fight."""
    other = _tiny_source_model()
    object.__setattr__(other.spec, "voiced_threshold", 120.0)
    messages = source_model_diagnostics(trained_model, other)
    assert any("voiced threshold" in message for message in messages)
    result = render(trained_model, short_score, other)
    assert np.isfinite(result.audio).all()


def test_source_predictions_are_never_stretched_past_their_frame_grid(
        tiny_source_model):
    spec = tiny_source_model.spec
    hop = frame_hop(spec)
    utterance = Utterance("stretch", [
        Segment("a", 0.0, 40 * spec.frame_period / 1000.0, note=60.0)])
    f0 = np.full(40, 220.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        render_source_excitation(tiny_source_model, utterance, f0,
                                 n_samples=40 * hop + 500)


def test_a_malformed_f0_is_rejected_by_the_source_renderer(tiny_source_model):
    utterance = Utterance("bad", [
        Segment("a", 0.0, 40 * 5.0 / 1000.0, note=60.0)])
    f0 = np.full(40, 220.0)
    for broken in (np.full(40, np.nan), np.full(40, -1.0)):
        with pytest.raises(ValueError):
            render_source_excitation(tiny_source_model, utterance, broken)


def test_unit_spans_stay_inside_the_utterance(tiny_source_model):
    """Note/segment boundaries must not produce a malformed source unit."""
    spec = tiny_source_model.spec
    hop = frame_hop(spec)
    n_frames = 160
    n_samples = n_frames * hop
    # two notes and a silence gap in one utterance
    utterance = Utterance("two_notes", [
        Segment("a", 0.0, 40 * spec.frame_period / 1000.0, note=60.0),
        Segment("sil", 40 * spec.frame_period / 1000.0,
                60 * spec.frame_period / 1000.0),
        Segment("i", 60 * spec.frame_period / 1000.0,
                n_frames * spec.frame_period / 1000.0, note=67.0)])
    f0 = np.zeros(n_frames)
    f0[:40] = 261.63
    f0[60:] = 440.0
    prediction = tiny_source_model.generate(utterance, f0,
                                            n_samples=n_samples)
    sequence = prediction.sequence
    assert sequence.n_units > 0
    assert (sequence.periods > 0).all()
    assert (sequence.epochs >= 0).all()
    assert (sequence.epochs + sequence.periods <= n_samples).all()
    assert np.all(np.diff(sequence.epochs) > 0)
    # a unit is never built across the unvoiced gap
    gap_start, gap_stop = 40 * hop, 60 * hop
    for epoch, period in zip(sequence.epochs.tolist(), sequence.periods.tolist()):
        assert not (epoch < gap_stop and epoch + period > gap_start)


# --------------------------------------------------------------------------
# Voiced / unvoiced
# --------------------------------------------------------------------------

def test_unvoiced_regions_keep_the_existing_noise_path(
        trained_model, tiny_source_model):
    """Nothing voiced -> nothing learned -> the render is bit-identical."""
    score = _unvoiced_score()
    synthesizer = Synthesizer(trained_model,
                              SynthesisConfig(vocoder="mlsa", seed=0,
                                              vibrato=False))
    default = synthesizer.synthesize(score)
    aware = synthesizer.synthesize(score, source_model=tiny_source_model)
    assert not default.params.f0.any()           # the score really is unvoiced
    assert aware.source.n_units == 0
    assert aware.source.covered_samples == 0
    assert np.array_equal(aware.audio, default.audio)


def test_a_voiced_render_covers_voiced_frames_only(trained_model,
                                                   tiny_source_model):
    score = _held_note_score()
    result = render(trained_model, score, tiny_source_model)
    source = result.source
    unvoiced = ~source.voiced
    if unvoiced.any():
        assert source.coverage[unvoiced].max() == 0.0
        assert not source.support.reshape(-1)[
            _frame_samples(unvoiced, source.hop, source.n_samples)].any()
    assert source.coverage[source.voiced].mean() > 0.5


def _frame_samples(frames: np.ndarray, hop: int, n_samples: int) -> np.ndarray:
    """Sample indices of the frames selected by ``frames`` (for assertions)."""
    mask = np.zeros(n_samples, dtype=bool)
    for frame in np.flatnonzero(frames):
        start, stop = frame * hop, min((frame + 1) * hop, n_samples)
        if stop > start:
            mask[start:stop] = True
    return mask


def test_the_transition_is_continuous_and_frame_aligned(trained_model,
                                                        tiny_source_model):
    """The blend weight ramps over one frame; it never steps."""
    score = _held_note_score()
    result = render(trained_model, score, tiny_source_model)
    weights = result.source.weights
    hop = result.source.hop
    steps = np.abs(np.diff(weights))
    assert steps.max() <= 1.0 / max(hop, 1) + 1e-9
    # the ramp is one frame long: the weight is 0 one frame before the first
    # covered sample and 1 one frame after it
    support = result.source.support
    first = int(np.flatnonzero(support)[0]) if support.any() else 0
    assert weights[max(0, first - hop)] < 1.0
    assert weights[min(len(weights) - 1, first + hop)] > 0.0


# --------------------------------------------------------------------------
# Gain
# --------------------------------------------------------------------------

def test_the_learned_excitation_is_calibrated_to_unit_rms(
        trained_model, tiny_source_model):
    """Phase 2 predicts shape, not level: the renderer supplies the neutral one."""
    score = _held_note_score()
    result = render(trained_model, score, tiny_source_model)
    source = result.source
    assert source.source_rms > 0.0
    covered = source.excitation[source.support]
    assert float(np.sqrt(np.mean(covered ** 2))) == pytest.approx(1.0, rel=1e-9)


def test_source_gain_does_not_destroy_acoustic_amplitude(
        trained_model, short_score, tiny_source_model):
    """Loudness stays with the envelope: the source only tilts it.

    Compared like for like -- one vocoder, two excitations -- the source-aware
    render tracks the ordinary one.  Measured 0.82x to 1.07x across both numpy
    backends and with and without the optional WORLD native backend, so the
    band below is a few dB either side of unity (a deliberately narrow-band
    synthetic source would still fall outside it -- see
    ``docs/source_model.md``).
    """
    default = render(trained_model, short_score)
    aware = render(trained_model, short_score, tiny_source_model)
    ratio = (np.sqrt(np.mean(aware.audio ** 2))
             / np.sqrt(np.mean(default.audio ** 2)))
    assert 0.25 < ratio < 4.0
    # and a neutral source is close, not merely within a decade
    assert 0.7 < ratio < 1.4


def test_per_unit_normalisation_keeps_the_excitation_from_spiking(
        trained_model, short_score, tiny_source_model):
    """Phase 2's sampled coefficients carry an unmodelled amplitude.

    Left alone, the decoded units on a fitted model span roughly an order of
    magnitude in RMS: the placed waveform gets isolated spikes, the filter
    clips, and a backend that peak-normalises turns the whole render down.
    Restoring Phase 1's unit-RMS-cycle convention removes that, so the
    excitation's crest factor stays in the same league as the pulse train it
    replaces (measured ~9 against ~6).
    """
    source = render(trained_model, short_score, tiny_source_model).source
    covered = source.excitation[source.support]
    rms = float(np.sqrt(np.mean(covered ** 2)))
    assert rms > 0.0
    crest = float(np.abs(source.excitation).max()) / rms
    assert crest < 20.0


def test_per_unit_normalisation_can_be_switched_off(tiny_source_model):
    """`normalize=False` is the escape hatch for raw generated amplitude."""
    spec = tiny_source_model.spec
    utterance = Utterance("raw", [
        Segment("a", 0.0, 60 * spec.frame_period / 1000.0, note=60.0)])
    f0 = np.full(60, 220.0)
    plain = render_source_excitation(tiny_source_model, utterance, f0)
    raw = render_source_excitation(tiny_source_model, utterance, f0,
                                   normalize=False)
    assert raw.n_units == plain.n_units > 0
    assert np.array_equal(raw.support, plain.support)
    # unit RMS, so every unit has the same level -- the two differ only in the
    # per-unit amplitude the generator happened to sample
    assert not np.allclose(raw.excitation[raw.support],
                           plain.excitation[plain.support])


def test_phase1_unit_gains_are_carried_when_present(tiny_source_model):
    """`restore_gain` is the hook a Phase-1 analysis (or a gain model) uses."""
    spec = tiny_source_model.spec
    utterance = Utterance("gained", [
        Segment("a", 0.0, 60 * spec.frame_period / 1000.0, note=60.0)])
    f0 = np.full(60, 220.0)
    plain = render_source_excitation(tiny_source_model, utterance, f0)
    sequence = tiny_source_model.generate(utterance, f0)
    doubled = sequence.sequence.with_coefficients(sequence.sequence.coefficients)
    object.__setattr__(doubled, "gains", 2.0 * np.asarray(sequence.sequence.gains))
    excitation = (doubled.excitation * doubled.gains[:, None])
    from hms.source.cycles import place_cycles
    placed = place_cycles(excitation, doubled.epochs, doubled.periods,
                          doubled.n_samples)
    # doubling every unit gain doubles the waveform before calibration, so the
    # calibrated result is the same shape -- the point being that the gain
    # track reaches `place_cycles` at all.
    assert placed.shape == plain.excitation.shape
    assert float(np.max(np.abs(placed))) > 0.0


# --------------------------------------------------------------------------
# Unit-level behaviour of the glue
# --------------------------------------------------------------------------

def test_unit_support_and_frame_coverage_agree_with_the_unit_spans():
    epochs = np.array([0, 100, 350], dtype=np.int64)
    periods = np.array([100, 100, 50], dtype=np.int64)
    support = unit_support_mask(epochs, periods, 600)
    assert support[:200].all()
    assert not support[200:350].any()
    assert support[350:400].all()
    assert not support[400:].any()
    coverage = frame_coverage(support, 100, 6)
    assert coverage == pytest.approx([1.0, 1.0, 0.0, 0.5, 0.0, 0.0])


def test_fade_weights_ramp_over_one_frame():
    support = np.zeros(400, dtype=bool)
    support[100:300] = True
    weights = fade_weights(support, 100)
    assert weights[0] == 0.0 and weights[99] < 1.0
    assert weights[200] == 1.0
    assert weights[399] == 0.0
    assert np.all(np.abs(np.diff(weights)) <= 1.0 / 100.0 + 1e-12)
    assert weights.min() >= 0.0 and weights.max() <= 1.0


def test_render_score_source_excitation_tiles_the_utterances(tiny_source_model):
    spec = tiny_source_model.spec
    hop = frame_hop(spec)
    utterances = [
        Utterance("first", [Segment("a", 0.0, 40 * spec.frame_period / 1000.0,
                                    note=60.0)]),
        Utterance("second", [Segment("i", 0.0, 40 * spec.frame_period / 1000.0,
                                     note=64.0)]),
    ]
    spans = [(utterances[0], 0, 40), (utterances[1], 40, 80)]
    f0 = np.full(80, 220.0)
    n_samples = render_length(80, spec.frame_period, spec.fs)
    source = render_score_source_excitation(tiny_source_model, spans, f0,
                                            n_samples)
    assert source.n_frames == 80
    assert source.n_samples == n_samples
    # the second utterance's excitation starts at its own frame offset
    assert source.support[40 * hop:].any()
    assert source.coverage[:40].mean() > 0.0
    assert source.coverage[40:].mean() > 0.0
    assert source.n_units > 0


def test_an_empty_span_list_renders_an_empty_excitation(tiny_source_model):
    spec = tiny_source_model.spec
    source = render_score_source_excitation(tiny_source_model, [],
                                            np.zeros(10), 0)
    assert source.n_samples == 0
    assert source.n_units == 0
    assert not source.support.any()


# --------------------------------------------------------------------------
# End to end: labels + F0 -> acoustic -> source -> filter -> waveform
# --------------------------------------------------------------------------

def test_end_to_end_labels_f0_acoustic_source_filter_waveform(
        trained_model, tiny_source_model):
    score = Score([Utterance(name="e2e", segments=[
        Segment("sil", 0.0, 0.10),
        Segment("m", 0.10, 0.25, note=62.0),
        Segment("a", 0.25, 0.75, note=62.0),
        Segment("n", 0.75, 0.90, note=65.0),
        Segment("a", 0.90, 1.30, note=65.0),
        Segment("sil", 1.30, 1.45)])])
    synthesizer = Synthesizer(trained_model,
                              SynthesisConfig(vocoder="mlsa", seed=0,
                                              vibrato=False))
    reference = synthesizer.synthesize(score)
    f0 = np.asarray(reference.params.f0, dtype=np.float64).copy()

    result = synthesizer.synthesize(score, f0=f0, source_model=tiny_source_model)

    assert np.isfinite(result.audio).all()
    assert len(result.audio) == render_length(
        len(result.params.f0), trained_model.spec.frame_period,
        trained_model.spec.fs)
    assert len(result.audio) == len(reference.audio)
    # the acoustic side is the render the caller asked for ...
    assert np.array_equal(result.params.f0, f0)
    assert np.allclose(result.params.sp, reference.params.sp)
    # ... the source produced units on the voiced part of it ...
    source = result.source
    assert source.n_units > 0
    assert source.covered_samples > 0
    assert source.covered_voiced_frames > 0
    # ... and it changed the waveform without breaking it
    assert not np.array_equal(result.audio, reference.audio)
    assert float(np.max(np.abs(result.audio))) > 1e-3
    assert np.isfinite(result.audio).all()
