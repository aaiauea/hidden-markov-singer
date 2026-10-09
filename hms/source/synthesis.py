"""Phase 3: render a Phase-2 prediction as an excitation for HMS synthesis.

Phases 1 and 2 built the source branch without touching synthesis:

    Phase 1   audio -> source units -> PCA -> reconstruction
    Phase 2   labels + F0 -> source HMM/GMM -> PCA coefficients -> Phase-1 decoder

This module is the seam between that branch and the filter.  It turns a
:class:`~hms.source.hmm.SourcePrediction` into an **excitation waveform on the
sample grid**, plus the two arrays a synthesis backend needs to splice it in:

    excitation   (n_samples,)  the learned source, level-calibrated
    support      (n_samples,)  bool: where that waveform really exists
    weights      (n_samples,)  the same, with a one-frame fade at every edge
    coverage     (n_frames,)   how much of each acoustic frame it covers

Nothing here knows about WORLD, MLSA or mel-cepstra.  The split is deliberate:

* **This module owns the source.** It calls ``SourceHMMModel.generate``,
  decodes through the saved PCA basis (the model already did that), places the
  units on the sample grid with the Phase-1 :func:`place_cycles`, calibrates the
  level and describes the coverage.
* **The vocoder owns the filter.** A backend that can filter a caller-supplied
  excitation says so (``Vocoder.supports_external_excitation``) and exposes
  ``synthesize_with_excitation``; a backend that cannot (WORLD) keeps rendering
  exactly what it renders today.

What the learned source replaces
--------------------------------
Only the *periodic* half of the excitation -- the pulse train.  The noise half
(aperiodicity) is left entirely alone, which is what keeps two existing
behaviours intact for free:

* an unvoiced frame is pure noise in every HMS backend (``ap`` is forced to 1),
  so it is unaffected by anything this module produces -- the learned source is
  not asked to be a noise generator, and it does not have to be muted;
* loudness stays with the acoustic model: the spectral envelope and the
  aperiodicity keep deciding how loud a frame is, as they did before.

Gain
----
Phase 1 stores a source unit as a **unit-RMS cycle plus a separate level**
(``SourceSequence.excitation`` is normalised, ``SourceSequence.gains`` carries
the analysed level, and ``vector * gain`` is the cycle that was analysed).
Phase 2 predicts *shape*: it emits PCA coefficients and unit gains of ``1.0``,
and the coefficients it samples carry an unmodelled amplitude of their own --
measured spread on a fitted model is an order of magnitude, which in a
waveform means isolated spikes, a peak that clips the filter, and (after a
backend's peak normalisation) a render that is several dB quieter than the
ordinary one.

So the render restores Phase 1's invariant before placing anything:

1. every generated unit is scaled to **unit RMS**, which is a no-op for a
   Phase-1 analysis (its cycles already are) and removes the unmodelled
   amplitude a generated one arrives with.  A degenerate unit -- numerically
   zero, so it carries no shape either -- is dropped rather than multiplied by
   ``1/eps``;
2. the sequence's own **per-unit gains** are then applied on top
   (``restore_gain``), so a real analysis reaches the output unchanged and only
   the level Phase 2 never learned is replaced by a constant;
3. the placed waveform is finally scaled to **unit RMS over the samples it
   covers** -- the calibration of the MLSA pulse train it replaces, which is
   built to have unit mean square at any F0 so the output level tracks ``sp``.
   The measured RMS is kept in :attr:`SourceExcitation.source_rms` so the
   correction is inspectable rather than hidden;
4. a single deliberate ``source_gain`` is applied on top (default 1.0 =
   neutral).

Net effect: **loudness stays with the acoustic model.**  Learning per-unit
gains needs a new training target in the source HMM, so it stays out of Phase
3; see ``docs/source_model.md``.

Timing
------
The acoustic branch and the source branch are made to agree on one frame grid
before anything is rendered:

* ``fs`` and ``frame_period`` must match the acoustic model exactly, or
  :func:`source_model_diagnostics` raises -- there is no resampling;
* the F0 handed to the source model is the render's own F0 track (Hz,
  ``0`` = unvoiced), sliced per utterance, so both branches see the same pitch
  and the same voicing decision by construction;
* the frame count is the acoustic branch's, never a second opinion;
* an utterance starting at frame ``lo`` is written at sample ``lo * hop`` -- the
  same frame-to-sample mapping the vocoder uses -- so no drift accumulates
  across a long score.

Samples the source does not cover keep their weight at 0, so the backend's own
excitation serves them exactly as it did before Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hms.core.features import FeatureSpec
from hms.source.cycles import (frame_coverage, place_cycles,
                               unit_support_mask)
from hms.vocoder.base import render_length

#: RMS below which a rendered source counts as silent (and is left alone rather
#: than being scaled up by 1/eps).
_MIN_SOURCE_RMS = 1e-9

#: RMS below which a single generated unit is treated as degenerate: it carries
#: no shape, so it is silenced instead of being normalised by 1/eps into a
#: spike that would dominate the whole excitation.
_MIN_UNIT_RMS = 1e-9


def frame_hop(spec: FeatureSpec) -> int:
    """Samples per analysis frame for ``spec`` (the HMS frame grid)."""
    return max(1, int(round(int(spec.fs) * float(spec.frame_period) / 1000.0)))


def normalize_units(cycles: np.ndarray) -> np.ndarray:
    """Scale every source unit to unit RMS (returns a new array).

    This is Phase 1's own storage convention (``SourceSequence.excitation``
    holds unit-RMS cycles and ``SourceSequence.gains`` holds the level), so for
    an analysed sequence this is the identity and for a generated one it strips
    the unmodelled amplitude the sampled PCA coefficients arrive with.  Units
    that are numerically silent are zeroed -- having no shape, they cannot be
    normalised into one.
    """
    cycles = np.asarray(cycles, dtype=np.float64)
    if cycles.ndim != 2 or cycles.size == 0:
        return np.zeros(cycles.shape, dtype=np.float64)
    rms = np.sqrt(np.mean(cycles * cycles, axis=1))
    scale = np.zeros_like(rms)
    usable = rms > _MIN_UNIT_RMS
    scale[usable] = 1.0 / rms[usable]
    return cycles * scale[:, None]


def fade_weights(support: np.ndarray, fade: int) -> np.ndarray:
    """``[0, 1]`` sample weights: ``support`` with a linear ramp at each edge.

    Switching between two excitations sample-by-sample would put a step into the
    filter's input, so every boundary is ramped over ``fade`` samples instead.

    The ramp is **inside** the covered span, never outside it.  That matters
    because the mix a backend performs is

        ``out = weights * learned + (1 - weights) * default``

    and the learned waveform is zero wherever ``support`` is false.  A weight
    that leaks past the last covered sample would therefore not crossfade into
    anything -- it would *attenuate* the backend's own excitation and replace it
    with silence.  A centred moving average does exactly that: it ramps from
    ``fade // 2`` samples before coverage starts, so an unvoiced gap next to a
    voiced run lost half its level at the boundary.  Taking the minimum of a
    backward and a forward moving average keeps the whole transition where both
    signals exist.

    Both averages come from one cumulative sum, so this is still O(n) with no
    kernel of length ``fade`` and no scipy.  Where a run starts at sample 0 the
    backward window is truncated, so the weight climbs from ``1 / fade`` over
    the render's first samples instead of switching on at full level; a run
    that reaches the last sample simply stays high, because there is nothing
    after it left to preserve.

    The minimum alone is not enough, and the result is masked by ``support``
    for that reason.  Each average reaches ``fade - 1`` samples past the end of
    the coverage it can see -- the backward one into the gap *after* a run, the
    forward one into the gap *before* the next -- so in an uncovered gap
    narrower than about two fades both are positive at once.  Masking makes the
    invariant unconditional rather than dependent on how far apart two runs
    happen to be, and costs nothing inside a run, where the mask is 1.

    ``fade`` defaults to one frame period at the call sites below, which is what
    makes the transition frame-aligned: it is one frame long and it starts at
    the sample where the source's coverage actually starts.
    """
    values = np.asarray(support, dtype=np.float64).reshape(-1)
    n = values.size
    fade = int(fade)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if fade <= 1:
        return np.clip(values, 0.0, 1.0)
    prefix = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(values, out=prefix[1:])
    index = np.arange(n, dtype=np.int64)
    # Backward average: ramps up over the first `fade` samples of a run, then
    # holds at 1.  Forward average: holds at 1, then ramps down over the last
    # `fade` samples.  Their minimum is 1 through the middle of a run, ramps
    # only where the run itself is, and is exactly 0 everywhere else.
    upward = (prefix[index + 1] - prefix[np.clip(index - fade + 1, 0, n)]
              ) * (1.0 / float(fade))
    downward = (prefix[np.clip(index + fade, 0, n)] - prefix[index]
                ) * (1.0 / float(fade))
    # The mask is the invariant, stated once: whatever the two averages do, a
    # sample the source does not cover is never handed to the learned source.
    return np.minimum(upward, downward) * np.clip(values, 0.0, 1.0)


@dataclass
class SourceExcitation:
    """A learned excitation rendered onto the sample grid, ready to filter.

    Attributes
    ----------
    excitation
        ``(n_samples,)`` waveform, calibrated to unit RMS over its support and
        scaled by ``source_gain``.  Zero where the source model has no units.
    support
        ``(n_samples,)`` bool: samples a generated source unit spans.
    weights
        ``(n_samples,)`` in ``[0, 1]``: ``support`` faded over one frame at each
        boundary.  A backend blends ``weights * learned + (1 - weights) * own``.
    coverage
        ``(n_frames,)`` in ``[0, 1]``: share of each acoustic frame covered.
    voiced, f0_hz
        The track the source was generated from (the render's own F0, Hz,
        ``0`` = unvoiced), one entry per acoustic frame.
    n_units
        Generated source units (pitch cycles for ``voice``, frame groups for
        ``residual``).
    source_rms, applied_gain
        The RMS of the placed waveform before calibration and the scalar that
        was applied -- the whole gain story, inspectable.
    """

    excitation: np.ndarray
    support: np.ndarray
    weights: np.ndarray
    coverage: np.ndarray
    voiced: np.ndarray
    f0_hz: np.ndarray
    fs: int
    frame_period: float
    n_units: int = 0
    backend: str = ""
    source_gain: float = 1.0
    source_rms: float = 0.0
    applied_gain: float = 1.0

    def __post_init__(self) -> None:
        self.excitation = np.asarray(self.excitation, dtype=np.float64).reshape(-1)
        self.support = np.asarray(self.support, dtype=bool).reshape(-1)
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        self.coverage = np.asarray(self.coverage, dtype=np.float64).reshape(-1)
        self.voiced = np.asarray(self.voiced, dtype=bool).reshape(-1)
        self.f0_hz = np.asarray(self.f0_hz, dtype=np.float64).reshape(-1)
        self.fs = int(self.fs)
        self.frame_period = float(self.frame_period)
        self.n_units = int(self.n_units)
        if not (len(self.support) == len(self.weights) == len(self.excitation)):
            raise ValueError("excitation, support and weights must be one array "
                             "of the same length")
        n_frames = len(self.f0_hz)
        if len(self.coverage) != n_frames or len(self.voiced) != n_frames:
            raise ValueError("coverage and voiced must have one entry per frame")

    @property
    def n_samples(self) -> int:
        return int(self.excitation.size)

    @property
    def n_frames(self) -> int:
        return int(self.f0_hz.size)

    @property
    def hop(self) -> int:
        return max(1, int(round(self.fs * self.frame_period / 1000.0)))

    @property
    def covered_samples(self) -> int:
        """Samples the learned source actually drives."""
        return int(self.support.sum())

    @property
    def covered_voiced_frames(self) -> int:
        """Voiced frames at least half covered by the learned source."""
        return int(np.count_nonzero(self.voiced & (self.coverage > 0.5)))

    def summary(self) -> str:
        """One-line diagnostic describing what the source branch supplied."""
        n_voiced = int(np.count_nonzero(self.voiced))
        fraction = (100.0 * self.covered_samples / max(self.n_samples, 1))
        covered = self.covered_voiced_frames
        return (f"source-aware synthesis: {self.n_units} learned "
                f"{self.backend or 'source'} unit(s) covering {fraction:.1f} % "
                f"of {self.n_samples} sample(s) and {covered}/{n_voiced} "
                f"voiced frame(s)")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<SourceExcitation {self.backend or 'source'} "
                f"units={self.n_units} frames={self.n_frames} "
                f"samples={self.n_samples} fs={self.fs}>")


def source_model_diagnostics(model, source_model) -> List[str]:
    """Check a Phase-2 source model against the acoustic model; raise on geometry.

    The two branches share one frame grid and one F0 track, so a disagreement
    is a caller error rather than something to paper over:

    * ``fs`` and ``frame_period`` must match **exactly** -- there is no
      resampling anywhere in HMS, and a 22050 Hz source silently driving a
      44100 Hz render would be a half-speed excitation with no warning;
    * a different ``voiced_threshold`` / ``f0_ref_hz`` is *reported*, not fatal:
      the source model derives its own voicing from the F0 it is given, and the
      acoustic branch's voicing still reaches the vocoder unchanged, so the two
      simply cannot fight (an unvoiced frame is pure noise whatever the source
      says).  It is still worth saying out loud.
    """
    acoustic: FeatureSpec = model.spec
    source: FeatureSpec = source_model.spec
    if int(acoustic.fs) != int(source.fs):
        raise ValueError(
            f"source model sample rate ({int(source.fs)} Hz) does not match the "
            f"acoustic model ({int(acoustic.fs)} Hz); HMS never resamples audio "
            f"or source predictions -- train the source model at the acoustic "
            f"model's rate")
    if float(acoustic.frame_period) != float(source.frame_period):
        raise ValueError(
            f"source model frame period ({float(source.frame_period)} ms) does "
            f"not match the acoustic model ({float(acoustic.frame_period)} ms); "
            f"HMS never resamples a source prediction onto another frame grid")
    diagnostics: List[str] = []
    if float(acoustic.voiced_threshold) != float(source.voiced_threshold):
        diagnostics.append(
            f"source model voiced threshold ({float(source.voiced_threshold):g} "
            f"Hz) differs from the acoustic model's "
            f"({float(acoustic.voiced_threshold):g} Hz); the F0 track is shared, "
            f"but each branch derives its own voicing from it, so a few boundary "
            f"frames may disagree")
    if float(acoustic.f0_ref_hz) != float(source.f0_ref_hz):
        diagnostics.append(
            f"source model F0 reference ({float(source.f0_ref_hz):g} Hz) differs "
            f"from the acoustic model's ({float(acoustic.f0_ref_hz):g} Hz); the "
            f"semitone scale used for F0 conditioning was fitted differently")
    return diagnostics


def render_source_excitation(source_model, utterance, f0_hz,
                             n_samples: Optional[int] = None, *,
                             mixture: str = "dominant",
                             variance_scale: float = 1.0,
                             source_gain: float = 1.0,
                             restore_gain: bool = True,
                             normalize: bool = True,
                             fade_samples: Optional[int] = None,
                             seed: int = 0) -> SourceExcitation:
    """One utterance: labels + F0 -> a learned excitation on the sample grid.

    ``f0_hz`` is the render's own F0 track for this utterance (Hz, one value per
    acoustic frame, ``0`` = unvoiced) and ``n_samples`` the samples this
    utterance occupies -- both come from the acoustic branch, so the two
    branches cannot disagree about pitch, voicing or length.

    ``source_gain`` scales the result after the neutral calibration described in
    this module's docstring; ``normalize`` switches the per-unit RMS
    normalisation that restores Phase 1's unit-RMS-cycle convention (default
    on: switch it off only to render a generated sequence's raw amplitude);
    ``fade_samples`` overrides the one-frame fade used at coverage boundaries.
    """
    spec = source_model.spec
    hop = frame_hop(spec)
    f0 = np.asarray(f0_hz, dtype=np.float64).reshape(-1)
    n_frames = int(f0.size)
    if n_samples is None:
        n_samples = n_frames * hop
    n_samples = int(n_samples)
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")

    prediction = source_model.generate(
        utterance, f0, n_samples=n_samples, mixture=mixture,
        variance_scale=variance_scale, seed=seed)
    sequence = prediction.sequence
    gains = (np.asarray(sequence.gains, dtype=np.float64)
             if restore_gain else np.ones(sequence.n_units, dtype=np.float64))
    cycles = np.asarray(sequence.excitation, dtype=np.float64)
    if normalize:
        cycles = normalize_units(cycles)
    placed = place_cycles(
        cycles * gains[:, None], sequence.epochs, sequence.periods, n_samples)
    support = unit_support_mask(sequence.epochs, sequence.periods, n_samples)

    source_rms = 0.0
    if support.any():
        covered = placed[support]
        source_rms = float(np.sqrt(np.mean(covered * covered)))
    applied = float(source_gain) / source_rms if source_rms > _MIN_SOURCE_RMS \
        else float(source_gain)
    if applied != 1.0:
        placed = placed * applied
    fade = hop if fade_samples is None else int(fade_samples)

    return SourceExcitation(
        excitation=placed, support=support, weights=fade_weights(support, fade),
        coverage=frame_coverage(support, hop, n_frames),
        voiced=np.asarray(prediction.voiced, dtype=bool),
        f0_hz=np.asarray(prediction.f0_hz, dtype=np.float64),
        fs=int(spec.fs), frame_period=float(spec.frame_period),
        n_units=int(sequence.n_units), backend=str(sequence.backend),
        source_gain=float(source_gain), source_rms=float(source_rms),
        applied_gain=float(applied))


def render_score_source_excitation(source_model,
                                   spans: Sequence[Tuple[object, int, int]],
                                   f0_hz, n_samples: int, *,
                                   mixture: str = "dominant",
                                   variance_scale: float = 1.0,
                                   source_gain: float = 1.0,
                                   restore_gain: bool = True,
                                   normalize: bool = True,
                                   fade_samples: Optional[int] = None,
                                   seed: int = 0) -> SourceExcitation:
    """Concatenate per-utterance source renderings onto one sample grid.

    ``spans`` is ``(utterance, start_frame, stop_frame)`` as the acoustic
    branch's ``plan`` produced them: Phase 2 is defined per utterance, so the
    source branch runs once per utterance and the pieces are written at
    ``start_frame * hop`` -- the same frame-to-sample mapping the vocoder uses.
    Running the whole score through one ``generate`` call is not possible (and
    would smear phone context across utterance boundaries), and running it
    utterance-by-utterance is exactly what keeps the frame grids aligned: the
    acoustic branch plans the same utterances, in the same order, at the same
    frame period.

    Samples no utterance renders (past the end of the frame grid, or an
    utterance with no frames) stay zero and uncovered, which leaves them to the
    backend's own excitation.
    """
    spec = source_model.spec
    hop = frame_hop(spec)
    f0 = np.asarray(f0_hz, dtype=np.float64).reshape(-1)
    n_samples = max(0, int(n_samples))
    excitation = np.zeros(n_samples, dtype=np.float64)
    support = np.zeros(n_samples, dtype=bool)
    coverage = np.zeros(len(f0), dtype=np.float64)
    voiced = np.zeros(len(f0), dtype=bool)
    n_units = 0
    backend = ""

    for entry in spans:
        utterance, lo, hi = entry
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            continue
        start = lo * hop
        stop = min(n_samples, start + (hi - lo) * hop)
        if stop <= start:
            continue
        f0_slice = f0[lo:hi]
        if len(f0_slice) != hi - lo:
            raise ValueError(f"F0 has {len(f0)} frame(s) but utterance "
                             f"{getattr(utterance, 'name', '')!r} asks for "
                             f"frames [{lo}, {hi}); a source prediction is "
                             f"never resized or padded")
        piece = render_source_excitation(
            source_model, utterance, f0_slice, n_samples=stop - start,
            mixture=mixture, variance_scale=variance_scale,
            source_gain=source_gain, restore_gain=restore_gain,
            normalize=normalize, fade_samples=fade_samples, seed=seed)
        excitation[start:stop] = piece.excitation[:stop - start]
        support[start:stop] = piece.support[:stop - start]
        coverage[lo:hi] = piece.coverage[:hi - lo]
        voiced[lo:hi] = piece.voiced[:hi - lo]
        n_units += piece.n_units
        backend = piece.backend or backend
        del piece

    fade = hop if fade_samples is None else int(fade_samples)
    return SourceExcitation(
        excitation=excitation, support=support,
        weights=fade_weights(support, fade), coverage=coverage, voiced=voiced,
        f0_hz=f0, fs=int(spec.fs), frame_period=float(spec.frame_period),
        n_units=n_units, backend=backend, source_gain=float(source_gain),
        source_rms=float(np.sqrt(np.mean(excitation[support] ** 2)))
        if support.any() else 0.0, applied_gain=float(source_gain))


__all__ = ["SourceExcitation", "render_source_excitation",
           "render_score_source_excitation", "source_model_diagnostics",
           "source_model_is_compatible", "fade_weights", "render_length",
           "frame_hop", "normalize_units"]


def source_model_is_compatible(model, source_model) -> bool:
    """True when ``source_model`` shares the acoustic model's frame grid."""
    try:
        source_model_diagnostics(model, source_model)
    except ValueError:
        return False
    return True
