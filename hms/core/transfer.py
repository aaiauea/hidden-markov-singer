"""Experimental cross-language voice transfer (classical, no neural parts).

The problem
-----------
HMS trains one voice from one corpus, in one language.  A singer who has
recorded a language *A* corpus cannot sing a language *B* score: every phone of
B that their corpus never contained has no acoustic model at all, so synthesis
falls back to a phone-class backoff -- a plausible sound, but not that phone.

This feature lets two *independently trained* HMS voices be combined at training
time:

    Voice 1 (target)     : the timbre that must be kept   -- corpus in language A
    Voice 2 (auxiliary)  : the phonetic coverage wanted   -- corpus in language B
                |
                v  classical statistical adaptation of Voice 2's units
                v  into Voice 1's acoustic space
    Voice 1 singing language B (one ordinary HMS model, one HMM per unit)

It is **not** multilingual G2P support, not a joint multi-language model, and
not a neural method: no embedding, no network, no external framework is
involved.  The output is an ordinary `hms.core.model.HMSModel` whose extra tier
is a handful of *transferred* HMMs, each one a plain diagonal-covariance GMM per
state, exactly like every other unit the engine has -- and everything that
already exists (WORLD, MLPG, F0/vibrato, contexts, class and global backoff,
disk-backed training, pitch conditioning) is untouched and stays in charge.

Why the target stays the target
-------------------------------
HMS normalises every corpus by its *own* per-dimension mean and standard
deviation before training (``z = (x - offset) * scale``,
`hms.core.trainer.Trainer._normalization_from_moments`).  A model's GMMs
therefore live in that model's own z-scored space, in which its corpus has
approximately zero mean and unit variance in every dimension.

That single fact is what makes the target the target: a transferred unit is
*expressed in Voice 1's space by construction*, because its numbers are written
in Voice 1's normalisation, with Voice 1's cepstral offset and Voice 1's
per-dimension spread.  Nothing Voice 2 owns -- its absolute spectral envelope,
its loudness, its average cepstrum, its variance -- survives that conversion.
What is taken from Voice 2 is the *shape* of the phone: how it sits relative to
the other phones of language B.  The regression below decides where that shape
lands inside Voice 1's space.

The mechanism
-------------
The static feature block is ``[0]`` = note-relative F0 in semitones,
``[1:1+n]`` = mel-cepstrum, ``[1+n:]`` = aperiodicity bands; the delta and
delta-delta streams repeat that block.

**0. Anchors.**  A phone that both voices have a dedicated HMM for is an
*anchor*: the same phonetic unit observed through two vocal tracts.  Anchors are
the only data the mapping is estimated from; they are simply the intersection of
the two voices' trained units.  When that intersection is *empty* -- two
languages need not share a single phone symbol -- the map is estimated from
pooled phoneme-class anchors instead (both voices' vowels, both voices'
silence, ...), which always exist but are coarser: the two pools hold different
phones.  A real phone correspondence is therefore always preferred, and the
class fallback is recorded as such.

**1. Anchor regression (MAP, shrunk towards the identity).**  For each anchor
phone ``p`` take the pooled mean of its HMM in each voice's own normalised space
(frame-weighted over states, mixture-weighted inside a state):

    x_p = pooled mean of p in Voice 2's space
    y_p = pooled mean of p in Voice 1's space

and fit the affine map ``y = A x + b`` that explains the largest amount of ``y``
while staying as close to the identity as the evidence allows:

    (A, b) = argmin  sum_p || y_p - A x_p - b ||^2  +  kappa * ||[A|b] - [I|0]||^2

    theta = (Z^T Z + kappa I)^-1 (Z^T Y + kappa [I|0]),  Z = [[x_p, 1]],
    theta = [A|b]^T

which is the closed-form posterior mean of the coefficients under a Gaussian
prior centred on the identity.  ``kappa`` (``map_prior_strength``) is the number
of anchor phones the identity is worth: with no anchors the map *is* the
identity, with one anchor it is the smallest deviation from the identity that
passes through that anchor, and with many anchors the data dominates.  Read in
absolute units, the identity map is mean/variance matching, so the prior is not
arbitrary: "keep Voice 2's geometry unless Voice 1's anchors show otherwise",
and every anchor phone has to earn its deviation.

``A`` is a full matrix, not one slope per dimension.  The dominant difference
between two vocal tracts is a *warping* of the spectral envelope, and a warping
mixes mel-cepstral coefficients: the diagonal is the "rescale each coefficient"
approximation of it, the off-diagonal entries the rest.  ``b`` is a constant
offset, so it is applied to the static block only -- a delta stream has no mean,
and adding one there would be a bug, not a model.

**2. Variance propagation.**  Applying the map to Voice 2's GMM means also
moves their variances.  Two effects are propagated, per dimension:

    sigma'^2_i  =  sum_j A[i, j]^2 * sigma_aux^2_j  +  s^2_i
    s^2_i       =  ( sum_p (y_pi - (A x_p + b)_i)^2  +  kappa * Var(y_i) )
                   / (P + kappa)

``s^2`` is the *residual*: the part of Voice 1's phone-to-phone variation the
anchor regression does not explain.  It is a Voice 1 quantity measured in Voice
1's own units, and it is what keeps a transferred unit from pretending to be
more certain than the mapping is.  The ``kappa * Var(y)`` term is what stops a
single anchor (or an exactly-determined fit) from claiming a zero-variance
transformation.  With no anchors at all, ``A = I``, ``b = 0`` and ``s^2`` falls
back to the spread of Voice 1's own phone means -- the transferred unit is
blurred by Voice 1's between-phone variance.  Variances are floored at
``MIN_TRANSFER_VARIANCE`` and never replace the mixture structure: components,
weights, self-loops and state durations keep their Voice 2 values.

**3. MAP mean/variance adaptation to Voice 1's own frames (when there are any).**
A phone that Voice 1's corpus *does* contain -- but far too rarely to earn its
own HMM -- has observations of its own, and they are worth using.  Writing ``n``
for the frames Voice 1 has of that phone, ``m_t``/``v_t`` for their pooled
mean/variance, ``m_p``/``v_p`` for the pooled mean/variance of the *mapped*
auxiliary model, and ``tau`` (``map_adapt_frames``) for the prior strength in
frames, the classical MAP blend is applied:

    m = (n * m_t + tau * m_p) / (n + tau)      -> a location shift of the unit
    v = (n * v_t + tau * v_p) / (n + tau)      -> a per-dimension spread factor

Both are pooled-moment corrections.  The mean correction shifts every component
of every state by the same offset, so the mixture's shape is preserved; the
delta streams, being differences, are not shifted.  The spread factor scales the
static dimension *and* the dynamic streams derived from it.  With ``n = 0`` --
the phone is completely absent from Voice 1's corpus, the case this feature
exists for -- the blend is a no-op and the mapped model stands as it is.  The
more Voice 1's own frames say, the further the unit is pulled towards Voice 1's
own observations instead of Voice 2's.

**4. What is deliberately *not* adapted.**  State durations, self-loop
probabilities, per-state voicing probabilities and -- when the voices carry them
-- the note-relative pitch statistics and voicing priors are copied from Voice
2, and its duration statistics are imported into the target's duration model.
Those describe *how language B's phone is produced* (how long it lasts, whether
it is voiced, how it moves around the note), not the singer's timbre; a
transferred unit whose timing came from language A would be a different phone.
They are copied as plain HMS statistics, not fitted again, so nothing about the
existing duration/pitch architecture changes.

**5. Pitch-conditioned units.**  When the target voice was trained with the
optional pitch-conditioned tier (`hms.core.pitch_condition`) *and* the auxiliary
voice carries the same tier with the same bin width, the auxiliary's
``(phone, bin)`` buckets of the transferred phones are mapped with the same
acoustic map and the same MAP correction as their phone-level model, and
installed as the target's own pitch-conditioned units.  A pitch bin is an
absolute MIDI range, so it means the same thing in both voices; the map is the
only thing that has to travel.  Bins the auxiliary never trained, and bins that
do not clear the target's support threshold, are simply absent -- exactly as for
native units, synthesis then falls back to the transferred phone model rather
than to a bin that does not exist.  A bin-width mismatch between the two voices
is a configuration error and says so, because silently reinterpreting Voice 2's
bins would attach the wrong pitch region to each model.

What this is not
----------------
* No unit is ever taken from Voice 2 *as it is*: the mean map, the variance
  propagation and the MAP blend rewrite every mean and variance that crosses the
  boundary.  A phone Voice 1 has a dedicated HMM for is never touched, whatever
  the auxiliary voice contains -- the transfer tier only fills units Voice 1
  cannot sing.
* No claim of quality is made.  What is testable, and what the tests check, is
  that the mechanism is deterministic, that it is exactly the arithmetic above,
  that a transferred unit lies in Voice 1's acoustic space, that it is used
  where the ground truth says it should be, and that everything else -- every
  other tier, every other feature -- behaves exactly as it did before.

Status: **experimental, off by default** (``transfer.enabled: false``; the
trainer only ever calls this module when `--transfer-model` /
`--transfer-labels` are given).  With the feature off, nothing here runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hms.core.duration import DurationStats
from hms.core.features import FeatureSpec
from hms.core.gmm import DiagGMM
from hms.core.hmm import HMMState, LeftToRightHMM, StateDurationStats
from hms.core.phonemes import PhonemeDef, PhonemeSet
from hms.core.pitch import PitchStats
from hms.core.pitch_condition import KIND_PHONE, PitchConditioning

#: Tier name `HMSModel.resolve_unit` reports for a transferred unit.  Tier names
#: are the model's own vocabulary (`phone`, `triphone`, `left`, `right`,
#: `phone+pitch`, `class`, `global`); this one is new and nothing else uses it.
TIER_TRANSFERRED = "transferred"

#: Default number of anchor observations the identity map is worth (``kappa``).
DEFAULT_MAP_PRIOR_STRENGTH = 1.0

#: Default number of Voice 1 frames the transferred unit is worth in the MAP
#: blend of stage 3 (``tau``).
DEFAULT_MAP_ADAPT_FRAMES = 100.0

#: Variance floor applied to every transferred component (normalised units).
MIN_TRANSFER_VARIANCE = 1e-6

#: Bounds on the pooled variance spread factor (keeps a degenerate anchor set
#: from collapsing or exploding a unit).
MIN_SPREAD_FACTOR = 1e-3
MAX_SPREAD_FACTOR = 1e3

#: Bounds on how far the map may move a static dimension's *mean* from where it
#: already is, in normalised units (one unit = one standard deviation of the
#: feature over the whole corpus).  A guard, not a modelling choice: a
#: degenerate anchor set must not be able to place a unit outside the space.
MAX_MEAN_SHIFT = 8.0

#: Residual variance used when neither anchors nor any Voice 1 phone mean exists
#: to measure one: the unit variance of a normalised feature dimension.
FALLBACK_RESIDUAL_VARIANCE = 1.0


# --------------------------------------------------------------------------
# Who takes part
# --------------------------------------------------------------------------


def _label(speaker: str, language: str) -> str:
    """``voice-2 (de)``, ``voice-2`` or ``(de)`` -- whatever is known."""
    speaker = str(speaker or "")
    language = str(language or "")
    if speaker and language:
        return f"{speaker} ({language})"
    return speaker or (f"({language})" if language else "unspecified")


@dataclass(frozen=True)
class CrossLanguageTransfer:
    """The transfer configuration *and* its outcome, as stored in a model.

    The configuration half is what the user asked for (the ``transfer:`` block
    in `parameters.yaml`, or the ``--transfer-*`` flags); the outcome half
    (``anchor_phones``, ``transferred_phones``, ``transferred_pitch_units``) is
    filled in by the trainer, so a saved model documents which phone pairs the
    mapping was estimated from and which units it contributed.  Nothing here is
    needed to *synthesise* with the model -- the units are already in it -- but
    everything here is needed to explain it.
    """

    enabled: bool = False
    target_speaker: str = ""
    target_language: str = ""
    auxiliary_speaker: str = ""
    auxiliary_language: str = ""
    #: Where Voice 2 came from: a model directory or a label file.
    auxiliary_source: str = ""
    #: Transfer the auxiliary's pitch-conditioned ``(phone, bin)`` units too.
    adapt_pitch_bins: bool = True
    #: ``kappa``: anchor observations the identity map is worth (must be > 0).
    map_prior_strength: float = DEFAULT_MAP_PRIOR_STRENGTH
    #: ``tau``: frames Voice 1's own observations of a phone are worth.
    map_adapt_frames: float = DEFAULT_MAP_ADAPT_FRAMES
    #: Anchor phones the acoustic map was estimated from (phones both voices
    #: have their own HMM for).
    anchor_phones: Tuple[str, ...] = ()
    #: Phoneme classes used as pooled anchors (always available, so the map is
    #: still estimated when the two languages share no phone at all).
    anchor_classes: Tuple[str, ...] = ()
    #: Phones that received a transferred HMM.
    transferred_phones: Tuple[str, ...] = ()
    #: Transferred pitch-conditioned ``(phone, bin)`` units.
    transferred_pitch_units: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "adapt_pitch_bins",
                           bool(self.adapt_pitch_bins))
        for name in ("target_speaker", "target_language", "auxiliary_speaker",
                     "auxiliary_language", "auxiliary_source"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        for name in ("anchor_phones", "anchor_classes", "transferred_phones"):
            values = getattr(self, name) or ()
            object.__setattr__(self, name,
                               tuple(sorted(str(value) for value in values)))
        strength = float(self.map_prior_strength)
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError(
                "transfer.map_prior_strength must be finite and positive (it "
                "is the identity prior's weight, in anchor observations), got "
                f"{self.map_prior_strength!r}")
        object.__setattr__(self, "map_prior_strength", strength)
        frames = float(self.map_adapt_frames)
        if not np.isfinite(frames) or frames < 0:
            raise ValueError(
                "transfer.map_adapt_frames must be finite and non-negative (it "
                f"is a frame count), got {self.map_adapt_frames!r}")
        object.__setattr__(self, "map_adapt_frames", frames)
        object.__setattr__(self, "transferred_pitch_units",
                           max(0, int(self.transferred_pitch_units)))

    @property
    def active(self) -> bool:
        """Whether this record describes an actual transfer."""
        return bool(self.enabled)

    @property
    def target_label(self) -> str:
        return _label(self.target_speaker, self.target_language)

    @property
    def auxiliary_label(self) -> str:
        return _label(self.auxiliary_speaker, self.auxiliary_language)

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "target_speaker": self.target_speaker,
            "target_language": self.target_language,
            "auxiliary_speaker": self.auxiliary_speaker,
            "auxiliary_language": self.auxiliary_language,
            "auxiliary_source": self.auxiliary_source,
            "adapt_pitch_bins": bool(self.adapt_pitch_bins),
            "map_prior_strength": float(self.map_prior_strength),
            "map_adapt_frames": float(self.map_adapt_frames),
            "anchor_phones": list(self.anchor_phones),
            "anchor_classes": list(self.anchor_classes),
            "transferred_phones": list(self.transferred_phones),
            "transferred_pitch_units": int(self.transferred_pitch_units),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]
                  ) -> "CrossLanguageTransfer":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if unknown:
            raise ValueError("unknown cross-language transfer options: "
                             f"{sorted(unknown)}")
        return cls(**data)

    def describe(self) -> str:
        """One-line summary for logs, diagnostics and `inspect-model`."""
        if not self.enabled:
            return "cross-language transfer: disabled"
        anchors = f"{len(self.anchor_phones)} anchor phone(s)"
        if self.anchor_classes:
            anchors += f" + {len(self.anchor_classes)} class anchor(s)"
        return (f"cross-language transfer: target {self.target_label} <- "
                f"auxiliary {self.auxiliary_label} "
                f"({anchors}, "
                f"{len(self.transferred_phones)} transferred phone(s), "
                f"{self.transferred_pitch_units} transferred pitch unit(s))")


@dataclass
class VoiceSource:
    """One trained voice, as the transfer stage reads it.

    A deliberately narrow view of `hms.core.model.HMSModel`: the transfer math
    touches no synthesis machinery, so it needs no model object -- just the
    statistics it adapts.  `from_model` fills this in from any object with the
    same attributes (duck-typed on purpose: importing ``HMSModel`` here would
    make this module and the model module import each other).
    """

    speaker: str = ""
    language: str = ""
    source: str = ""
    hmms: Dict[str, LeftToRightHMM] = field(default_factory=dict)
    duration_model: Optional[object] = None
    pitch_model: Optional[object] = None
    pitch_models: Dict[Tuple[str, int], LeftToRightHMM] = field(
        default_factory=dict)
    pitch_index: Dict[Tuple[str, int], dict] = field(default_factory=dict)
    pitch_conditioning: PitchConditioning = field(
        default_factory=PitchConditioning)
    phoneme_set: Optional[PhonemeSet] = None
    spec: Optional[FeatureSpec] = None

    @classmethod
    def from_model(cls, model, *, speaker: str = "", language: str = "",
                   source: str = "") -> "VoiceSource":
        """Read the transfer-relevant parts of a trained model."""
        return cls(
            speaker=speaker or str(getattr(model, "name", "") or ""),
            language=language,
            source=source,
            hmms=dict(getattr(model, "hmms", {}) or {}),
            duration_model=getattr(model, "duration_model", None),
            pitch_model=getattr(model, "pitch_model", None),
            pitch_models=dict(getattr(model, "pitch_models", {}) or {}),
            pitch_index=dict(getattr(model, "pitch_index", {}) or {}),
            pitch_conditioning=getattr(model, "pitch_conditioning",
                                       PitchConditioning()),
            phoneme_set=getattr(model, "phoneme_set", None),
            spec=getattr(model, "spec", None),
        )

    @property
    def static_dim(self) -> int:
        return int(self.spec.static_dim) if self.spec is not None else 0

    @property
    def dim(self) -> int:
        return int(self.spec.dim) if self.spec is not None else 0

    def hmm(self, phone: str) -> Optional[LeftToRightHMM]:
        return self.hmms.get(str(phone))


# --------------------------------------------------------------------------
# Stage 1+2: the acoustic map
# --------------------------------------------------------------------------


@dataclass
class AcousticMap:
    """The MAP-shrunk affine map that carries Voice 2 into Voice 1's space.

    The map is one ``static_dim x static_dim`` matrix plus a ``static_dim``
    intercept, applied to every stream of a unit's feature vector::

        static block    : m -> matrix @ m + intercept
        dynamic streams : m -> matrix @ m               (no intercept)

    A delta is a difference of static frames, so a constant offset cancels out
    of it -- but the *linear* part does not: the same acoustic transformation
    applies to the trajectory the delta describes, which is why all streams
    share one matrix.  A full matrix (rather than one slope per dimension)
    matters because the dominant difference between two vocal tracts is a
    *warping* of the spectral envelope, and a warping mixes cepstral
    dimensions: the diagonal is the "rescale each coefficient" approximation of
    it, the off-diagonal entries the rest.

    Variances follow the same linear part, diagonalised -- component variances
    in HMS are diagonal, so an exact ``matrix @ var`` would not be one either::

        var_i -> sum_j matrix[i, j]^2 * var_j + residual_variance[i]

    The per-dimension residual is the variance the regression could not
    explain.  It is what keeps a transferred unit from becoming over-confident
    (too narrow) about a phone it was never fitted on.

    Two guards keep a degenerate anchor set from producing nonsense: the linear
    variance factor is clipped to :data:`MIN_SPREAD_FACTOR`..:data:`MAX_SPREAD_FACTOR`,
    and a mean is never moved more than :data:`MAX_MEAN_SHIFT` normalised units
    from where it already is.  Both are documented bounds, not tuning knobs.

    The map is stored in the model file (``transfer.npz``) so a saved voice
    documents the exact transformation that produced its transferred units.
    """

    matrix: np.ndarray
    intercept: np.ndarray
    residual_variance: np.ndarray
    static_dim: int
    #: Phones both voices have their own HMM for (one anchor observation each).
    anchor_phones: Tuple[str, ...] = ()
    #: Phoneme classes pooled across each voice (one anchor observation each).
    anchor_classes: Tuple[str, ...] = ()
    #: True when no anchor existed at all, so the map is the identity and the
    #: residual is the fallback spread of Voice 1's own phone means.
    identity: bool = False

    def __post_init__(self) -> None:
        self.static_dim = int(self.static_dim)
        if self.static_dim < 1:
            raise ValueError("an acoustic map needs a positive static_dim")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim == 1 and matrix.size == self.static_dim ** 2:
            matrix = matrix.reshape(self.static_dim, self.static_dim)
        self.matrix = matrix
        self.intercept = np.asarray(self.intercept,
                                    dtype=np.float64).reshape(-1)
        self.residual_variance = np.asarray(self.residual_variance,
                                            dtype=np.float64).reshape(-1)
        self.anchor_phones = tuple(sorted(str(phone)
                                          for phone in self.anchor_phones))
        self.anchor_classes = tuple(sorted(str(name)
                                           for name in self.anchor_classes))
        if self.matrix.shape != (self.static_dim, self.static_dim):
            raise ValueError(
                f"an acoustic map needs a {self.static_dim}x{self.static_dim} "
                f"matrix, got {self.matrix.shape}")
        for name in ("intercept", "residual_variance"):
            if getattr(self, name).size != self.static_dim:
                raise ValueError(
                    f"acoustic map {name} must have {self.static_dim} entries")
        if not (np.isfinite(self.matrix).all()
                and np.isfinite(self.intercept).all()
                and np.isfinite(self.residual_variance).all()):
            raise ValueError("acoustic map parameters must be finite")
        if (self.residual_variance <= 0).any():
            raise ValueError("acoustic map residual variances must be positive")

    @property
    def dim(self) -> int:
        """Feature dimensions the static block occupies (== static_dim)."""
        return self.static_dim

    @property
    def n_anchors(self) -> int:
        """Number of anchor observations the map was fitted from."""
        return len(self.anchor_phones) + len(self.anchor_classes)

    @property
    def n_streams(self) -> int:
        """Number of streams the map is applied to (1: the static block)."""
        return 1

    @property
    def linear_deviation(self) -> float:
        """Frobenius norm of ``matrix - identity``: how far from a no-op."""
        return float(np.linalg.norm(self.matrix - np.eye(self.static_dim)))

    def _block_count(self, dim: int) -> int:
        dim = int(dim)
        if dim < self.static_dim or dim % self.static_dim:
            raise ValueError(
                f"a feature vector of {dim} dimensions does not contain whole "
                f"static blocks of {self.static_dim}")
        return dim // self.static_dim

    def _block_correction(self, block_values: np.ndarray,
                          with_intercept: bool) -> np.ndarray:
        """The mean correction the map applies to one block of dimensions."""
        values = np.asarray(block_values, dtype=np.float64)
        linear = values @ (self.matrix.T - np.eye(self.static_dim))
        if with_intercept:
            linear = linear + self.intercept
        return np.clip(linear, -MAX_MEAN_SHIFT, MAX_MEAN_SHIFT)

    def apply_means(self, means: np.ndarray) -> np.ndarray:
        """Map GMM means ``(..., dim)`` into the target voice's space.

        The static block takes the whole affine map; every dynamic stream takes
        its linear part only -- a constant offset would give a delta stream a
        nonzero mean, which is not a delta.
        """
        means = np.asarray(means, dtype=np.float64)
        out = means.copy()
        streams = self._block_count(means.shape[-1])
        for stream in range(streams):
            block = slice(stream * self.static_dim,
                          (stream + 1) * self.static_dim)
            out[..., block] = means[..., block] \
                + self._block_correction(means[..., block], stream == 0)
        return out

    def apply_variances(self, variances: np.ndarray,
                        floor: float = MIN_TRANSFER_VARIANCE) -> np.ndarray:
        """Propagate the linear part through variances, plus the residual."""
        variances = np.asarray(variances, dtype=np.float64)
        out = variances.copy()
        squared = self.matrix ** 2
        for stream in range(self._block_count(variances.shape[-1])):
            block = slice(stream * self.static_dim,
                          (stream + 1) * self.static_dim)
            local = np.maximum(variances[..., block], floor)
            factor = (variances[..., block] @ squared.T) / local
            out[..., block] = np.clip(factor, MIN_SPREAD_FACTOR,
                                      MAX_SPREAD_FACTOR) \
                * variances[..., block] + self.residual_variance
        return np.maximum(out, floor)

    def describe(self) -> str:
        """One-line summary for logs and `inspect-model`."""
        rms = float(np.sqrt(self.residual_variance).mean())
        if self.identity:
            return (f"acoustic map: identity (no anchor observation), "
                    f"residual rms {rms:.3f}")
        anchors = f"{len(self.anchor_phones)} anchor phone(s)"
        if self.anchor_classes:
            anchors += f" + {len(self.anchor_classes)} class anchor(s)"
        return (f"acoustic map: {anchors}, ||A - I|| "
                f"{self.linear_deviation:.3f}, residual rms {rms:.3f}")

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "map_matrix": self.matrix,
            "map_intercept": self.intercept,
            "map_residual_variance": self.residual_variance,
            "map_static_dim": np.asarray(self.static_dim, dtype=np.int64),
            "map_identity": np.asarray(bool(self.identity)),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray],
                    anchor_phones: Sequence[str] = (),
                    anchor_classes: Sequence[str] = ()) -> "AcousticMap":
        return cls(
            matrix=arrays["map_matrix"],
            intercept=arrays["map_intercept"],
            residual_variance=arrays["map_residual_variance"],
            static_dim=int(np.asarray(arrays["map_static_dim"]).reshape(-1)[0]),
            anchor_phones=tuple(anchor_phones),
            anchor_classes=tuple(anchor_classes),
            identity=bool(np.asarray(arrays["map_identity"]).reshape(-1)[0]),
        )


def unit_dim(hmm: LeftToRightHMM) -> int:
    """Feature dimension of a unit, from its trained states (0 if untrained)."""
    for state in hmm.states:
        if state is not None:
            return int(state.gmm.dim)
    return 0


def _state_weights(hmm: LeftToRightHMM) -> np.ndarray:
    """State occupancy weights, restricted to the trained states."""
    proportions = np.asarray(hmm.duration_proportions(), dtype=np.float64)
    trained = np.asarray([state is not None for state in hmm.states])
    weights = np.where(trained, proportions, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return np.full(len(proportions), 1.0 / max(len(proportions), 1))
    return weights / total


def pooled_mean(hmm: LeftToRightHMM) -> np.ndarray:
    """Frame-weighted pooled mean of a unit's GMM means.

    State weights are the HMM's own expected state durations (how many frames
    each state takes) and each state's mean is its mixture mean.  This is the
    one summary of "where in feature space this unit sits"; using the same
    estimator on both voices is what makes the anchor pairs comparable.
    """
    dim = unit_dim(hmm)
    total = np.zeros(dim, dtype=np.float64)
    for weight, state in zip(_state_weights(hmm), hmm.states):
        if state is None:
            continue
        total += weight * (state.gmm.weights @ state.gmm.means)
    return total


def pooled_variance(hmm: LeftToRightHMM) -> np.ndarray:
    """Frame-weighted pooled variance of a unit's emissions."""
    dim = unit_dim(hmm)
    mean = pooled_mean(hmm)
    second = np.zeros(dim, dtype=np.float64)
    for weight, state in zip(_state_weights(hmm), hmm.states):
        if state is None:
            continue
        gmm = state.gmm
        second += weight * (gmm.weights @ (gmm.variances + gmm.means ** 2))
    return np.maximum(second - mean ** 2, 0.0)


def unit_spread(means: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-dimension population variance of a set of unit means."""
    if not means:
        return np.zeros(0, dtype=np.float64)
    stacked = np.stack([np.asarray(value, dtype=np.float64)
                        for _key, value in sorted(means.items())])
    if len(stacked) < 2:
        return np.zeros(stacked.shape[1], dtype=np.float64)
    return np.asarray(stacked.var(axis=0), dtype=np.float64)


def fit_acoustic_map(anchor_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                     static_dim: int,
                     prior_strength: float = DEFAULT_MAP_PRIOR_STRENGTH,
                     fallback_spread: Optional[np.ndarray] = None,
                     anchor_phones: Sequence[str] = (),
                     anchor_classes: Sequence[str] = ()
                     ) -> AcousticMap:
    """Estimate the anchor regression (stage 1) and its residual (stage 2).

    ``anchor_pairs`` are ``(Voice 2 static mean, Voice 1 static mean)``
    observations, one per anchor; a phone both voices trained and a phoneme
    class pooled in both voices are each one observation.  The map is the MAP
    (ridge-regularised) affine fit of the second on the first::

        theta = argmin  sum_p || y_p - Z_p theta ||^2 + kappa ||theta - theta0||^2
              = (Z^T Z + kappa I)^-1 (Z^T y + kappa theta0)

    with ``Z_p = [x_p, 1]``, ``theta0 = [I, 0]`` -- the *identity* map, whose
    ``kappa`` is the prior in units of anchor observations.  With no anchor at
    all the estimate is ``theta0`` exactly (which, in the per-corpus normalised
    space both voices live in, *is* mean/variance matching: a dimension has
    zero mean and unit spread in each voice by construction).  With one weak
    anchor the solution is the closest map to the identity that honour it; with
    many, the fit dominates and the prior only guards the directions the
    anchors do not span.

    The per-dimension residual is the fit's own squared error augmented with
    ``kappa`` pseudo-observations whose error is the anchor targets' spread, so
    a one-anchor fit cannot claim a zero-variance transformation.  With no
    anchors the residual falls back to ``fallback_spread`` -- Voice 1's own
    phone-to-phone spread -- or to the unit variance of the normalised space.
    """
    prior_strength = float(prior_strength)
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("map_prior_strength must be finite and positive")
    static_dim = int(static_dim)
    if static_dim < 1:
        raise ValueError("static_dim must be positive")

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for aux_mean, target_mean in anchor_pairs:
        x = np.asarray(aux_mean, dtype=np.float64).reshape(-1)[:static_dim]
        y = np.asarray(target_mean, dtype=np.float64).reshape(-1)[:static_dim]
        if x.size != static_dim or y.size != static_dim:
            raise ValueError(
                "anchor statistics must cover the whole static block "
                f"({static_dim} dimensions)")
        if np.isfinite(x).all() and np.isfinite(y).all():
            pairs.append((x, y))

    fallback = (np.full(static_dim, FALLBACK_RESIDUAL_VARIANCE)
                if fallback_spread is None
                else np.asarray(fallback_spread, dtype=np.float64).reshape(-1))
    if fallback.size != static_dim or not np.isfinite(fallback).all():
        fallback = np.full(static_dim, FALLBACK_RESIDUAL_VARIANCE)
    fallback = np.maximum(fallback, MIN_TRANSFER_VARIANCE)

    if not pairs:
        return AcousticMap(matrix=np.eye(static_dim),
                           intercept=np.zeros(static_dim),
                           residual_variance=fallback,
                           static_dim=static_dim,
                           anchor_phones=anchor_phones,
                           anchor_classes=anchor_classes,
                           identity=True)

    x_all = np.stack([x for x, _ in pairs])
    y_all = np.stack([y for _, y in pairs])
    count = len(pairs)
    design = np.hstack([x_all, np.ones((count, 1), dtype=np.float64)])
    gram = design.T @ design + prior_strength * np.eye(static_dim + 1)
    right = design.T @ y_all \
        + prior_strength * np.eye(static_dim + 1)[:, :static_dim]
    try:
        theta = np.linalg.solve(gram, right)
        # A ridge system is positive definite for kappa > 0; this only guards
        # against a numerically singular one.
    except np.linalg.LinAlgError:  # pragma: no cover - kappa > 0
        theta = np.vstack([np.eye(static_dim),
                           np.zeros((1, static_dim))])
    fit = design @ theta
    squared_error = ((fit - y_all) ** 2).sum(axis=0)
    target_spread = ((y_all - y_all.mean(axis=0)) ** 2).mean(axis=0)
    residual = (squared_error + prior_strength * target_spread) \
        / (count + prior_strength)
    # `theta` is indexed [input, output] (its rows are the coefficients of one
    # input dimension across all outputs); the map stores the [output, input]
    # form, so that `means @ matrix.T == matrix @ means`.
    return AcousticMap(matrix=theta[:static_dim].T,
                       intercept=theta[static_dim],
                       residual_variance=np.maximum(residual,
                                                    MIN_TRANSFER_VARIANCE),
                       static_dim=static_dim,
                       anchor_phones=anchor_phones,
                       anchor_classes=anchor_classes,
                       identity=False)


# --------------------------------------------------------------------------
# Applying the map to a unit (+ stage 3: MAP blend with Voice 1's own frames)
# --------------------------------------------------------------------------


@dataclass
class PhoneMoments:
    """Running first/second moments of one phone's frames in one voice.

    Accumulated over *frames*, stored as ``count`` plus two ``dim``-vectors, so
    the memory a transfer run holds is bounded by phones x dimensions and never
    by corpus length: frames are read through the training cache's memory maps
    and released immediately.
    """

    phone: str = ""
    count: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sumsquares: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self.count = int(self.count)
        self.sums = np.asarray(self.sums, dtype=np.float64).reshape(-1)
        self.sumsquares = np.asarray(self.sumsquares,
                                     dtype=np.float64).reshape(-1)
        if self.sums.shape != self.sumsquares.shape:
            raise ValueError("sums and sumsquares must have the same shape")

    @property
    def dim(self) -> int:
        return int(self.sums.size)

    @property
    def mean(self) -> np.ndarray:
        if self.count <= 0:
            return np.zeros(self.dim, dtype=np.float64)
        return self.sums / float(self.count)

    @property
    def variance(self) -> np.ndarray:
        """Population variance over this phone's frames (strictly positive)."""
        if self.count <= 0:
            return np.zeros(self.dim, dtype=np.float64)
        return np.maximum(self.sumsquares / float(self.count)
                          - self.mean ** 2, 1e-12)

    @property
    def nbytes(self) -> int:
        return int(self.sums.nbytes + self.sumsquares.nbytes)

    def to_dict(self) -> Dict[str, object]:
        return {"phone": self.phone, "count": int(self.count),
                "mean": [float(value) for value in self.mean],
                "variance": [float(value) for value in self.variance]}


def accumulate_phone_moments(moments: Dict[str, PhoneMoments], phone: str,
                             frames: np.ndarray, static_dim: int) -> None:
    """Fold a batch of frames into a phone's running moments (static block)."""
    block = np.asarray(frames, dtype=np.float64)
    if block.ndim == 1:
        block = block[None, :]
    if block.size == 0:
        return
    block = block[:, :int(static_dim)]
    entry = moments.get(phone)
    if entry is None:
        entry = PhoneMoments(phone=phone,
                             sums=np.zeros(int(static_dim), dtype=np.float64),
                             sumsquares=np.zeros(int(static_dim),
                                                dtype=np.float64))
        moments[phone] = entry
    entry.sums += block.sum(axis=0)
    entry.sumsquares += (block ** 2).sum(axis=0)
    entry.count += int(block.shape[0])


def phone_moments_nbytes(moments: Mapping[str, PhoneMoments]) -> int:
    """Total bytes the accumulated phone moments occupy."""
    return int(sum(entry.nbytes for entry in moments.values()))


def map_hmm(hmm: LeftToRightHMM, acoustic_map: AcousticMap,
            shift: Optional[np.ndarray] = None,
            spread: Optional[np.ndarray] = None) -> LeftToRightHMM:
    """A copy of ``hmm`` with its emissions mapped into the target's space.

    Means and variances go through ``acoustic_map``; the optional ``shift``
    (added to every dimension) and ``spread`` (multiplied into every variance)
    carry the pooled MAP correction of stage 3.  Mixture weights, self-loops,
    state durations and voicing probabilities are copied unchanged: the
    transferred unit keeps language B's timing and phonation, and only its
    spectral statistics are adapted.
    """
    mapped = LeftToRightHMM(n_states=hmm.n_states, allow_skip=hmm.allow_skip,
                            covariance_type=hmm.covariance_type)
    mapped.self_loops = np.array(hmm.self_loops, dtype=np.float64)
    mapped.dim = hmm.dim or unit_dim(hmm)
    for index, state in enumerate(hmm.states):
        if state is None:  # pragma: no cover - trained models have no gaps
            continue
        gmm = state.gmm
        means = acoustic_map.apply_means(gmm.means)
        variances = acoustic_map.apply_variances(gmm.variances)
        if shift is not None:
            means = means + np.asarray(shift, dtype=np.float64)
        if spread is not None:
            variances = variances * np.asarray(spread, dtype=np.float64)
        mapped.states[index] = HMMState(
            gmm=DiagGMM(np.array(gmm.weights, dtype=np.float64), means,
                        variances, gmm.covariance_type),
            duration=StateDurationStats(state.duration.mean,
                                        state.duration.variance,
                                        state.duration.count),
            voiced_prob=float(state.voiced_prob))
    return mapped


def map_correction(mapped: LeftToRightHMM, moments: Optional[PhoneMoments],
                   static_dim: int, adapt_frames: float
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """The stage-3 MAP mean/variance correction for one transferred unit.

    Returns ``(shift, spread, record)``: a per-dimension location shift and a
    per-dimension variance factor, built from Voice 1's own frames for the phone
    (``moments``) and the prior strength ``adapt_frames``.  With no Voice 1
    frames -- the phone is absent from the target corpus -- both are the
    identity, which is the honest answer: there is nothing to blend with.
    """
    dim = unit_dim(mapped) or int(static_dim)
    static_dim = min(int(static_dim), dim)
    frames = float(adapt_frames)
    if moments is None or moments.count <= 0 or frames <= 0:
        # No target-side observation of this phone at all: the mapped unit
        # stands alone (the correction is the identity), which the record says
        # with the same keys as every other unit's record.
        return (np.zeros(dim, dtype=np.float64),
                np.ones(dim, dtype=np.float64),
                {"adapted_frames": 0, "adaptation_weight": 0.0,
                 "mean_shift_rms": 0.0, "spread_mean": 1.0})

    prior_mean = pooled_mean(mapped)
    prior_variance = pooled_variance(mapped)
    target_mean = moments.mean[:static_dim]
    target_variance = moments.variance[:static_dim]
    weight = float(moments.count) / (float(moments.count) + frames)

    final_mean = weight * target_mean + (1.0 - weight) * prior_mean[:static_dim]
    prior_static = np.maximum(prior_variance[:static_dim],
                              MIN_TRANSFER_VARIANCE)
    final_variance = weight * target_variance + (1.0 - weight) * prior_static
    scale = np.clip(final_variance / prior_static, MIN_SPREAD_FACTOR,
                    MAX_SPREAD_FACTOR)

    shift = np.zeros(dim, dtype=np.float64)
    shift[:static_dim] = final_mean - prior_mean[:static_dim]
    # The spread factor lives on the static dimensions and is repeated over the
    # dynamic streams derived from them.
    spread = np.ones(dim, dtype=np.float64)
    for stream in range(dim // static_dim):
        spread[stream * static_dim:(stream + 1) * static_dim] = scale
    record = {
        "adapted_frames": int(moments.count),
        "adaptation_weight": round(float(weight), 6),
        "mean_shift_rms": round(float(np.sqrt(np.mean(shift ** 2))), 6),
        "spread_mean": round(float(scale.mean()), 6),
    }
    return shift, spread, record


def map_pitch_stat(stat: PitchStats, acoustic_map: AcousticMap) -> PitchStats:
    """Map a per-state relative-pitch statistic through dimension 0.

    Dimension 0 is note-relative F0 in semitones, part of the same normalised
    feature block, so the same map applies to it.  A per-state pitch statistic
    is a scalar, not a feature vector, so only dimension 0's own row of the map
    can be used (a pitch statistic carries no information about the other
    dimensions to mix with).  These statistics are only read by the optional
    ``state_means`` / ``acoustic`` pitch sources; the default score-driven F0
    path never touches them.
    """
    slope = float(acoustic_map.matrix[0, 0])
    intercept = float(acoustic_map.intercept[0])
    residual = float(acoustic_map.residual_variance[0])
    return PitchStats(mean=slope * float(stat.mean) + intercept,
                      variance=slope ** 2 * float(stat.variance) + residual,
                      count=int(stat.count))


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------


@dataclass
class TransferBuild:
    """Everything one transfer run contributes to the target model."""

    record: CrossLanguageTransfer = field(
        default_factory=CrossLanguageTransfer)
    #: phone -> transferred HMM (never shadows a native unit)
    units: Dict[str, LeftToRightHMM] = field(default_factory=dict)
    index: Dict[str, dict] = field(default_factory=dict)
    acoustic_map: Optional[AcousticMap] = None
    #: phone -> (shift, spread) already applied to that phone's unit
    corrections: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict)
    #: (phone, pitch bin) -> transferred pitch-conditioned HMM
    pitch_models: Dict[Tuple[str, int], LeftToRightHMM] = field(
        default_factory=dict)
    pitch_index: Dict[Tuple[str, int], dict] = field(default_factory=dict)
    #: language-B duration statistics, for the target's duration model
    duration_stats: Dict[str, DurationStats] = field(default_factory=dict)
    #: language-B relative-pitch statistics, mapped into the target's space
    pitch_stats: Dict[str, List[PitchStats]] = field(default_factory=dict)
    #: language-B voicing priors
    voiced_prior: Dict[str, float] = field(default_factory=dict)
    #: inventory entries the target's phoneme set is missing
    phoneme_definitions: Dict[str, PhonemeDef] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.record.active and self.units)

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_pitch_units(self) -> int:
        return len(self.pitch_models)

    @property
    def n_anchors(self) -> int:
        return len(self.record.anchor_phones)

    def describe(self) -> List[str]:
        """Human-readable summary lines (the trainer logs these)."""
        if not self.record.active:
            return ["cross-language transfer: disabled"]
        lines = [f"  {self.record.describe()}"]
        if self.acoustic_map is not None:
            lines.append(f"  {self.acoustic_map.describe()}")
        if self.units:
            lines.append("  transferred phones: "
                         + _listing(sorted(self.units)))
        if self.pitch_models:
            bins = sorted({bin_index for _unit, bin_index
                           in self.pitch_models})
            lines.append(f"  transferred pitch-conditioned units: "
                         f"{len(self.pitch_models)} over bin(s) "
                         + _listing([str(value) for value in bins]))
        if self.phoneme_definitions:
            lines.append("  inventory entries imported from the auxiliary "
                         "voice: " + _listing(sorted(self.phoneme_definitions)))
        lines.extend(f"  ! {message}" for message in self.diagnostics)
        return lines


def _listing(values: Sequence[str], limit: int = 8) -> str:
    """``a, b, c, ... (12 phones)`` -- bounded so a log line stays a line."""
    values = list(values)
    head = ", ".join(values[:limit])
    if len(values) > limit:
        head += f", ... ({len(values)})"
    return head


def check_compatibility(target: VoiceSource, auxiliary: VoiceSource,
                        request: CrossLanguageTransfer) -> None:
    """Refuse combinations that would silently mean something else.

    Raises ``ValueError`` for a feature-space mismatch (the two voices' numbers
    would not be comparable at all) and for a pitch-bin width mismatch when the
    auxiliary's pitch-conditioned units are about to be transferred.
    """
    if target.spec is None or auxiliary.spec is None:
        return
    if target.spec.to_dict() != auxiliary.spec.to_dict():
        differences = sorted(
            key for key in set(target.spec.to_dict())
            | set(auxiliary.spec.to_dict())
            if target.spec.to_dict().get(key)
            != auxiliary.spec.to_dict().get(key))
        raise ValueError(
            "cross-language transfer needs both voices to share one feature "
            f"definition (differing: {', '.join(differences)}); train the "
            "auxiliary voice with the same acoustic settings as the target")
    if not request.adapt_pitch_bins or not target.pitch_conditioning.active:
        return
    if not auxiliary.pitch_conditioning.active or not auxiliary.pitch_models:
        return
    if int(auxiliary.pitch_conditioning.bin_size) \
            != int(target.pitch_conditioning.bin_size):
        raise ValueError(
            "cross-language transfer cannot reinterpret the auxiliary voice's "
            f"pitch bins ({auxiliary.pitch_conditioning.bin_size} semitones) as "
            f"the target's ({target.pitch_conditioning.bin_size} semitones); "
            "train the auxiliary voice with the same "
            "pitch_conditioning.bin_size, or set transfer.adapt_pitch_bins to "
            "false to transfer its phones without their pitch conditions")


def _canonical_units(source: VoiceSource, canonical) -> Dict[str, LeftToRightHMM]:
    """``phone -> HMM`` under the target inventory's canonical symbols."""
    units: Dict[str, LeftToRightHMM] = {}
    for phone in sorted(source.hmms):
        units.setdefault(canonical(phone), source.hmms[phone])
    return units


def build_transfer(*, target: VoiceSource, auxiliary: VoiceSource,
                   request: CrossLanguageTransfer,
                   moments: Optional[Mapping[str, PhoneMoments]] = None,
                   min_phone_frames: int = 1,
                   log=None) -> TransferBuild:
    """Build the transferred units for ``target`` from ``auxiliary``.

    The target's own units are only ever *read* (for the anchor regression): no
    native HMM, normalisation, duration statistic or pitch statistic is modified
    here.  The result is returned, not installed, so the caller stays in charge
    of assembling the model.
    """
    log = log or (lambda message: None)
    if not request.active:
        return TransferBuild(record=request)
    check_compatibility(target, auxiliary, request)

    phoneme_set = target.phoneme_set
    canonical = phoneme_set.canonical if phoneme_set is not None \
        else (lambda symbol: symbol)
    static_dim = target.static_dim or auxiliary.static_dim
    dim = target.dim or auxiliary.dim
    if static_dim < 1 or dim < static_dim:
        raise ValueError("cross-language transfer needs a feature spec on both "
                         "voices")

    target_units = _canonical_units(target, canonical)
    auxiliary_units = _canonical_units(auxiliary, canonical)

    # -- anchors ----------------------------------------------------------
    # Anchors are the phones both voices trained their own model for -- the
    # tightest correspondence available (same symbol, same segment type, two
    # speakers).  Only when the two languages share no such phone does the map
    # fall back to pooled phoneme-class anchors, which are coarser (the two
    # classes do not hold the same phones) but exist in every language.
    anchors = sorted(set(target_units) & set(auxiliary_units))
    aux_means = {phone: pooled_mean(auxiliary_units[phone])[:static_dim]
                 for phone in anchors}
    target_means = {phone: pooled_mean(target_units[phone])[:static_dim]
                    for phone in anchors}
    if anchors:
        class_anchors: List[str] = []
        class_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    else:
        class_anchors, class_pairs = _class_anchors(
            target, auxiliary, target_units, auxiliary_units, static_dim)
    anchor_pairs = [(aux_means[phone], target_means[phone])
                    for phone in anchors]
    anchor_pairs.extend(class_pairs)

    fallback_spread = unit_spread(
        {phone: pooled_mean(hmm)[:static_dim]
         for phone, hmm in target_units.items()})
    acoustic_map = fit_acoustic_map(
        anchor_pairs, static_dim,
        prior_strength=request.map_prior_strength,
        fallback_spread=fallback_spread,
        anchor_phones=anchors, anchor_classes=class_anchors)

    build = TransferBuild(record=request, acoustic_map=acoustic_map)
    if not anchors:
        build.diagnostics.append(
            "the two voices share no trained phone, so the acoustic map rests "
            "on the pooled class anchors alone (or is the identity if neither "
            "voice has usable statistics): the transferred units keep the "
            "auxiliary voice's phone geometry, placed in the target's acoustic "
            "space, but no per-phone correspondence could be estimated")

    # -- which phones are transferred (native models always win) ----------
    transferred: List[str] = []
    for phone in sorted(auxiliary_units):
        if phone in target_units:
            continue
        definition = None
        if auxiliary.phoneme_set is not None:
            definition = auxiliary.phoneme_set.resolve(phone)
        if definition is not None and phoneme_set is not None \
                and phoneme_set.resolve(phone) is None:
            # The target inventory does not describe this phone at all: import
            # the auxiliary's definition (states, components, voiced flag) so
            # the saved model can describe its own units.  Existing entries are
            # never overwritten.
            build.phoneme_definitions[phone] = definition
        transferred.append(phone)

    # -- the units ---------------------------------------------------------
    for phone in transferred:
        entry_moments = None if moments is None else moments.get(phone)
        mapped = map_hmm(auxiliary_units[phone], acoustic_map)
        shift, spread, record = map_correction(mapped, entry_moments,
                                               static_dim,
                                               request.map_adapt_frames)
        unit = map_hmm(auxiliary_units[phone], acoustic_map,
                       shift=shift, spread=spread)
        build.units[phone] = unit
        build.corrections[phone] = (shift, spread)
        build.index[phone] = {
            "kind": TIER_TRANSFERRED,
            "phone": phone,
            "source_speaker": request.auxiliary_speaker,
            "source_language": request.auxiliary_language,
            "source": request.auxiliary_source,
            "anchors": len(anchors),
            "class_anchors": len(class_anchors),
            "n_states": unit.n_states,
            "n_components": unit.states[0].gmm.n_components,
            "covariance": unit.covariance_type,
            "allow_skip": bool(unit.allow_skip),
            "n_free_params": unit.n_free_params,
            **record,
        }

    # -- language-B duration / pitch / voicing statistics ------------------
    for phone in transferred:
        stats = _duration_stats(auxiliary, phone)
        if stats is not None:
            build.duration_stats[phone] = stats
        pitch_stats = _pitch_stats(auxiliary, phone)
        if pitch_stats:
            build.pitch_stats[phone] = [map_pitch_stat(stat, acoustic_map)
                                        for stat in pitch_stats]
        prior = _voiced_prior(auxiliary, phone)
        if prior is not None:
            build.voiced_prior[phone] = prior

    # -- pitch-conditioned units -------------------------------------------
    _build_pitch_units(build, target=target, auxiliary=auxiliary,
                       request=request, transferred=transferred,
                       min_phone_frames=min_phone_frames)

    build.record = replace(
        request,
        anchor_phones=tuple(anchors),
        anchor_classes=tuple(class_anchors),
        transferred_phones=tuple(transferred),
        transferred_pitch_units=len(build.pitch_models))
    for line in build.describe():
        log(line)
    return build


def _phoneme_class(phoneme_set: Optional[PhonemeSet],
                   phone: str) -> Optional[str]:
    """Inventory class of a phone (``vowel``, ``silence``, ...), if known."""
    if phoneme_set is None:
        return None
    definition = phoneme_set.resolve(phone)
    return definition.type if definition is not None else None


def _class_anchors(target: VoiceSource, auxiliary: VoiceSource,
                   target_units: Mapping[str, LeftToRightHMM],
                   auxiliary_units: Mapping[str, LeftToRightHMM],
                   static_dim: int
                   ) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """Pooled per-class anchor observations shared by the two voices.

    For every phoneme class both voices trained at least one phone of, the
    anchor is the mean of those phones' pooled means in each voice.  Classes
    are language-independent (they come from the inventory, not from the
    phones), which is what makes them the fallback when the two languages share
    no phone at all -- but they are only a fallback: two voices' "vowel" pools
    hold different phones, so the pairing is biased by whatever else each
    corpus contains, and a real phone-to-phone correspondence is always the
    better anchor.
    """
    aux_by_class: Dict[str, List[str]] = {}
    for phone in auxiliary_units:
        klass = _phoneme_class(auxiliary.phoneme_set, phone)
        if klass:
            aux_by_class.setdefault(klass, []).append(phone)
    target_by_class: Dict[str, List[str]] = {}
    for phone in target_units:
        klass = _phoneme_class(target.phoneme_set, phone)
        if klass:
            target_by_class.setdefault(klass, []).append(phone)

    names: List[str] = []
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for klass in sorted(set(aux_by_class) & set(target_by_class)):
        aux_mean = np.mean([pooled_mean(auxiliary_units[phone])[:static_dim]
                            for phone in aux_by_class[klass]], axis=0)
        target_mean = np.mean([pooled_mean(target_units[phone])[:static_dim]
                               for phone in target_by_class[klass]], axis=0)
        names.append(klass)
        pairs.append((aux_mean, target_mean))
    return names, pairs


def _build_pitch_units(build: TransferBuild, *, target: VoiceSource,
                       auxiliary: VoiceSource,
                       request: CrossLanguageTransfer,
                       transferred: Sequence[str],
                       min_phone_frames: int) -> None:
    """Install the auxiliary's pitch-conditioned units for transferred phones.

    Only meaningful when the *target* was trained with the pitch-conditioned
    tier: a pitch condition is a model-selection device, so a transferred bin
    only has something to select between if the target voice selects by pitch.
    Each bucket is mapped with the same acoustic map and the same MAP correction
    as its phone-level unit, so a phone and its pitch-conditioned bins stay one
    coherent model.
    """
    if not transferred:
        return
    if not request.adapt_pitch_bins:
        return
    if not target.pitch_conditioning.active:
        if auxiliary.pitch_models:
            build.diagnostics.append(
                "the auxiliary voice carries pitch-conditioned units but the "
                "target was trained without pitch conditioning, so they were "
                "mapped at the phone level only (train the target with "
                "pitch_conditioning.enabled to keep the pitch conditions)")
        return
    if not auxiliary.pitch_models:
        build.diagnostics.append(
            "the auxiliary voice has no pitch-conditioned units, so "
            "transferred phones are covered at the phone tier only (retrain "
            "the auxiliary voice with pitch conditioning to transfer its "
            "pitch-binned units)")
        return

    wanted = set(transferred)
    skipped = 0
    for (unit_key, pitch_bin), hmm in sorted(auxiliary.pitch_models.items()):
        if unit_key not in wanted:
            continue
        info = auxiliary.pitch_index.get((unit_key, int(pitch_bin))) or {}
        try:
            frames = int(info.get("frames", 0))
        except (TypeError, ValueError):
            frames = 0
        if frames < int(min_phone_frames):
            skipped += 1
            continue
        shift, spread = build.corrections.get(
            unit_key, (np.zeros(build.acoustic_map.dim),
                       np.ones(build.acoustic_map.dim)))
        mapped = map_hmm(hmm, build.acoustic_map, shift=shift, spread=spread)
        key = (unit_key, int(pitch_bin))
        build.pitch_models[key] = mapped
        build.pitch_index[key] = {
            "kind": KIND_PHONE,
            "unit": unit_key,
            "curr": unit_key,
            "pitch_bin": int(pitch_bin),
            # The marker that makes this unit auditable at synthesis time: a
            # `phone+pitch` tier hit whose record says `transferred`.
            "transferred": True,
            "source_speaker": request.auxiliary_speaker,
            "source_language": request.auxiliary_language,
            "frames": frames,
            "occurrences": int(info.get("occurrences", 0) or 0),
            "anchors": len(build.record.anchor_phones),
            "adapted_frames": (build.index.get(unit_key) or {})
            .get("adapted_frames", 0),
            "n_states": mapped.n_states,
            "n_components": mapped.states[0].gmm.n_components,
            "covariance": mapped.covariance_type,
            "allow_skip": bool(mapped.allow_skip),
            "n_free_params": mapped.n_free_params,
        }
    if skipped:
        build.diagnostics.append(
            f"{skipped} auxiliary pitch-conditioned unit(s) were not "
            f"transferred: fewer than {int(min_phone_frames)} frames in that "
            "pitch bin (a bin holds a subset of its phone's data)")
    if not build.pitch_models:
        build.diagnostics.append(
            "no auxiliary pitch-conditioned unit cleared the support "
            "threshold, so every transferred phone is covered at the phone "
            "tier and its pitch bins fall back to it")


def _duration_stats(auxiliary: VoiceSource, phone: str
                    ) -> Optional[DurationStats]:
    model = auxiliary.duration_model
    stats = getattr(model, "stats", {}).get(phone) \
        if model is not None else None
    if stats is None:
        return None
    copied = DurationStats(float(stats.mean), float(stats.variance),
                           int(stats.count))
    copied.frames = float(np.exp(copied.mean))
    return copied


def _pitch_stats(auxiliary: VoiceSource, phone: str) -> List[PitchStats]:
    model = auxiliary.pitch_model
    stats = getattr(model, "stats", {}).get(phone) \
        if model is not None else None
    return list(stats or [])


def _voiced_prior(auxiliary: VoiceSource, phone: str) -> Optional[float]:
    model = auxiliary.pitch_model
    prior = getattr(model, "voiced_prior", {}).get(phone) \
        if model is not None else None
    return None if prior is None else float(prior)
