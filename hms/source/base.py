"""Generic source/excitation representation and the source-model interface.

Why this exists
---------------
HMS's acoustic model describes the *filter*: a mel-cepstrum spectral envelope
per frame which the HMM/GMM learns.  The *source* is left to the vocoder, which
invents it from scratch -- a pulse train plus noise mixed by the aperiodicity.
That is a perfectly good synthesiser, but it means the excitation is a fixed
recipe rather than something a model can learn, and the recipe only knows how
to be a voice.

This package is **Phase 1** of giving the excitation its own representation,
deliberately without touching the acoustic model, the parameter generation or
the vocoder.  The eventual architecture is::

                     ┌── spectral model ──→ spectral envelope ──┐
        HMM output ──┤                                           ├─→ filter ─→ audio
                     └── source model ────→ excitation ─────────┘

A source is a sequence of **units** -- one pitch period for a voice, one
analysis frame for an unpitched or unknown source -- and a unit is described by
:class:`SourceFrame`::

    SourceFrame
        f0                   Hz of the unit (0 when the source has no pitch)
        voiced               whether the unit is pitched
        excitation           fixed-length source vector (128 samples)
        source_coefficients  compact (PCA) coefficients for it, if encoded
        noise_level          0 = one clean excitation event, 1 = noise-like
        gain                 level removed by the fixed-length normalisation

Nothing in this file knows about glottal pulses, vowels or voices: the voice
backend (:mod:`hms.source.voice`) does.  A future backend for a wind
instrument, a bowed string or a drum only has to fill the same fields --
:meth:`SourceModel.analyze` in, :meth:`SourceModel.synthesize` out -- and gets
PCA, testing and the eventual HMM coupling for free.

Conventions
-----------
* Sample rate and frame period follow the rest of HMS: F0 is Hz with ``0.0``
  meaning unvoiced, frame ``t`` is centred on sample ``t * hop`` with
  ``hop = round(fs * frame_period / 1000)``.
* ``excitation`` is *normalised*: every vector has the same length and unit
  RMS, and the level it lost is stored in ``gain``.  Shape (what a model can
  learn) and level (what the spectral envelope already carries) are kept apart
  on purpose.
* A unit that cannot be extracted is *dropped*, never filled with a malformed
  vector: silence, unvoiced regions, cycles running past the end of the signal
  and degenerate periods simply do not appear in ``excitation``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from hms.core.features import DEFAULT_N_MCEP
from hms.source.cycles import frame_coverage, place_cycles, unit_support_mask

#: Default length of one fixed-length source vector, in samples.
DEFAULT_CYCLE_LENGTH = 128

#: Smallest sensible fixed-length source vector (below this, a "cycle" is
#: barely more than a couple of samples and the representation is meaningless).
MIN_CYCLE_LENGTH = 8


@dataclass
class SourceFrame:
    """One source unit: a pitch period, or one analysis frame.

    Attributes
    ----------
    f0
        Unit frequency in Hz (``fs / period`` for cycle backends); ``0.0`` when
        the unit has no pitch.
    voiced
        Whether the unit is pitched.
    excitation
        ``(cycle_length,)`` fixed-length, unit-RMS source vector, or ``None``
        when only the analysis track describes this frame.
    source_coefficients
        Compact coefficients from :class:`hms.source.pca.SourcePCA`, when the
        unit has been encoded.
    noise_level
        Fraction of the unit that is *not* a single clean excitation event,
        in ``[0, 1]``.  A cheap proxy (see :func:`hms.source.cycles.noise_level`)
        for how much noise a synthesis backend should mix in.
    gain
        RMS of the excitation before the fixed-length normalisation, so
        ``excitation * gain * scale`` restores the analysed level.
    epoch, period
        Sample index of the unit's start and the number of samples it spans.
        ``period`` is exact (it is the measured pitch period for a cycle
        backend), ``f0`` is derived from it.
    index
        Position of the unit in its :class:`SourceSequence`, or ``-1``.
    """

    f0: float = 0.0
    voiced: bool = False
    excitation: Optional[np.ndarray] = None
    source_coefficients: Optional[np.ndarray] = None
    noise_level: float = 0.0
    gain: float = 1.0
    epoch: int = -1
    period: int = 0
    index: int = -1

    @property
    def is_valid(self) -> bool:
        """True when this unit carries a usable fixed-length excitation."""
        if self.excitation is None:
            return False
        vector = np.asarray(self.excitation, dtype=np.float64).reshape(-1)
        return bool(vector.size) and bool(np.isfinite(vector).all())


@dataclass
class SourceSequence:
    """The source side of one utterance: an analysis track plus its units.

    ``f0``/``voiced``/``noise_level`` are **per analysis frame** (the same frame
    grid the acoustic features use), while ``excitation``/``gains``/``epochs``/
    ``periods`` are **per source unit** (one pitch period for the voice
    backend, one frame for the frame-synchronous backend).  Keeping the two
    grids explicit is what lets a source model be plugged in next to the
    acoustic model later without re-defining time.
    """

    f0: np.ndarray                       # (T,) Hz, 0 = unvoiced
    voiced: np.ndarray                   # (T,) bool
    noise_level: np.ndarray              # (T,) in [0, 1]
    excitation: np.ndarray               # (K, cycle_length), unit RMS
    gains: np.ndarray                    # (K,)
    epochs: np.ndarray                   # (K,) sample index of the unit start
    periods: np.ndarray                  # (K,) samples spanned
    cycle_length: int = DEFAULT_CYCLE_LENGTH
    fs: int = 44100
    frame_period: float = 5.0
    n_samples: int = 0
    coefficients: Optional[np.ndarray] = None
    backend: str = ""

    def __post_init__(self) -> None:
        self.f0 = np.asarray(self.f0, dtype=np.float64).reshape(-1)
        self.voiced = np.asarray(self.voiced, dtype=bool).reshape(-1)
        self.noise_level = np.asarray(self.noise_level, dtype=np.float64).reshape(-1)
        self.excitation = np.atleast_2d(np.asarray(self.excitation, dtype=np.float64))
        if self.excitation.shape[1] != int(self.cycle_length):
            raise ValueError(
                f"excitation must have shape (units, {self.cycle_length}), got "
                f"{self.excitation.shape}")
        self.gains = np.asarray(self.gains, dtype=np.float64).reshape(-1)
        self.epochs = np.asarray(self.epochs, dtype=np.int64).reshape(-1)
        self.periods = np.asarray(self.periods, dtype=np.int64).reshape(-1)
        for name in ("gains", "epochs", "periods"):
            if len(getattr(self, name)) != self.n_units:
                raise ValueError(f"{name} must have one entry per source unit")
        for name in ("voiced", "noise_level"):
            if len(getattr(self, name)) != self.n_frames:
                raise ValueError(f"{name} must have one entry per analysis frame")
        if self.coefficients is not None:
            self.coefficients = np.atleast_2d(
                np.asarray(self.coefficients, dtype=np.float64))
            if len(self.coefficients) != self.n_units:
                raise ValueError("coefficients must have one row per source unit")
        self.fs = int(self.fs)
        self.n_samples = int(self.n_samples)

    # -- geometry ----------------------------------------------------------

    def __len__(self) -> int:
        return self.n_frames

    @property
    def n_frames(self) -> int:
        return int(self.f0.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.epochs.shape[0])

    @property
    def hop(self) -> int:
        return max(1, int(round(self.fs * self.frame_period / 1000.0)))

    @property
    def duration(self) -> float:
        return float(self.n_frames * self.frame_period / 1000.0)

    @property
    def voiced_fraction(self) -> float:
        """Share of analysis frames the F0 track calls voiced."""
        return float(self.voiced.mean()) if self.n_frames else 0.0

    @property
    def unit_f0(self) -> np.ndarray:
        """Per-unit frequency in Hz, derived from the measured periods."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.periods > 0, self.fs / np.maximum(self.periods, 1), 0.0)

    def support_mask(self, n_samples: Optional[int] = None) -> np.ndarray:
        """``(n_samples,)`` bool: the samples a source unit spans.

        Added for Phase 3: a renderer that mixes a learned source with the
        backend's own excitation has to know where the learned one actually
        exists (see :func:`hms.source.cycles.unit_support_mask`).  The default
        length is the sequence's own ``n_samples``; pass one to ask about a
        longer or shorter render.
        """
        length = int(self.n_samples if n_samples is None else n_samples)
        return unit_support_mask(self.epochs, self.periods, length)

    @property
    def coverage(self) -> np.ndarray:
        """Per-frame covered fraction in ``[0, 1]`` (one entry per frame)."""
        return frame_coverage(self.support_mask(), self.hop, self.n_frames)

    # -- generic view ------------------------------------------------------

    def frames(self) -> Iterator[SourceFrame]:
        """Yield one :class:`SourceFrame` per source unit."""
        unit_f0 = self.unit_f0
        coefficients = self.coefficients
        for i in range(self.n_units):
            yield SourceFrame(
                f0=float(unit_f0[i]),
                voiced=bool(unit_f0[i] > 0.0),
                excitation=self.excitation[i],
                source_coefficients=None if coefficients is None else coefficients[i],
                noise_level=float(self._frame_noise_for(i)),
                gain=float(self.gains[i]),
                epoch=int(self.epochs[i]),
                period=int(self.periods[i]),
                index=i,
            )

    def _frame_noise_for(self, unit: int) -> float:
        """Noise level of the analysis frame nearest to this unit's centre."""
        if not self.n_frames:
            return 0.0
        frame = int(round((self.epochs[unit] + 0.5 * self.periods[unit]) / self.hop))
        frame = min(max(frame, 0), self.n_frames - 1)
        return float(self.noise_level[frame])

    def with_coefficients(self, coefficients: np.ndarray) -> "SourceSequence":
        """Return a copy carrying per-unit coefficients (from ``SourceModel.encode``)."""
        clone = SourceSequence(
            f0=self.f0, voiced=self.voiced, noise_level=self.noise_level,
            excitation=self.excitation, gains=self.gains, epochs=self.epochs,
            periods=self.periods, cycle_length=self.cycle_length, fs=self.fs,
            frame_period=self.frame_period, n_samples=self.n_samples,
            coefficients=coefficients, backend=self.backend)
        return clone

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<SourceSequence {self.backend or 'source'} "
                f"frames={self.n_frames} units={self.n_units} "
                f"cycle_length={self.cycle_length} fs={self.fs}>")


class SourceModel(abc.ABC):
    """Interface every source/excitation backend implements.

        model = get_source_model("voice")
        sequence = model.analyze(audio, fs)          # audio   -> cycles
        coefficients = model.encode(sequence, pca)   # cycles  -> coefficients
        cycles = model.decode(coefficients, pca)     # coefficients -> cycles
        residual = model.synthesize(sequence)        # cycles  -> excitation signal

    ``encode``/``decode``/``synthesize`` are concrete here because they are the
    same for every backend: they are the PCA codec plus the placement of
    fixed-length units back on the sample grid.  Only the analysis
    (:meth:`analyze`) is backend specific -- that is the piece a new physical
    source type has to supply.

    This is Phase 1 infrastructure: nothing in the synthesiser calls these
    methods yet, so adding a backend cannot change what HMS renders today.
    """

    name: str = "abstract"

    def __init__(self, cycle_length: int = DEFAULT_CYCLE_LENGTH, fs: int = 44100,
                 frame_period: float = 5.0, n_mcep: int = DEFAULT_N_MCEP,
                 f_min: float = 0.0, seed: int = 0) -> None:
        if int(cycle_length) < MIN_CYCLE_LENGTH:
            raise ValueError(f"cycle_length must be >= {MIN_CYCLE_LENGTH}, "
                             f"got {cycle_length}")
        self.cycle_length = int(cycle_length)
        self.default_fs = int(fs)
        self.default_frame_period = float(frame_period)
        self.n_mcep = int(n_mcep)
        self.f_min = float(f_min)
        self.seed = int(seed)

    # -- analysis (backend specific) ---------------------------------------

    @abc.abstractmethod
    def analyze(self, x: np.ndarray, fs: Optional[int] = None,
                frame_period: Optional[float] = None,
                f0: Optional[np.ndarray] = None, **kwargs) -> SourceSequence:
        """Waveform -> :class:`SourceSequence`.

        ``f0`` (Hz, 0/NaN = unvoiced, one value per analysis frame) is optional:
        a backend that can estimate pitch does so when it is missing, and a
        backend that does not need pitch ignores it.
        """

    # -- codec (shared) ----------------------------------------------------

    def encode(self, source, pca) -> np.ndarray:
        """Fixed-length source vectors -> compact coefficients.

        ``source`` is a :class:`SourceSequence` or a ``(K, cycle_length)`` array;
        ``pca`` a fitted :class:`hms.source.pca.SourcePCA`.
        """
        return pca.encode(self._cycles_of(source))

    def decode(self, coefficients, pca, gains: Optional[np.ndarray] = None
               ) -> np.ndarray:
        """Compact coefficients -> fixed-length source vectors.

        With ``gains`` (per unit) the analysed level is restored as well; without
        them the vectors stay unit RMS, which is the natural domain for
        comparing shapes.
        """
        cycles = pca.decode(coefficients)
        if gains is not None:
            gains = np.asarray(gains, dtype=np.float64).reshape(-1)
            if len(gains) != len(cycles):
                raise ValueError("gains must have one entry per coefficient row")
            cycles = cycles * gains[:, None]
        return cycles

    def synthesize(self, source, pca=None, restore_gain: bool = True) -> np.ndarray:
        """Place the source vectors back on the sample grid.

        This renders the *excitation* the model extracted, not audio: no
        spectral filter is applied (that is the synthesiser's job, and hooking
        it up is Phase 2).  Voiced spans are reconstructed from their cycles;
        unvoiced spans are left silent because their excitation in HMS is the
        vocoder's noise, which the source model does not replace yet.

        Pass ``pca`` to synthesise from the compact coefficients instead of the
        stored vectors; ``restore_gain=False`` keeps the vectors unit RMS.
        """
        if not isinstance(source, SourceSequence):
            raise TypeError("synthesize expects a SourceSequence")
        cycles = source.excitation
        if pca is not None:
            if source.coefficients is None:
                raise ValueError("this sequence has no coefficients; call encode "
                                 "before synthesising from a PCA")
            cycles = pca.decode(source.coefficients)
        gains = source.gains if restore_gain else np.ones(source.n_units)
        return place_cycles(cycles * gains[:, None], source.epochs, source.periods,
                            source.n_samples)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _cycles_of(source) -> np.ndarray:
        if isinstance(source, SourceSequence):
            return source.excitation
        cycles = np.atleast_2d(np.asarray(source, dtype=np.float64))
        if cycles.size and not np.isfinite(cycles).all():
            raise ValueError("source vectors must be finite")
        return cycles

    def hop(self, fs: int, frame_period: float) -> int:
        """Samples per analysis frame (the frame grid convention of HMS)."""
        return max(1, int(round(int(fs) * float(frame_period) / 1000.0)))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<{type(self).__name__} name={self.name!r} "
                f"cycle_length={self.cycle_length}>")


def assign_frame_noise(noise_per_unit: np.ndarray, unit_starts: np.ndarray,
                       unit_periods: np.ndarray, track: np.ndarray, hop: int,
                       mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Copy per-unit noise levels onto a per-frame track.

    Frame ``t`` takes the value of the unit whose centre (``epoch + period/2``)
    is nearest to sample ``t * hop``; frames ``mask`` excludes keep whatever
    ``track`` already holds.  Shared by the backends so both grids agree on what
    a frame's noise level means: frame 0.0 = a clean excitation event, 1.0 =
    noise.  Unvoiced frames are 1.0 by convention -- their excitation *is* noise
    -- and a voiced frame no unit covers stays 0.0.
    """
    track = np.asarray(track, dtype=np.float64).reshape(-1).copy()
    if not track.size or len(unit_starts) == 0:
        return track
    centres = np.arange(len(track), dtype=np.float64) * float(hop)
    mids = (np.asarray(unit_starts, dtype=np.float64)
            + 0.5 * np.asarray(unit_periods, dtype=np.float64))
    order = np.argsort(mids)
    mids = mids[order]
    values = np.asarray(noise_per_unit, dtype=np.float64)[order]

    right = np.clip(np.searchsorted(mids, centres), 0, len(mids) - 1)
    left = np.clip(right - 1, 0, len(mids) - 1)
    nearest = np.where(np.abs(mids[left] - centres) <= np.abs(mids[right] - centres),
                       left, right)
    painted = values[nearest]
    if mask is None:
        return painted
    return np.where(np.asarray(mask, dtype=bool).reshape(-1), painted, track)


__all__ = ["SourceFrame", "SourceSequence", "SourceModel", "assign_frame_noise",
           "DEFAULT_CYCLE_LENGTH", "MIN_CYCLE_LENGTH"]
