"""Vocoder interface shared by every backend."""

from __future__ import annotations

import abc
from typing import Dict, Optional, Tuple

import numpy as np

from hms.core.features import AcousticFrameSequence


class VocoderUnavailable(RuntimeError):
    """Raised when a backend cannot be used on this machine."""


def render_length(n_frames: int, frame_period: float, fs: int) -> int:
    """Samples a render of ``n_frames`` frames produces.

    This is WORLD's duration convention (``f0_length * frame_period * fs``),
    which every backend reproduces so that a phrase's length does not depend on
    the backend -- and, since Phase 3, so that a source excitation rendered
    outside the backend is laid out on exactly the grid the backend filters.
    Note that it is *not* ``n_frames * hop``: at 44.1 kHz / 5 ms a frame
    advances by 220.5 samples, so the two drift apart by half a sample per
    frame over a long render.
    """
    return int(int(n_frames) * float(frame_period) / 1000.0 * int(fs))


def blend_excitation(default: np.ndarray, learned: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Mix a caller-supplied excitation into the backend's own, per sample.

        ``out = weights * learned + (1 - weights) * default``

    ``weights`` is ``(n_samples,)`` in ``[0, 1]``, aligned with ``learned``:
    ``1`` hands the sample to the learned source, ``0`` keeps the backend's
    pulse train.  It is supplied by the source layer
    (:func:`hms.source.synthesis.fade_weights`), which ramps it over one frame
    at every coverage boundary -- a step here would be a step in the filter's
    input.  ``None`` means "take the learned source wherever it is non-zero",
    the hard switch, which exists for callers that do their own fade.

    ``learned`` shorter than ``default`` is zero-padded rather than rejected:
    a backend internally renders more samples than it outputs (the
    overlap-add tail), and the uncovered tail is exactly where its own
    excitation belongs.
    """
    default = np.asarray(default, dtype=np.float64).reshape(-1)
    learned = np.asarray(learned, dtype=np.float64).reshape(-1)
    n = max(default.size, learned.size)
    if learned.size != n:
        padded = np.zeros(n, dtype=np.float64)
        padded[:learned.size] = learned
        learned = padded
    if default.size != n:
        padded = np.zeros(n, dtype=np.float64)
        padded[:default.size] = default
        default = padded
    if weights is None:
        weights = (learned != 0.0).astype(np.float64)
    weights = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if weights.size != n:
        padded = np.zeros(n, dtype=np.float64)
        padded[:min(weights.size, n)] = weights[:min(weights.size, n)]
        weights = padded
    return default + (learned - default) * weights


def limit_peak(audio: np.ndarray, ceiling: float = 1.0,
               headroom: float = 1.02) -> np.ndarray:
    """Scale ``audio`` down so its peak sits at ``ceiling / headroom``.

    Headroom is the *writer's* job -- :func:`hms.data.wavio.write_wav` already
    normalises whatever it is given -- so backends do not apply this to their
    output.  It is provided for callers that want a signal guaranteed to fit in
    ``[-1, 1]`` (e.g. before handing samples to a fixed-point device).

    A backend's ``synthesize`` must therefore return the waveform exactly as
    produced; anything else is a hidden, material-dependent gain.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        audio = audio * (ceiling / (peak * headroom))
    return audio


class Vocoder(abc.ABC):
    """Analyse audio into WORLD parameters and synthesise audio back.

    Implementations must agree on the parameter conventions used across HMS:

    ``f0``  (T,)      Hz; 0.0 means unvoiced
    ``sp``  (T, F+1)  linear power spectral envelope
    ``ap``  (T, F+1)  aperiodicity, 0 (periodic) .. 1 (noise)
    """

    name: str = "abstract"

    #: True when `synthesize_with_excitation` is implemented.  WORLD-style
    #: backends take (f0, sp, ap) and nothing else, so this stays False for
    #: them; overriding it is how a backend opts into source-aware synthesis.
    supports_external_excitation: bool = False

    def __init__(self, fft_size: int | None = None, fs: int = 44100,
                 frame_period: float = 5.0) -> None:
        self._fft_size = fft_size
        #: True when the caller pinned `fft_size` explicitly (see
        #: `fft_size_for` / `_resolve_fft_size`): an explicit mismatch is an
        #: error, an unpinned default follows the backend.
        self._fft_size_pinned = fft_size is not None
        self.default_fs = fs
        self.default_frame_period = frame_period

    # -- geometry ----------------------------------------------------------

    @property
    @abc.abstractmethod
    def fft_size(self) -> int:
        """FFT size the backend uses for the spectral envelope."""

    @property
    def n_bins(self) -> int:
        """Bins per frame at this backend's *default* sample rate.

        Backends whose FFT size depends on the sample rate (WORLD) vary this;
        use `fft_size_for(fs)` / the frame sequence's own ``fft_size`` when the
        rate is not the default.
        """
        return self.fft_size // 2 + 1

    def resolve_fft_size(self, fs: int) -> int:
        """FFT size to use for ``fs``, rejecting an explicit mismatch."""
        expected = self.fft_size_for(fs)
        if self._fft_size_pinned and int(self.fft_size) != expected:
            raise ValueError(
                f"this backend uses fft_size={expected} at {fs} Hz, but it was "
                f"configured with fft_size={self.fft_size}; pass fft_size=None "
                f"to follow the backend")
        return expected

    def fft_size_for(self, fs: int) -> int:
        """FFT size this backend uses when analysing ``fs``.

        Backends that derive the FFT size from the sample rate (WORLD does)
        override this; the rest ignore ``fs``.  Callers must use it instead of
        assuming `fft_size` applies to every sample rate, otherwise the feature
        vectors and the vocoder disagree about how many bins a frame has.
        """
        return self.fft_size

    # -- core operations ---------------------------------------------------

    @abc.abstractmethod
    def analyze(self, x: np.ndarray, fs: int | None = None,
                frame_period: float | None = None,
                f0_floor: float = 71.0, f0_ceil: float = 800.0,
                f0_estimation: str = "dio", refine_f0: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Waveform -> (f0, sp, ap)."""

    @abc.abstractmethod
    def synthesize(self, params: AcousticFrameSequence) -> np.ndarray:
        """WORLD parameters -> float64 waveform.

        The waveform is returned exactly as the backend synthesised it: no
        peak normalisation is applied here, because a gain that depends on the
        signal's peak would make otherwise identical parameters produce
        different output depending on what else is in the utterance.  WORLD's
        synthesis can exceed +/-1 on high-peak (very periodic) material, so
        normalise before writing: :func:`hms.data.wavio.write_wav` does, or call
        :func:`limit_peak` for a hard ceiling.
        """

    def synthesize_with_excitation(self, params: AcousticFrameSequence,
                                   excitation: np.ndarray,
                                   weights: Optional[np.ndarray] = None
                                   ) -> np.ndarray:
        """Filter a caller-supplied excitation instead of generating one.

        This is the Phase-3 seam: the source layer renders an excitation
        (see :mod:`hms.source.synthesis`) and the backend -- which owns the
        filter -- splices it in where it covers the timeline, keeping its own
        excitation everywhere else.  Only the *periodic* component is replaced:
        aperiodicity and the unvoiced noise path are untouched, so an unvoiced
        frame stays exactly as it is today.

        A backend that builds its excitation inside a closed synthesis call
        (WORLD: ``Synthesis(f0, sp, ap)``) cannot honour this, and says so by
        leaving :attr:`supports_external_excitation` False -- the default.  The
        synthesis layer turns that into an error naming the backends that do,
        rather than silently ignoring the caller's source model.
        """
        raise VocoderUnavailable(
            f"the {self.name!r} backend cannot filter a caller-supplied "
            f"excitation (it synthesizes f0/sp/ap internally); source-aware "
            f"synthesis needs a backend that owns its excitation -- use "
            f"'builtin' or 'mlsa'")

    # -- convenience -------------------------------------------------------

    def analyze_to_sequence(self, x: np.ndarray, fs: int,
                            frame_period: float = 5.0, **kwargs
                            ) -> AcousticFrameSequence:
        f0, sp, ap = self.analyze(x, fs=fs, frame_period=frame_period, **kwargs)
        return AcousticFrameSequence(f0=f0, sp=sp, ap=ap,
                                     frame_period=frame_period, fs=fs,
                                     fft_size=self.fft_size_for(fs))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} fft={self.fft_size}>"
