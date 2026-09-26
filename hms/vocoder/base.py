"""Vocoder interface shared by every backend."""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np

from hms.core.features import AcousticFrameSequence


class VocoderUnavailable(RuntimeError):
    """Raised when a backend cannot be used on this machine."""


class Vocoder(abc.ABC):
    """Analyse audio into WORLD parameters and synthesise audio back.

    Implementations must agree on the parameter conventions used across HMS:

    ``f0``  (T,)      Hz; 0.0 means unvoiced
    ``sp``  (T, F+1)  linear power spectral envelope
    ``ap``  (T, F+1)  aperiodicity, 0 (periodic) .. 1 (noise)
    """

    name: str = "abstract"

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
        """WORLD parameters -> float64 waveform in [-1, 1]."""

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
