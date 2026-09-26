"""Native WORLD backend: real WORLD vocoder over a ctypes C ABI.

The WORLD sources are compiled into `libhms_world.so` by
``tools/build_world.sh`` (see ``tools/world_native/hms_world_capi.cpp``).  This
module loads that library, so HMS gets the genuine WORLD algorithm set --
DIO/Harvest for F0, StoneMask for pitch refinement, CheapTrick for the spectral
envelope, D4C for aperiodicity, and Synthesis for the waveform -- without
assuming any pip package exists.

If the shared library has not been built yet and a compiler is present, the
first use attempts a build automatically.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np

from hms.core.dsp import harmonicity_aperiodicity, is_degenerate_aperiodicity
from hms.core.features import AcousticFrameSequence
from hms.vocoder.base import Vocoder, VocoderUnavailable

_NATIVE_DIR = Path(__file__).resolve().parent / "_native"
_LIB_NAMES = ("libhms_world.so", "libhms_world.dylib", "hms_world.dll")


def _find_library() -> Path | None:
    for name in _LIB_NAMES:
        candidate = _NATIVE_DIR / name
        if candidate.exists():
            return candidate
    return None


def _try_build(timeout: int = 600) -> Path | None:
    """Attempt `tools/build_world.sh` once; return the library path on success."""
    root = Path(__file__).resolve().parents[2]
    script = root / "tools" / "build_world.sh"
    if not script.exists() or os.environ.get("HMS_NO_AUTO_BUILD"):
        return None
    try:
        subprocess.run(["bash", str(script)], check=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception:
        return None
    return _find_library()


class NativeWorldVocoder(Vocoder):
    """WORLD vocoder bound through ``libhms_world.so``."""

    name = "native"

    def __init__(self, fft_size: int | None = None, fs: int = 44100,
                 frame_period: float = 5.0, auto_build: bool = True) -> None:
        self._lib_path = _find_library()
        if self._lib_path is None and auto_build:
            self._lib_path = _try_build()
        if self._lib_path is None:
            raise VocoderUnavailable(
                "libhms_world.so not found. Build it with "
                "`tools/build_world.sh` (needs a C++ compiler).")
        try:
            self._lib = ctypes.CDLL(str(self._lib_path))
        except OSError as exc:  # pragma: no cover - platform specific
            raise VocoderUnavailable(f"cannot load {self._lib_path}: {exc}")
        self.last_aperiodicity_fallback = False
        self._bind()
        self._native_fft_size = int(self._lib.hms_fft_size(int(fs)))
        super().__init__(fft_size=fft_size or self._native_fft_size, fs=fs,
                         frame_period=frame_period)
        # remember whether the caller really asked for this size: when it did
        # not, the size follows the sample rate (see `resolve_fft_size`)
        self._fft_size_pinned = fft_size is not None

    # -- ctypes plumbing ---------------------------------------------------

    def _bind(self) -> None:
        lib = self._lib
        d, i, pd = ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double)

        lib.hms_f0_length.restype = i
        lib.hms_f0_length.argtypes = [i, i, d]
        lib.hms_fft_size.restype = i
        lib.hms_fft_size.argtypes = [i]
        lib.hms_synth_length.restype = i
        lib.hms_synth_length.argtypes = [i, i, d]
        lib.hms_f0.restype = i
        lib.hms_f0.argtypes = [pd, i, i, d, d, d, i, i, pd, pd]
        lib.hms_spectral_envelope.restype = i
        lib.hms_spectral_envelope.argtypes = [pd, i, i, pd, pd, i, d, pd]
        lib.hms_aperiodicity.restype = i
        lib.hms_aperiodicity.argtypes = [pd, i, i, pd, pd, i, d, pd]
        lib.hms_synthesize.restype = None
        lib.hms_synthesize.argtypes = [pd, i, pd, pd, i, d, i, i, pd]

    @staticmethod
    def _ptr(array: np.ndarray):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # -- Vocoder interface -------------------------------------------------

    @property
    def fft_size(self) -> int:
        return self._fft_size

    def fft_size_for(self, fs: int) -> int:
        """WORLD chooses CheapTrick's FFT size from the sample rate (2**ceil
        such that 3 periods of the lowest F0 fit; 2048 at 44.1 kHz, 1024 at
        22.05 kHz).  Analysing with any other size scrambles the parameter
        layout, so the two must agree."""
        return int(self._lib.hms_fft_size(int(fs)))

    def _consistent_fft_size(self, fs: int) -> int:
        """Reject an explicit size that WORLD cannot honour at this rate."""
        return self.resolve_fft_size(fs)

    def analyze(self, x: np.ndarray, fs: int | None = None,
                frame_period: float | None = None,
                f0_floor: float = 71.0, f0_ceil: float = 800.0,
                f0_estimation: str = "dio", refine_f0: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fs = int(fs or self.default_fs)
        frame_period = float(frame_period or self.default_frame_period)
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
        n = len(x)

        alg = 1 if f0_estimation == "harvest" else 0
        n_frames = int(self._lib.hms_f0_length(fs, n, frame_period))
        times = np.zeros(n_frames, dtype=np.float64)
        f0 = np.zeros(n_frames, dtype=np.float64)
        self._lib.hms_f0(self._ptr(x), n, fs, frame_period, float(f0_floor),
                         float(f0_ceil), alg, int(bool(refine_f0)),
                         self._ptr(times), self._ptr(f0))

        fft_size = self._consistent_fft_size(fs)
        bins = fft_size // 2 + 1
        sp = np.zeros((n_frames, bins), dtype=np.float64)
        ap = np.zeros((n_frames, bins), dtype=np.float64)
        written = int(self._lib.hms_spectral_envelope(
            self._ptr(x), n, fs, self._ptr(times), self._ptr(f0), n_frames,
            frame_period, self._ptr(sp)))
        if written != bins:                     # defensive: never accept
            raise RuntimeError(f"CheapTrick wrote {written} bins, expected "
                               f"{bins} -- FFT size mismatch")   # pragma: no cover
        written = int(self._lib.hms_aperiodicity(
            self._ptr(x), n, fs, self._ptr(times), self._ptr(f0), n_frames,
            frame_period, self._ptr(ap)))
        if written != bins:                     # pragma: no cover
            raise RuntimeError(f"D4C wrote {written} bins, expected {bins} "
                               f"-- FFT size mismatch")

        # D4C saturates to aperiodicity == 1.0 for every frame of a *voiced*
        # signal when its internal VUV guard (D4CLoveTrain) rejects the frame,
        # which happens on pathologically clean, noise-free material.  Silently
        # training on that would teach the model "all voiced frames are noise",
        # so detect it and fall back to a harmonicity estimate.
        if is_degenerate_aperiodicity(ap, f0 > 0):
            warnings.warn(
                "WORLD D4C returned a degenerate (all-noise) aperiodicity; "
                "falling back to the harmonicity estimator. This usually means "
                "the input has no noise floor.", RuntimeWarning, stacklevel=2)
            power = np.maximum(sp, 1e-30)
            _, ap = harmonicity_aperiodicity(power, f0, fs, n_band=5)
            self.last_aperiodicity_fallback = True
        else:
            self.last_aperiodicity_fallback = False
        return f0, sp, ap

    def synthesize(self, params: AcousticFrameSequence) -> np.ndarray:
        f0 = np.ascontiguousarray(np.asarray(params.f0, dtype=np.float64)
                                  .reshape(-1))
        n_frames = len(f0)
        if n_frames == 0:
            return np.zeros(0, dtype=np.float64)
        fs = int(params.fs or self.default_fs)
        frame_period = float(params.frame_period or self.default_frame_period)
        fft_size = self._consistent_fft_size(fs)
        bins = fft_size // 2 + 1
        # the bin count is the real contract: WORLD needs exactly `bins` values
        # per frame, and a stale `params.fft_size` cannot mislead us

        sp = np.ascontiguousarray(np.asarray(params.sp, dtype=np.float64)
                                  .reshape(n_frames, -1))
        ap = np.ascontiguousarray(np.asarray(params.ap, dtype=np.float64)
                                  .reshape(n_frames, -1))
        if sp.shape[1] != bins:
            raise ValueError(f"spectral envelope has {sp.shape[1]} bins, "
                             f"backend fft_size={fft_size} needs {bins}")
        # WORLD tolerates ap == 0 but a tiny floor avoids brittle excitation.
        ap = np.clip(ap, 1e-4, 1.0)

        y_length = int(self._lib.hms_synth_length(n_frames, fs, frame_period))
        y = np.zeros(y_length, dtype=np.float64)
        self._lib.hms_synthesize(self._ptr(f0), n_frames, self._ptr(sp),
                                 self._ptr(ap), fft_size, frame_period, fs,
                                 y_length, self._ptr(y))
        return self._limit(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))

    @staticmethod
    def _limit(y: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:
            y = y / (peak * 1.02)
        return y
