"""Optional backend: the `pyworld` pip package.

Preferred when available (pip wheels exist for common platforms), otherwise the
identical code path is served by `hms.vocoder.world_native`.  Keeping both
around costs ~60 lines and removes a build requirement from the user's plate.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from hms.core.features import AcousticFrameSequence
from hms.vocoder.base import Vocoder, VocoderUnavailable


class PyWorldVocoder(Vocoder):
    """WORLD via the `pyworld` bindings."""

    name = "pyworld"

    def __init__(self, fft_size: int | None = None, fs: int = 44100,
                 frame_period: float = 5.0) -> None:
        try:
            import pyworld
        except Exception as exc:  # pragma: no cover - depends on install
            raise VocoderUnavailable(f"pyworld is not importable: {exc}")
        self._pyworld = pyworld
        native_fft = int(pyworld.get_fft_size(fs, frame_period))
        super().__init__(fft_size=fft_size or native_fft, fs=fs,
                         frame_period=frame_period)

    @staticmethod
    def available() -> bool:
        try:
            import pyworld  # noqa: F401
            return True
        except Exception:
            return False

    @property
    def fft_size(self) -> int:
        return int(self._fft_size)

    def analyze(self, x: np.ndarray, fs: int | None = None,
                frame_period: float | None = None, f0_floor: float = 71.0,
                f0_ceil: float = 800.0, f0_estimation: str = "dio",
                refine_f0: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pw = self._pyworld
        fs = int(fs or self.default_fs)
        frame_period = float(frame_period or self.default_frame_period)
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))

        if f0_estimation == "harvest":
            f0, times = pw.harvest(x, fs, f0_floor=f0_floor, f0_ceil=f0_ceil,
                                   frame_period=frame_period)
        else:
            f0, times = pw.dio(x, fs, f0_floor=f0_floor, f0_ceil=f0_ceil,
                               frame_period=frame_period)
        if refine_f0:
            f0 = pw.stonemask(x, f0, times, fs)
        sp = pw.cheaptrick(x, f0, times, fs, fft_size=self.fft_size)
        ap = pw.d4c(x, f0, times, fs, fft_size=self.fft_size)
        return f0, sp, ap

    def synthesize(self, params: AcousticFrameSequence) -> np.ndarray:
        pw = self._pyworld
        f0 = np.ascontiguousarray(np.asarray(params.f0, dtype=np.float64)
                                  .reshape(-1))
        if len(f0) == 0:
            return np.zeros(0)
        sp = np.ascontiguousarray(np.asarray(params.sp, dtype=np.float64))
        ap = np.clip(np.asarray(params.ap, dtype=np.float64), 1e-4, 1.0)
        fs = int(params.fs or self.default_fs)
        frame_period = float(params.frame_period or self.default_frame_period)
        y = pw.synthesize(f0, sp, np.ascontiguousarray(ap), fs, frame_period,
                          fft_size=int(params.fft_size or self.fft_size))
        # keep both WORLD backends consistent: no hidden peak normalisation
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
