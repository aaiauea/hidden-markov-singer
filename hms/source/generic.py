"""Frame-synchronous residual backend: the pitch-free source model.

The voice backend assumes quasi-periodicity, because that is what a singing
voice has.  A wind instrument, a cymbal, a breathy consonant or a percussive hit
does not, and a source representation that only spoke "pitch period" would
silently exclude them.  This backend is the other axis of the design space::

    audio ──► inverse filter (mel-cepstrum) ──► one residual window per frame
                                                    │
                                                    └──► cycle_length samples

It is the *same* residual, the *same* fixed-length normalisation and therefore
the same `SourceSequence`, `SourcePCA` and `SourceModel.encode/decode/
synthesize` as the voice backend -- the only difference is where the unit
boundaries come from.  No F0 is needed: each analysis frame contributes one
unit, so ``periods`` is the hop and ``epochs`` is the frame grid.  An optional F0
track is accepted and stored as annotation (voicing, per-frame ``f0``), but it
never changes the segmentation: this backend does not consider "unpitched" a
defect.

Two honest limitations, both inherent to any frame-synchronous source model and
both visible in the numbers a benchmark prints:

* a window shorter than one period cannot contain a whole excitation event, so
  the per-unit ``noise_level`` describes the window, not a period;
* the fixed-length vector low-passes the window to ``cycle_length / 2``
  harmonics, which for 5 ms frames is a ~12 kHz band edge at 44.1 kHz.  Raise
  ``cycle_length`` if the source has meaningful energy above that.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hms.core.features import DEFAULT_N_MCEP
from hms.source.base import (DEFAULT_CYCLE_LENGTH, SourceModel, SourceSequence,
                             assign_frame_noise)
from hms.source.cycles import CycleSet, noise_level, resample_cycle
from hms.source.residual import default_fft_size, whiten

#: A unit needs at least this many samples of residual to describe anything.
MIN_UNIT_SAMPLES = 4


class GenericResidualSourceModel(SourceModel):
    """Frame-synchronous whitened residual, one source unit per analysis frame.

    Parameters
    ----------
    cycle_length
        Samples per source vector (128 by default).
    fs, frame_period
        Defaults when the caller does not pass geometry to :meth:`analyze`.
    unit_frames
        Analysis frames each unit spans (1 by default).  A larger value gives
        each vector a longer stretch of the residual at the cost of time
        resolution -- useful for very low-pitched or slowly varying sources.
    fft_size, n_mcep, f_min
        Inverse-filter geometry, shared with the voice backend.
    resample
        ``"fft"`` (band-limited; default) or ``"linear"``.
    """

    name = "residual"

    def __init__(self, cycle_length: int = DEFAULT_CYCLE_LENGTH, fs: int = 44100,
                 frame_period: float = 5.0, fft_size: Optional[int] = None,
                 n_mcep: int = DEFAULT_N_MCEP, f_min: float = 0.0,
                 unit_frames: int = 1, resample: str = "fft", seed: int = 0) -> None:
        super().__init__(cycle_length=cycle_length, fs=fs,
                         frame_period=frame_period, n_mcep=n_mcep, f_min=f_min,
                         seed=seed)
        if int(unit_frames) < 1:
            raise ValueError("unit_frames must be at least 1")
        self.unit_frames = int(unit_frames)
        self.fft_size = None if fft_size is None else int(fft_size)
        self.resample = str(resample)

    def analyze(self, x: np.ndarray, fs: Optional[int] = None,
                frame_period: Optional[float] = None,
                f0: Optional[np.ndarray] = None) -> SourceSequence:
        """Waveform -> one fixed-length residual unit per analysis frame."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size and not np.isfinite(x).all():
            raise ValueError("audio must be finite")
        fs = int(fs or self.default_fs)
        frame_period = float(frame_period or self.default_frame_period)
        hop = self.hop(fs, frame_period)
        n_frames = int(np.ceil(x.size / hop)) if x.size else 0
        track = self._frame_track(f0, n_frames)
        voiced = track > 0.0

        if x.size == 0:
            return SourceSequence(
                f0=track, voiced=voiced,
                noise_level=np.where(voiced, 0.0, 1.0),
                excitation=np.zeros((0, self.cycle_length)), gains=np.zeros(0),
                epochs=np.zeros(0, dtype=np.int64),
                periods=np.zeros(0, dtype=np.int64), cycle_length=self.cycle_length,
                fs=fs, frame_period=frame_period, n_samples=0, backend=self.name)

        fft_size = int(self.fft_size or default_fft_size(fs, frame_period))
        residual = whiten(x, fs, frame_period, fft_size=fft_size,
                          n_mcep=self.n_mcep, f_min=self.f_min)

        step = self.unit_frames * hop
        epochs = np.arange(0, x.size, step, dtype=np.int64)
        # The last unit is extended to the end of the signal rather than being
        # dropped: a percussive tail is exactly the part one least wants to lose.
        stops = np.minimum(epochs + step, x.size)
        cycles = self._extract_units(residual, epochs, stops)

        noise = np.where(voiced, 0.0, 1.0)
        noise = assign_frame_noise(cycles.noise, cycles.epochs, cycles.periods,
                                   noise, hop)
        return SourceSequence(
            f0=track, voiced=voiced, noise_level=noise, excitation=cycles.vectors,
            gains=cycles.gains, epochs=cycles.epochs, periods=cycles.periods,
            cycle_length=self.cycle_length, fs=fs, frame_period=frame_period,
            n_samples=int(x.size), backend=self.name)

    # -- helpers -----------------------------------------------------------

    def _extract_units(self, residual: np.ndarray, epochs: np.ndarray,
                       stops: np.ndarray) -> CycleSet:
        """Fixed-length normalisation of frame-sized residual windows."""
        vectors, gains, noise = [], [], []
        kept_epochs, kept_periods = [], []
        for epoch, stop in zip(epochs.tolist(), stops.tolist()):
            window = residual[int(epoch):int(stop)]
            if window.size < MIN_UNIT_SAMPLES or not np.isfinite(window).all():
                continue
            rms = float(np.sqrt(np.mean(window * window)))
            if not np.isfinite(rms) or rms <= 1e-9:
                continue   # digital silence: nothing to represent
            vector = resample_cycle(window, self.cycle_length, self.resample)
            if not np.isfinite(vector).all():
                continue
            # Normalise after resampling so the vector is exactly unit RMS and
            # the gain restores exactly what was normalised away (see
            # `extract_cycles`, which explains the convention).
            norm = float(np.sqrt(np.mean(vector * vector)))
            if not np.isfinite(norm) or norm <= 1e-12:
                continue
            vectors.append(vector / norm)
            gains.append(norm)
            noise.append(noise_level(vector))
            kept_epochs.append(int(epoch))
            kept_periods.append(int(stop - epoch))
        if not vectors:
            return CycleSet(vectors=np.zeros((0, self.cycle_length)),
                            gains=np.zeros(0), noise=np.zeros(0),
                            epochs=np.zeros(0, dtype=np.int64),
                            periods=np.zeros(0, dtype=np.int64))
        return CycleSet(
            vectors=np.asarray(vectors, dtype=np.float64).reshape(-1,
                                                                  self.cycle_length),
            gains=np.asarray(gains, dtype=np.float64),
            noise=np.asarray(noise, dtype=np.float64),
            epochs=np.asarray(kept_epochs, dtype=np.int64),
            periods=np.asarray(kept_periods, dtype=np.int64))

    @staticmethod
    def _frame_track(f0: Optional[np.ndarray], n_frames: int) -> np.ndarray:
        """Optional F0 annotation -> a finite, non-negative track of n_frames."""
        track = np.zeros(max(0, int(n_frames)), dtype=np.float64)
        if f0 is None or not track.size:
            return track
        given = np.asarray(f0, dtype=np.float64).reshape(-1)
        given = np.where(np.isfinite(given) & (given > 0.0), given, 0.0)
        keep = min(track.size, given.size)
        track[:keep] = given[:keep]      # a short/long track is clipped, not stretched
        return track


__all__ = ["GenericResidualSourceModel", "MIN_UNIT_SAMPLES"]
