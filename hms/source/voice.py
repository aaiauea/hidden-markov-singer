"""Voice source backend: pitch-synchronous residual cycles.

    audio ──► F0 (given, or estimated) ──► voiced/unvoiced ──► epochs
              │                                   │
              └──► inverse filter (mel-cepstrum) ─┴──► one residual cycle per
                   (hms.source.residual)                period, resampled to
                                                        cycle_length samples

This is the first concrete source backend and it is deliberately a *voice*
backend: the excitation of a singing voice is quasi-periodic, so one pitch
period is the natural unit of the source representation.  The generic pieces it
builds on (the residual, the cycle machinery, the PCA, the `SourceModel`
interface) are not voice specific -- a later backend for a lip-reed, a bowed
string or a percussive hit fills the same fields from its own analysis.

Details that matter
-------------------
* **F0 comes from outside when the caller has it.**  A vocoder or a score-derived
  contour is better than a second estimator, so ``analyze(..., f0=...)`` uses it
  as-is.  Without it, the same pure-numpy autocorrelation + spectral-flatness
  estimator the fallback vocoder uses (``hms.core.dsp``) runs here, bounded by
  ``f0_floor``/``f0_ceil`` -- no new dependency, the same frame grid, and by
  default the same window, lag range, threshold and voicing decisions, so the
  rest of HMS sees what it would have seen.  ``f0_window_periods`` optionally
  stretches that window to a few periods of the lowest expected pitch: a window
  holding only two or three periods of the *actual* pitch can lock onto a strong
  formant or harmonic instead of the fundamental, and the same measure over a
  longer window is unambiguous.  It is off by default because it costs time
  resolution (and can lock an octave low on high material); see
  ``docs/source_model.md`` for the measurements.
* **A bad F0 never produces a bad vector.**  Non-finite, negative and
  below-floor/above-ceiling values are unvoiced or clamped periods; cycles that
  run past the signal, contain no energy or are shorter than a few samples are
  dropped by :func:`hms.source.cycles.extract_cycles`.  The result is always a
  ``(K, cycle_length)`` array of finite, unit-RMS vectors, including for empty,
  fully unvoiced and very short input.
* **One cycle, one period.**  A cycle spans exactly the distance between two
  epochs, so it crosses analysis frames, notes and file boundaries safely: the
  frame grid is used only for F0 and for the per-frame noise track.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from hms.core.dsp import frame_signal, spectral_flatness
from hms.core.features import DEFAULT_N_MCEP
from hms.source.base import (DEFAULT_CYCLE_LENGTH, SourceModel, SourceSequence,
                             assign_frame_noise)
from hms.source.cycles import MIN_PERIOD_SAMPLES, extract_cycles, pick_epochs
from hms.source.residual import default_fft_size, whiten

#: Frames flatter than this are called unvoiced whatever the autocorrelation
#: peak says (mirrors `BuiltinVocoder.FLATNESS_THRESHOLD`).
FLATNESS_THRESHOLD = 0.30


class VoiceSourceModel(SourceModel):
    """Pitch-synchronous residual extractor (the voice-oriented backend).

    Parameters
    ----------
    cycle_length
        Samples per source vector (128 by default).
    fs, frame_period
        Defaults when the caller does not pass geometry to :meth:`analyze`.
    fft_size
        Analysis window for the inverse filter; ``None`` -> ``4 * hop`` rounded
        to a power of two (1024 at 44.1 kHz).
    n_mcep
        Order of the mel-cepstrum envelope removed from the signal.  This is the
        *model's* envelope definition, reused, so the residual is free of
        exactly the colour the acoustic model fits.
    f0_floor, f0_ceil
        Source pitch range in Hz.  They bound the F0 *estimator* when no track is
        given and clamp the periods when one is (a value outside the range is
        rendered at the nearest supported period -- it is not turned into a
        hole in the excitation).
    refine_epochs
        Snap epochs onto the strongest residual sample nearby (see
        :func:`hms.source.cycles.pick_epochs`).  Measured on the demo corpus
        (k = 8 held-out relative RMSE / cycle-to-cycle correlation): off
        0.713/--, ratio 0.15 0.657/0.940, 0.25 0.604/0.950, 0.35 0.569/0.957,
        0.50 0.529/0.965.  ``0.35`` is the default: a wider snap keeps improving
        the numbers because the excitation event is what it locks onto, but past
        half a period an epoch could reach its neighbour's event, which is the
        failure mode the neighbour bound exists to prevent.
    f0_smoothing
        Median window (in frames, odd, 1 = off) applied to the F0 track *this
        backend estimates*.  Single-frame octave errors are the most common
        failure of an autocorrelation F0 estimator and a 5-frame median removes
        them without touching sustained notes.  A caller-supplied ``f0`` is
        never smoothed: that track is authoritative.
    f0_window_periods
        Stretch the F0 estimator's window to hold at least this many periods of
        ``f0_floor`` (never fewer than four frame periods).  ``0`` -- the
        default -- keeps the plain ``4 * hop`` window the rest of HMS uses,
        which tracks pitch and vibrato closely; a positive value (2-3 is a good
        range) resolves low pitches and formant/octave lock-in much better at the
        cost of a smeared pitch contour, so it is for material with a low
        ``f0_floor`` and little pitch movement.  The measured trade-off on the
        demo corpus is in ``docs/source_model.md``.
    resample
        ``"fft"`` (periodic, band-limited; default) or ``"linear"``.
    """

    name = "voice"

    def __init__(self, cycle_length: int = DEFAULT_CYCLE_LENGTH, fs: int = 44100,
                 frame_period: float = 5.0, fft_size: Optional[int] = None,
                 n_mcep: int = DEFAULT_N_MCEP, f_min: float = 0.0,
                 f0_floor: float = 50.0, f0_ceil: float = 2000.0,
                 refine_epochs: bool = True, refine_ratio: float = 0.35,
                 f0_smoothing: int = 5, f0_window_periods: float = 0.0,
                 resample: str = "fft", seed: int = 0) -> None:
        super().__init__(cycle_length=cycle_length, fs=fs,
                         frame_period=frame_period, n_mcep=n_mcep, f_min=f_min,
                         seed=seed)
        if f0_floor <= 0 or f0_ceil <= f0_floor:
            raise ValueError("F0 range must satisfy 0 < f0_floor < f0_ceil")
        self.fft_size = None if fft_size is None else int(fft_size)
        self.f0_floor = float(f0_floor)
        self.f0_ceil = float(f0_ceil)
        self.refine_epochs = bool(refine_epochs)
        self.refine_ratio = float(refine_ratio)
        self.f0_smoothing = int(f0_smoothing)
        self.f0_window_periods = float(f0_window_periods)
        self.resample = str(resample)

    # -- analysis ----------------------------------------------------------

    def analyze(self, x: np.ndarray, fs: Optional[int] = None,
                frame_period: Optional[float] = None,
                f0: Optional[np.ndarray] = None,
                f0_floor: Optional[float] = None,
                f0_ceil: Optional[float] = None) -> SourceSequence:
        """Waveform -> pitch-synchronous residual cycles.

        ``f0`` may carry a per-frame track in Hz (``0``/``NaN`` = unvoiced); when
        it is missing, F0 is estimated.  Either way the track is sanitised
        (non-finite and negative entries become unvoiced) before it drives the
        epoch tracker.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return self._empty_sequence(x, fs, frame_period, f0)
        if not np.isfinite(x).all():
            raise ValueError("audio must be finite")

        fs = int(fs or self.default_fs)
        frame_period = float(frame_period or self.default_frame_period)
        hop = self.hop(fs, frame_period)
        floor = float(self.f0_floor if f0_floor is None else f0_floor)
        ceil = float(self.f0_ceil if f0_ceil is None else f0_ceil)
        fft_size = int(self.fft_size or default_fft_size(fs, frame_period))

        if f0 is None:
            track = self._estimate_f0(x, fs, frame_period, floor, ceil, fft_size)
            track = smooth_f0(track, self.f0_smoothing)
        else:
            track = self._sanitise_f0(f0)
        voiced = track > 0.0

        residual = whiten(x, fs, frame_period, fft_size=fft_size,
                          n_mcep=self.n_mcep, f_min=self.f_min)
        epochs, runs = pick_epochs(track, hop, x.size, fs, floor, ceil,
                                   refine=self.refine_epochs, residual=residual,
                                   refine_ratio=self.refine_ratio)
        cycles = extract_cycles(residual, epochs, self.cycle_length,
                                min_period=self._min_period(fs, ceil),
                                method=self.resample, runs=runs)

        noise = np.where(voiced, 0.0, 1.0)
        noise = assign_frame_noise(cycles.noise, cycles.epochs, cycles.periods,
                                   noise, hop, mask=voiced)
        return SourceSequence(
            f0=track, voiced=voiced, noise_level=noise, excitation=cycles.vectors,
            gains=cycles.gains, epochs=cycles.epochs, periods=cycles.periods,
            cycle_length=self.cycle_length, fs=fs, frame_period=frame_period,
            n_samples=int(x.size), backend=self.name)

    # -- helpers -----------------------------------------------------------

    def _empty_sequence(self, x: np.ndarray, fs: Optional[int],
                        frame_period: Optional[float],
                        f0: Optional[np.ndarray]) -> SourceSequence:
        """The zero-length/empty-audio case: an empty but well-formed sequence."""
        fs = int(fs or self.default_fs)
        frame_period = float(frame_period or self.default_frame_period)
        track = (np.zeros(0) if f0 is None
                 else self._sanitise_f0(f0))
        voiced = track > 0.0
        return SourceSequence(
            f0=track, voiced=voiced,
            noise_level=np.where(voiced, 0.0, 1.0),
            excitation=np.zeros((0, self.cycle_length)), gains=np.zeros(0),
            epochs=np.zeros(0, dtype=np.int64), periods=np.zeros(0, dtype=np.int64),
            cycle_length=self.cycle_length, fs=fs, frame_period=frame_period,
            n_samples=int(x.size), backend=self.name)

    @staticmethod
    def _sanitise_f0(f0: np.ndarray) -> np.ndarray:
        """Any F0 input -> finite, non-negative Hz (0 = unvoiced)."""
        track = np.asarray(f0, dtype=np.float64).reshape(-1)
        return np.where(np.isfinite(track) & (track > 0.0), track, 0.0)

    def _estimate_f0(self, x: np.ndarray, fs: int, frame_period: float,
                     f0_floor: float, f0_ceil: float, fft_size: int) -> np.ndarray:
        """Autocorrelation F0 with the fallback vocoder's flatness gate."""
        hop = self.hop(fs, frame_period)
        f0 = estimate_f0(x, fs, hop, f0_floor, f0_ceil,
                         window_periods=self.f0_window_periods)
        frames = frame_signal(x, fft_size, hop) * np.hanning(fft_size)
        power = np.abs(np.fft.rfft(frames, fft_size, axis=1)) ** 2
        flat = spectral_flatness(power)
        # The frame grid is the one the rest of HMS uses: the plain window's
        # frame count, capped by the spectral frames (same `min` as the fallback
        # vocoder).  A wider estimator window must not move the grid, so the
        # track is cut/padded to that length here.
        _, f0_frames = _padded_frames(x, 4 * hop, hop)
        n_frames = min(f0_frames, len(flat))
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        return np.where(flat[:n_frames] < FLATNESS_THRESHOLD, f0[:n_frames], 0.0)

    def _min_period(self, fs: int, f0_ceil: float) -> int:
        """Shortest cycle the extractor keeps (the same floor `pick_epochs` uses)."""
        return int(max(MIN_PERIOD_SAMPLES, int(np.floor(fs / max(f0_ceil, 1.0)))))


def _padded_frames(x: np.ndarray, frame_length: int, hop: int):
    """The padded signal and frame count of :func:`hms.core.dsp.frame_signal`.

    ``frame_signal`` zero-pads an input shorter than one window up to a window,
    then pads half a window on both sides, so frame ``t`` covers original samples
    ``[t * hop - frame_length // 2, ...)``.  Reproducing that padding here lets
    the frames be built block by block (a wide analysis window on a long file
    would otherwise be a very large 2-D array) without moving the frame grid.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    frame_length = int(frame_length)
    if x.size < frame_length:
        x = np.pad(x, (0, frame_length - x.size))
    pad = frame_length // 2
    padded = np.pad(x, (pad, pad + frame_length))
    n_frames = 1 + (len(padded) - frame_length) // max(1, int(hop))
    return padded, int(n_frames)


def _cmnd(frame: np.ndarray, window: np.ndarray, n_fft: int,
          lag_max: int) -> np.ndarray:
    """CMND of a block of frames, lags ``1..lag_max`` (column ``j`` = lag ``j + 1``)."""
    spectrum = np.fft.rfft(frame * window, n_fft, axis=1)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, n_fft, axis=1)[:, :len(window)]
    acf = acf / np.maximum(acf[:, :1], 1e-12)
    lags = np.arange(1, acf.shape[1] + 1, dtype=np.float64)
    mean_abs = np.cumsum(np.abs(acf), axis=1) / lags
    # lag L is normalised by the mean up to and including lag L, exactly as in
    # `hms.core.dsp.autocorrelation_f0` (the cumulative-mean normalisation of
    # the CMND); column j of the result is lag j + 1.
    return (acf[:, 1:lag_max + 1] / np.maximum(mean_abs[:, 1:lag_max + 1], 1e-12))


def estimate_f0(x: np.ndarray, fs: int, hop: int, f0_floor: float = 50.0,
                f0_ceil: float = 2000.0, threshold: float = 0.30,
                window_periods: float = 3.0, block_frames: int = 256) -> np.ndarray:
    """F0 track in Hz (``0.0`` = unvoiced) from a CMND over a wide window.

    This is :func:`hms.core.dsp.autocorrelation_f0` with one deliberate
    difference: the analysis window is at least ``window_periods`` periods of
    ``f0_floor`` long (and at least four frame hops), instead of four hops.
    Autocorrelation needs several periods of the candidate lag to see them, so
    the plain short window mistakes a strong formant or a harmonic for the
    fundamental on exactly the frames where the true period is long -- the
    classic low-pitch / octave failure.  The source backend wants a track it can
    cut cycles on, so it pays for the wider window; the shared estimator keeps
    its own behaviour for the vocoder.

    Frames are processed in blocks of ``block_frames`` so the memory cost does
    not grow with the window; the framing, the lag range and the voicing
    threshold follow :func:`hms.core.dsp.autocorrelation_f0` exactly.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fs = int(fs)
    hop = max(1, int(hop))
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    floor = float(max(f0_floor, 1.0))
    ceil = float(max(f0_ceil, floor + 1.0))
    wanted = int(np.ceil(window_periods * fs / floor)) if window_periods > 0 else 0
    frame_length = int(max(4 * hop, wanted, 4))
    lag_min = max(1, int(fs / ceil))
    lag_max = min(frame_length - 2, int(fs / floor))

    padded, n_frames = _padded_frames(x, frame_length, hop)
    if lag_max <= lag_min:                        # window shorter than one period
        return np.zeros(n_frames, dtype=np.float64)

    n_fft = 1 << int(np.ceil(np.log2(2 * frame_length)))
    window = np.hanning(frame_length)
    offsets = np.arange(frame_length)
    track = np.zeros(n_frames, dtype=np.float64)
    for start in range(0, n_frames, int(block_frames)):
        stop = min(n_frames, start + int(block_frames))
        index = np.arange(start, stop)[:, None] * hop + offsets[None, :]
        cmnd = _cmnd(padded[index], window, n_fft, lag_max)
        segment = cmnd[:, lag_min - 1:]
        if segment.shape[1] == 0:
            continue
        choice = np.argmax(segment, axis=1)
        peak = segment[np.arange(len(segment)), choice]
        best = choice + lag_min
        track[start:stop] = np.where(peak > threshold, fs / np.maximum(best, 1), 0.0)
    return track


def smooth_f0(f0: np.ndarray, window: int = 5) -> np.ndarray:
    """Median-filter an F0 track, leaving unvoiced frames unvoiced.

    Only the voiced entries take part in the median (unvoiced frames are NaN in
    the window), so a voiced/unvoiced boundary is not smeared, and the result is
    never voicing-positive where the input was not.  An even ``window`` is
    rounded up to the next odd number.
    """
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    window = int(window)
    if window <= 1 or f0.size == 0:
        return f0.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    values = np.where(np.isfinite(f0) & (f0 > 0.0), f0, np.nan)
    if not np.isfinite(values).any():
        return np.where(np.isfinite(values), values, 0.0)
    if values.size < window:
        return np.where(np.isfinite(values), values, 0.0)
    padded = np.pad(values, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        # An all-unvoiced window is a normal situation here, not a problem to
        # report; those frames keep their original (unvoiced) value.
        warnings.simplefilter("ignore", RuntimeWarning)
        filtered = np.nanmedian(windows, axis=1)
    voiced = np.isfinite(f0) & (f0 > 0.0)
    out = np.where(np.isfinite(filtered), filtered, f0)
    return np.where(voiced, out, 0.0)


__all__ = ["VoiceSourceModel", "estimate_f0", "smooth_f0",
           "FLATNESS_THRESHOLD"]
