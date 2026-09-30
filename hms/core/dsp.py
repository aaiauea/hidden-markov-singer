"""Small, dependency-free DSP helpers.

These are deliberately *approximate* estimators, used for two jobs:

1. `BuiltinVocoder` analysis (the fallback when WORLD cannot be built).
2. A sanity fallback when WORLD's D4C aperiodicity estimator degenerates
   (it saturates to 1.0 -- "pure noise" -- on pathologically clean signals
   such as synthetic test tones without a noise floor; see
   `NativeWorldVocoder.analyze`).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from hms.core.features import band_edges


def frame_signal(x: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """Split into overlapping frames with edge padding -> (T, frame_length)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < frame_length:
        x = np.pad(x, (0, frame_length - n))
    pad_left = frame_length // 2
    padded = np.pad(x, (pad_left, pad_left + frame_length))
    n_frames = 1 + (len(padded) - frame_length) // hop
    idx = np.arange(frame_length)[None, :] + hop * np.arange(n_frames)[:, None]
    return padded[idx]


def autocorrelation_f0(x: np.ndarray, fs: int, frame_length: int, hop: int,
                       f0_floor: float = 71.0, f0_ceil: float = 800.0,
                       threshold: float = 0.30) -> np.ndarray:
    """Normalized-autocorrelation F0 with enough cycles for its low end.

    Keep at least 2.5 periods of the requested floor in each analysis frame.
    This helps distinguish a low fundamental from its strong second harmonic;
    the hop and output frame count remain those of the caller's original grid.
    """
    frames = frame_signal(x, frame_length, hop)
    analysis_length = frame_length
    if f0_floor > 0.0:
        minimum_length = int(np.ceil(2.5 * fs / f0_floor))
        if minimum_length > analysis_length:
            analysis_length = minimum_length
            if analysis_length % 2 == 0:
                analysis_length += 1
            frame_count = len(frames)
            del frames
            frames = frame_signal(x, analysis_length, hop)[:frame_count]
    frames = frames * np.hanning(analysis_length)
    n_fft = 1 << int(np.ceil(np.log2(2 * analysis_length)))
    spec = np.fft.rfft(frames, n_fft, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, n_fft, axis=1)[:, :analysis_length]
    acf = acf / np.maximum(acf[:, :1], 1e-12)

    lags = np.arange(1, analysis_length + 1)
    mean_abs = np.cumsum(np.abs(acf), axis=1) / lags
    cmnd = acf / np.maximum(mean_abs, 1e-12)

    lag_min = max(1, int(fs / max(f0_ceil, 1.0)))
    lag_max = min(analysis_length - 2, int(fs / max(f0_floor, 1.0)))
    segment = cmnd[:, lag_min:lag_max + 1]
    best = np.argmax(segment, axis=1) + lag_min
    peak = cmnd[np.arange(len(cmnd)), best]
    return np.where(peak > threshold, fs / np.maximum(best, 1), 0.0)


def harmonicity_aperiodicity(power: np.ndarray, f0: np.ndarray, fs: int,
                             n_band: int = 5
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-band aperiodicity from pitch-synchronous spectral similarity.

    For a frame with period P, the magnitude spectrum of a periodic signal is
    (nearly) unchanged when shifted by 1/P; for noisy signals it is not.  The
    normalised correlation between |X(f)| and |X(f + 1/P)| inside a frequency
    band is therefore a harmonic-to-noise measure, and 1 - correlation is a
    usable aperiodicity value.

    Returns
    -------
    (band_ap, ap)
        ``band_ap`` has shape (T, n_band); ``ap`` is the same estimate
        interpolated onto the spectrum bins, shape (T, bins).
    """
    power = np.atleast_2d(np.asarray(power, dtype=np.float64))
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    mag = np.sqrt(np.maximum(power, 0.0))
    n_frames, bins = mag.shape
    freqs = np.linspace(0.0, fs / 2.0, bins)
    edges = band_edges(n_band, fs)
    centers = np.maximum.accumulate(0.5 * (edges[:-1] + edges[1:]))
    bin_hz = fs / (2.0 * (bins - 1))

    band_ap = np.full((n_frames, n_band), 0.9)
    for t in range(n_frames):
        if not np.isfinite(f0[t]) or f0[t] <= 0:
            continue                                    # unvoiced -> 0.9
        shift = int(round((fs / f0[t]) / bin_hz))
        if shift < 1 or shift >= bins // 2:
            band_ap[t] = 0.5
            continue
        v = mag[t, :bins - shift]
        w = mag[t, shift:]
        v = v - v.mean()
        w = w - w.mean()
        for b in range(n_band):
            sel = ((freqs[:bins - shift] >= edges[b])
                   & (freqs[:bins - shift] < edges[b + 1]))
            if sel.sum() < 4:
                continue
            num = float((v[sel] * w[sel]).sum())
            den = float(np.sqrt((v[sel] ** 2).sum() * (w[sel] ** 2).sum()))
            band_ap[t, b] = np.clip(1.0 - max(num / (den + 1e-12), 0.0), 0.02, 1.0)

    ap = np.empty((n_frames, bins), dtype=np.float64)
    for t in range(n_frames):
        ap[t] = np.interp(freqs, centers, band_ap[t])
    return band_ap, ap


def spectral_flatness(power: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Geometric mean over arithmetic mean, per frame -> (T,).

    ~1 for white noise (energy spread over every bin), ~0 for a strongly
    harmonic or resonant frame.  Used as a second opinion on voicing: it is
    what stops the pure-numpy fallback from calling noise "voiced" just because
    the normalised autocorrelation happens to peak somewhere.
    """
    power = np.atleast_2d(np.asarray(power, dtype=np.float64))
    safe = np.maximum(power, eps)
    return np.exp(np.mean(np.log(safe), axis=1)) / np.mean(safe, axis=1)


def is_degenerate_aperiodicity(ap: np.ndarray, voiced: np.ndarray,
                               threshold: float = 0.99,
                               fraction: float = 0.9) -> bool:
    """Detect WORLD's D4C failure mode: ~1.0 aperiodicity on voiced frames."""
    ap = np.atleast_2d(ap)
    voiced = np.asarray(voiced, dtype=bool).reshape(-1)
    if voiced.sum() == 0:
        return False
    mean_ap = ap[voiced].mean(axis=1)
    return bool((mean_ap > threshold).mean() > fraction)
