"""Acoustic feature extraction and the feature <-> WORLD parameter mapping.

The acoustic model never sees the raw WORLD parameter grid.  It sees a small,
fixed-dimension vector per frame (see :class:`FeatureSpec`), because a compact
feature space is what makes HMM/GMM training viable with a few minutes of audio.

    WORLD frame -> FeatureSpec.encode() -> static feature vector
                                           |
                       +-------------------+-------------------+
                       v                   v                   v
                    static              delta            delta-delta
                                           |
                                           v
                                 FeatureSpec.decode() -> WORLD parameters

WORLD parameters
----------------
f0   : (T,)      Hz, 0.0 marks unvoiced
sp   : (T, F+1)  linear power spectral envelope, F = fft_size/2
ap   : (T, F+1)  aperiodicity in [0, 1]

Feature layout (default spec, dim = static_dim = 1 + n_mcep + n_band)
--------------------------------------------------------------------
 [0]                                log F0 in semitones re. `f0_ref_hz`
 [1 : 1+n_mcep]                     mel-cepstrum of the log spectral envelope,
                                    c0 ... c_{n_mcep-1}
 [1+n_mcep : 1+n_mcep+n_band]       band aperiodicity (mean aperiodicity per band)

Design notes
------------
*Log-F0 is note-relative at training time.*  The trainer subtracts the note
actually sung on each frame, so the acoustic model learns *how this singer moves
around the note* (onset scoops, drift, vibrato) instead of memorising absolute
pitch.  At synthesis the target note supplies the base and the model adds the
learned deviation.  See `hms.core.pitch`.

*Mel-cepstrum instead of raw log-spectrum.*  The HMM needs per-dimension
random variables it can put a Gaussian on.  Raw log-spectrum bins are 513
strongly correlated numbers per frame; a mel-cepstrum is ~30 decorrelated,
smooth numbers that map back onto the WORLD grid through a documented
filterbank + DCT pair (`power_to_mcep` / `mcep_to_power`, written with plain
numpy so there is no SPTK or librosa dependency).  Because the pair is built
from one orthonormal basis, encode/decode round-trips to numerical precision,
which is what the tests check.

*No hard-coded formants.*  Nothing in this file knows anything about vowels;
the spectral model is entirely data-driven mel-cepstrum statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: Floor applied before taking logs of power values.
EPS = 1e-10

#: Mel-cepstrum size (c0 .. c_{n-1}, inclusive) used by default.
DEFAULT_N_MCEP = 30

#: Number of aperiodicity bands.
DEFAULT_N_BAND = 5


# --------------------------------------------------------------------------
# Basic transforms
# --------------------------------------------------------------------------


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    """HTK-style mel scale (1 kHz -> 1000 mel, logarithmic above)."""
    f = np.asarray(f, dtype=np.float64)
    return 1127.01048 * np.log1p(f / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return 700.0 * (np.expm1(m / 1127.01048))


def hz_to_semitone(f: np.ndarray | float, f_ref: float) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    out = np.full(f.shape, -np.inf)
    np.log2(np.maximum(f, 1e-12) / f_ref, out=out)
    return 12.0 * out


def semitone_to_hz(s: np.ndarray | float, f_ref: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return f_ref * np.power(2.0, s / 12.0)


def mel_filterbank(n_filters: int, fft_size: int, fs: int,
                   f_min: float = 0.0) -> np.ndarray:
    """Triangular mel filterbank, shape (n_filters, fft_size // 2 + 1)."""
    n_bins = fft_size // 2 + 1
    freqs = np.linspace(0.0, fs / 2.0, n_bins)
    mel_lo = float(hz_to_mel(f_min))
    mel_hi = float(hz_to_mel(fs / 2.0))
    hz_edges = mel_to_hz(np.linspace(mel_lo, mel_hi, n_filters + 2))

    bank = np.zeros((n_filters, n_bins), dtype=np.float64)
    for i in range(n_filters):
        left, center, right = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        up = (freqs - left) / max(center - left, 1e-9)
        down = (right - freqs) / max(right - center, 1e-9)
        bank[i] = np.clip(np.minimum(up, down), 0.0, None)
    return bank


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, shape (n, n), rows indexed by frequency k."""
    k = np.arange(n)[:, None]
    basis = np.cos(np.pi * k * (np.arange(n)[None, :] + 0.5) / n)
    scale = np.full((n, 1), np.sqrt(2.0 / n))
    scale[0] = np.sqrt(1.0 / n)
    return basis * scale


def dct2(x: np.ndarray) -> np.ndarray:
    """Orthonormal DCT-II along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    basis = _dct_matrix(x.shape[-1])
    return x @ basis.T


def idct2(c: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`dct2` (same basis, transposed)."""
    c = np.asarray(c, dtype=np.float64)
    basis = _dct_matrix(c.shape[-1])
    return c @ basis


# --------------------------------------------------------------------------
# Power spectrum <-> mel cepstrum
# --------------------------------------------------------------------------


def _bank_weight_sums(bank: np.ndarray) -> np.ndarray:
    """Per-filter weight sums, used to turn band sums into power *density*.

    FFT bins are uniformly spaced in frequency, so dividing a filter output by
    the filter's weight sum yields a mean power density comparable to the
    spectrogram values themselves.  Without this step the reconstructed
    envelope is scaled by the (frequency dependent) band width.
    """
    sums = bank.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return sums


def power_to_logmel(sp: np.ndarray, n_bands: int, fft_size: int, fs: int,
                    f_min: float = 0.0) -> np.ndarray:
    """Linear power spectrum (T, bins) -> log mel band densities (T, n_bands)."""
    n_bins = fft_size // 2 + 1
    sp = np.atleast_2d(np.asarray(sp, dtype=np.float64))
    if sp.shape[1] < n_bins:
        raise ValueError(f"expected >= {n_bins} spectral bins, got {sp.shape[1]}")
    bank = mel_filterbank(n_bands, fft_size, fs, f_min=f_min)
    mel_power = (sp[:, :n_bins] @ bank.T) / _bank_weight_sums(bank).T
    return np.log(np.maximum(mel_power, EPS))


def logmel_to_power(logmel: np.ndarray, n_bands: int, fft_size: int, fs: int,
                    out_bins: int | None = None, f_min: float = 0.0
                    ) -> np.ndarray:
    """Log mel band densities -> linear power spectrum on the WORLD grid.

    Interpolates the (smooth, low-order) envelope back onto the linear
    frequency axis, using the mel-warped axis so the low-frequency spacing
    stays as dense as the filterbank.  Interpolation weights are normalised per
    bin, so the output is again a power density and encode/decode round-trips.
    """
    if out_bins is None:
        out_bins = fft_size // 2 + 1
    logmel = np.atleast_2d(np.asarray(logmel, dtype=np.float64))
    if logmel.shape[1] != n_bands:
        raise ValueError(f"expected {n_bands} bands, got {logmel.shape[1]}")

    mel_lo = float(hz_to_mel(f_min))
    mel_hi = float(hz_to_mel(fs / 2.0))
    band_mels = np.linspace(mel_lo, mel_hi, n_bands + 2)[1:-1]

    freqs = np.linspace(0.0, fs / 2.0, out_bins)
    bin_mels = np.clip(hz_to_mel(freqs), band_mels[0], band_mels[-1])
    # Weight matrix (out_bins, n_bands) for linear interpolation on the mel axis.
    pos = np.searchsorted(band_mels, bin_mels, side="right") - 1
    pos = np.clip(pos, 0, n_bands - 2)
    frac = (bin_mels - band_mels[pos]) / (band_mels[pos + 1] - band_mels[pos])
    weight = np.zeros((out_bins, n_bands), dtype=np.float64)
    rows = np.arange(out_bins)
    weight[rows, pos] = 1.0 - frac
    weight[rows, pos + 1] += frac

    return np.exp(logmel @ weight.T)


def power_to_mcep(sp: np.ndarray, fft_size: int, fs: int, n_mcep: int,
                  f_min: float = 0.0) -> np.ndarray:
    """Linear power spectrum -> mel-cepstrum (T, n_mcep), c0..c_{n-1}."""
    n_bands = n_mcep + 2
    logmel = power_to_logmel(sp, n_bands, fft_size, fs, f_min=f_min)
    return dct2(logmel)[:, :n_mcep]


def mcep_to_power(mcep: np.ndarray, fft_size: int, fs: int, n_mcep: int,
                  out_bins: int | None = None, f_min: float = 0.0) -> np.ndarray:
    """Mel-cepstrum -> linear power spectrum on the WORLD grid.

    Exact inverse of :func:`power_to_mcep` up to the truncation of the last two
    cepstral coefficients (which is the intended compactness).
    """
    mcep = np.atleast_2d(np.asarray(mcep, dtype=np.float64))
    if mcep.shape[1] != n_mcep:
        raise ValueError(f"expected {n_mcep} cepstral coefficients, "
                         f"got {mcep.shape[1]}")
    n_bands = n_mcep + 2
    full = np.zeros((mcep.shape[0], n_bands), dtype=np.float64)
    full[:, :n_mcep] = mcep
    logmel = idct2(full)
    return logmel_to_power(logmel, n_bands, fft_size, fs, out_bins=out_bins,
                           f_min=f_min)


# --------------------------------------------------------------------------
# Aperiodicity bands
# --------------------------------------------------------------------------


def band_edges(n_band: int, fs: int) -> np.ndarray:
    """Mel-spaced aperiodicity band edges in Hz, length n_band + 1."""
    return mel_to_hz(np.linspace(float(hz_to_mel(0.0)), float(hz_to_mel(fs / 2.0)),
                                 n_band + 1))


def aperiodicity_to_bands(ap: np.ndarray, n_band: int, fs: int) -> np.ndarray:
    """(T, bins) aperiodicity -> (T, n_band) mean aperiodicity per band."""
    ap = np.atleast_2d(np.asarray(ap, dtype=np.float64))
    edges = band_edges(n_band, fs)
    freqs = np.linspace(0.0, fs / 2.0, ap.shape[1])
    out = np.zeros((ap.shape[0], n_band), dtype=np.float64)
    for i in range(n_band):
        upper = freqs <= edges[i + 1] + 1e-9 if i == n_band - 1 else freqs < edges[i + 1]
        sel = (freqs >= edges[i]) & upper
        if not sel.any():
            sel = np.zeros_like(freqs, dtype=bool)
            sel[np.argmin(np.abs(freqs - 0.5 * (edges[i] + edges[i + 1])))] = True
        out[:, i] = ap[:, sel].mean(axis=1)
    return out


def bands_to_aperiodicity(bands: np.ndarray, n_band: int, fs: int,
                          out_bins: int) -> np.ndarray:
    """(T, n_band) -> (T, out_bins), interpolated across the mel axis."""
    bands = np.atleast_2d(np.asarray(bands, dtype=np.float64))
    edges = band_edges(n_band, fs)
    centers = np.maximum.accumulate(0.5 * (edges[:-1] + edges[1:]))
    freqs = np.linspace(0.0, fs / 2.0, out_bins)
    band_mels = hz_to_mel(centers)
    bin_mels = np.clip(hz_to_mel(freqs), band_mels[0], band_mels[-1])

    pos = np.searchsorted(band_mels, bin_mels, side="right") - 1
    pos = np.clip(pos, 0, n_band - 2)
    frac = (bin_mels - band_mels[pos]) / (band_mels[pos + 1] - band_mels[pos])
    out = bands[:, pos] * (1.0 - frac)[None, :] + bands[:, pos + 1] * frac[None, :]
    return np.clip(out, 1e-4, 1.0)


# --------------------------------------------------------------------------
# Feature specification
# --------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Geometry and layout of the acoustic feature vector.

    Attributes
    ----------
    fs, frame_period, fft_size
        Analysis/synthesis geometry; `fft_size` must match the vocoder backend
        (see `hms.vocoder.world.WorldVocoder.fft_size`).
    n_mcep
        Mel-cepstral coefficients stored, *including* c0 (overall energy).
    n_band
        Number of aperiodicity bands.
    use_delta / use_delta2
        Append first / second order difference streams.  Deltas are what let a
        left-to-right HMM model onsets and releases, and what the MLPG
        trajectory generator needs to join phonemes smoothly.
    f0_ref_hz
        Reference frequency for the semitone scale (C4 by default).  Only a
        unit choice -- absolute pitch comes from the score at synthesis time.
    """

    fs: int = 44100
    frame_period: float = 5.0
    fft_size: int = 2048
    n_mcep: int = DEFAULT_N_MCEP
    n_band: int = DEFAULT_N_BAND
    use_delta: bool = True
    use_delta2: bool = False
    f0_ref_hz: float = 261.6255653005986   # C4 = MIDI 60
    f0_floor: float = 71.0
    f0_ceil: float = 800.0
    pitch_bins_per_semitone: int = 8
    voiced_threshold: float = 5.0
    mcep_f_min: float = 0.0
    f0_estimation: str = "dio"             # "dio" (fast) or "harvest" (robust)
    refine_f0: bool = True
    delta_window: int = 2

    # -- geometry ----------------------------------------------------------

    @property
    def n_bins(self) -> int:
        return self.fft_size // 2 + 1

    @property
    def static_dim(self) -> int:
        return 1 + self.n_mcep + self.n_band

    @property
    def dim(self) -> int:
        d = self.static_dim
        if self.use_delta:
            d += self.static_dim
        if self.use_delta2:
            d += self.static_dim
        return d

    @property
    def stream_sizes(self) -> Tuple[int, ...]:
        """GMM sub-vector sizes, one per dynamic stream."""
        streams: List[int] = [self.static_dim]
        if self.use_delta:
            streams.append(self.static_dim)
        if self.use_delta2:
            streams.append(self.static_dim)
        return tuple(streams)

    def __post_init__(self) -> None:
        if self.n_mcep < 2:
            raise ValueError("n_mcep must be >= 2 (c0 plus at least one shape coeff)")

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        fields = ("fs", "frame_period", "fft_size", "n_mcep", "n_band",
                  "use_delta", "use_delta2", "f0_ref_hz", "f0_floor", "f0_ceil",
                  "pitch_bins_per_semitone", "voiced_threshold", "mcep_f_min",
                  "f0_estimation", "refine_f0", "delta_window")
        return {f: getattr(self, f) for f in fields}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSpec":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path) -> "FeatureSpec":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- encoding ----------------------------------------------------------

    def encode(self, f0: np.ndarray, sp: np.ndarray, ap: np.ndarray) -> np.ndarray:
        """WORLD parameters -> static feature matrix (T, static_dim)."""
        f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
        sp = np.atleast_2d(np.asarray(sp, dtype=np.float64))
        ap = np.atleast_2d(np.asarray(ap, dtype=np.float64))
        if len(f0) == 0:
            return np.zeros((0, self.static_dim), dtype=np.float64)

        voiced = f0 >= self.voiced_threshold
        logf0 = np.where(voiced, hz_to_semitone(np.maximum(f0, 1e-6),
                                                self.f0_ref_hz), 0.0)
        mcep = power_to_mcep(sp, self.fft_size, self.fs, self.n_mcep,
                             f_min=self.mcep_f_min)
        bands = aperiodicity_to_bands(ap, self.n_band, self.fs)
        return np.concatenate([logf0[:, None], mcep, bands], axis=1)

    def decode(self, static: np.ndarray, f0_semitones: np.ndarray | None = None
               ) -> "AcousticFrameSequence":
        """Static feature matrix -> WORLD parameters.

        Parameters
        ----------
        static : (T, static_dim)
        f0_semitones : (T,), optional
            Absolute log-F0 (semitones re. `f0_ref_hz`) to use instead of
            feature dimension 0.  This is how the pitch model injects the
            musical note; NaN (or values below `f0_floor`) become unvoiced
            frames.

            Note that feature dimension 0 stores 0.0 -- not NaN -- for unvoiced
            frames when ``encode`` wrote them, because a Gaussian cannot be
            fitted around NaN.  Voicing is therefore *not* recoverable from the
            feature vector alone; pass this argument (as the synthesizer does)
            when voicing matters.
        """
        static = np.atleast_2d(np.asarray(static, dtype=np.float64))
        f0_semi = (static[:, 0] if f0_semitones is None
                   else np.asarray(f0_semitones, dtype=np.float64))
        f0 = semitone_to_hz(f0_semi, self.f0_ref_hz)
        unvoiced = (~np.isfinite(f0)) | (f0 < self.f0_floor) | (f0 > self.f0_ceil)
        f0 = np.where(unvoiced, 0.0, f0)

        mcep = static[:, 1:1 + self.n_mcep]
        sp = mcep_to_power(mcep, self.fft_size, self.fs, self.n_mcep,
                           f_min=self.mcep_f_min)
        bands = static[:, 1 + self.n_mcep:1 + self.n_mcep + self.n_band]
        ap = bands_to_aperiodicity(bands, self.n_band, self.fs, self.n_bins)
        return AcousticFrameSequence(f0=f0, sp=sp, ap=ap,
                                     frame_period=self.frame_period, fs=self.fs,
                                     fft_size=self.fft_size)


# --------------------------------------------------------------------------
# Container types
# --------------------------------------------------------------------------


@dataclass
class AcousticFrameSequence:
    """An utterance / synthesis result in WORLD parameter space."""

    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    frame_period: float = 5.0
    fs: int = 44100
    fft_size: int = 2048

    def __len__(self) -> int:
        return int(np.asarray(self.f0).shape[0])

    @property
    def voiced(self) -> np.ndarray:
        return np.asarray(self.f0) > 0.0

    @property
    def duration(self) -> float:
        return float(len(self) * self.frame_period / 1000.0)


# --------------------------------------------------------------------------
# Note-relative pitch helpers
# --------------------------------------------------------------------------


def relative_pitch(f0: np.ndarray, note: np.ndarray, f_ref: float,
                   voiced: np.ndarray | None = None) -> np.ndarray:
    """absolute log-F0 -> note-relative log-F0 (semitones)."""
    f0 = np.asarray(f0, dtype=np.float64)
    note = np.asarray(note, dtype=np.float64)
    if voiced is None:
        voiced = f0 > 0.0
    logf0 = hz_to_semitone(np.maximum(f0, 1e-6), f_ref)
    return np.where(voiced, logf0 - note, 0.0)


def absolute_pitch(relative: np.ndarray, note: np.ndarray, f_ref: float,
                   voiced: np.ndarray) -> np.ndarray:
    """note-relative log-F0 -> absolute log-F0 (semitones)."""
    rel = np.asarray(relative, dtype=np.float64) + np.asarray(note, dtype=np.float64)
    return np.where(voiced, rel, np.nan)


# --------------------------------------------------------------------------
# Delta features
# --------------------------------------------------------------------------


def delta_coeffs(window: int = 2) -> np.ndarray:
    """Normalised regression coefficients for a (2*window+1) tap delta."""
    denom = 2.0 * sum(k * k for k in range(1, window + 1))
    return np.array([k / denom for k in range(-window, window + 1)])


def add_dynamic_features(static: np.ndarray, use_delta: bool = True,
                         use_delta2: bool = False,
                         window: int = 2) -> np.ndarray:
    """Stack static (+ delta, + delta-delta) streams along the feature axis.

    Edge replication is used at utterance boundaries so the output has exactly
    the same number of frames as the input.
    """
    static = np.atleast_2d(np.asarray(static, dtype=np.float64))
    n, dim = static.shape
    if n == 0:
        return np.zeros((0, dim * (1 + use_delta + use_delta2)))

    def pad(x: np.ndarray) -> np.ndarray:
        return np.pad(x, ((window, window), (0, 0)), mode="edge")

    coeffs = delta_coeffs(window)
    padded = pad(static)
    streams = [static]
    if use_delta or use_delta2:
        # padded[t + k] == static[t + k - window] (edge replicated), so the
        # delta of frame t is simply sum_k coeffs[k] * padded[t + k].
        delta = sum(coeffs[k] * padded[k: k + n]
                    for k in range(2 * window + 1))
        if use_delta:
            streams.append(delta)
        if use_delta2:
            padded_d = pad(delta)
            delta2 = sum(coeffs[k] * padded_d[k: k + n]
                         for k in range(2 * window + 1))
            streams.append(delta2)
    return np.concatenate(streams, axis=1)


def remove_dynamic_features(features: np.ndarray, static_dim: int
                            ) -> np.ndarray:
    """Keep only the static block of a stacked feature matrix."""
    return np.atleast_2d(features)[:, :static_dim]


def split_streams(features: np.ndarray, stream_sizes: Sequence[int]
                  ) -> List[np.ndarray]:
    """Split a stacked feature matrix into GMM streams."""
    out, pos = [], 0
    features = np.atleast_2d(features)
    for size in stream_sizes:
        out.append(features[:, pos:pos + size])
        pos += size
    if pos != features.shape[1]:
        raise ValueError(f"stream_sizes {tuple(stream_sizes)} do not sum to "
                         f"{features.shape[1]}")
    return out
