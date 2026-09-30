"""Experimental source-audio observations for voice-to-voice (V2V) research.

This module is deliberately an *analysis-only* side path.  HMS's trained
acoustic HMMs are not acoustic-to-acoustic converters: training uses labelled
phoneme spans, and ``Synthesizer.synthesize`` needs a score containing phone
identity, context and duration.  This frontend does not infer those labels and
is not wired into the normal synthesis path.

For a source waveform it returns the configured HMS vocoder's F0 track (0 Hz
means unvoiced; real WORLD is used when available/configured), frame RMS, stable
LPC coefficients, LPC-derived cepstra and a normalised LPC log-spectral
envelope.  The backend's ``sp``/``ap`` arrays are used only as analysis return
values and are discarded; target HMS
synthesis must continue to use the target model's own acoustic statistics.

The LPC calculation uses a Hann-windowed, pre-emphasised frame and
biased-autocorrelation Levinson-Durbin recursion.  Reflection coefficients
are clipped strictly inside the unit circle, which keeps the resulting
all-pole filter stable.  Cepstra are the causal log-amplitude cepstrum of the
all-pole transfer function ``1 / A(z)`` (gain is intentionally separate); the
log spectrum is evaluated directly from that polynomial and mean-centred over
frequency.  These are source observations, not the HMS FeatureSpec layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from hms.vocoder import BACKENDS, Vocoder, get_vocoder


@dataclass(frozen=True)
class V2VAnalysisConfig:
    """Geometry and numerical controls for the experimental frontend.

    ``frame_period_ms`` defines the shared F0/LPC output grid.  The WORLD
    interface in HMS returns F0 but not its time vector; its conventional grid
    is represented as ``arange(T) * frame_period_ms`` (frame zero at 0 s).
    LPC frames are centred on the nearest input sample to each grid time.

    ``frame_length_ms`` is the LPC analysis window, independently of the hop.
    It is rounded up to an odd sample count and raised to at least
    ``lpc_order + 2`` samples when necessary, so each window has an integer
    sample at its centre.  The power spectrum from LPC is represented by
    ``spectrum_bins`` uniformly
    spaced bins from DC to Nyquist.  ``lpc_cepstra`` and ``lpc_log_spectrum``
    omit overall gain; ``energy_rms`` carries that separately.

    The default vocoder is HMS's existing ``auto`` backend selection
    (pyworld -> native WORLD -> builtin fallback).  For a reproducible run on
    machines without optional WORLD binaries, select ``vocoder='builtin'``.
    """

    frame_period_ms: float = 5.0
    frame_length_ms: float = 25.0
    lpc_order: int = 16
    cepstral_order: int = 20
    spectrum_bins: int = 129
    preemphasis: float = 0.97
    silence_rms: float = 1e-8
    reflection_limit: float = 0.98
    f0_floor: float = 71.0
    f0_ceil: float = 800.0
    f0_estimation: str = "dio"
    refine_f0: bool = True
    vocoder: str = "auto"
    batch_frames: int = 128

    def __post_init__(self) -> None:
        numeric = (self.frame_period_ms, self.frame_length_ms,
                   self.preemphasis, self.silence_rms,
                   self.reflection_limit, self.f0_floor, self.f0_ceil)
        try:
            if not all(np.isfinite(value) for value in numeric):
                raise ValueError("V2V analysis settings must be finite")
        except TypeError as exc:
            raise ValueError("V2V analysis settings must be numeric") from exc
        if self.frame_period_ms <= 0 or self.frame_length_ms <= 0:
            raise ValueError("frame period and LPC frame length must be positive")
        for name, value, minimum in (
                ("lpc_order", self.lpc_order, 1),
                ("cepstral_order", self.cepstral_order, 1),
                ("spectrum_bins", self.spectrum_bins, 2),
                ("batch_frames", self.batch_frames, 1)):
            if not isinstance(value, (int, np.integer)) \
                    or isinstance(value, (bool, np.bool_)) \
                    or int(value) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        # rfft uses 2 * (bins - 1) samples.  Keep every LPC polynomial
        # coefficient in that transform rather than silently truncating it.
        if 2 * (self.spectrum_bins - 1) < self.lpc_order + 1:
            raise ValueError("spectrum_bins is too small for the configured "
                             "LPC order")
        if not 0.0 <= self.preemphasis < 1.0:
            raise ValueError("preemphasis must be in [0, 1)")
        if self.silence_rms < 0:
            raise ValueError("silence_rms must be non-negative")
        if not 0.0 < self.reflection_limit < 1.0:
            raise ValueError("reflection_limit must be strictly between 0 and 1")
        if self.f0_floor <= 0 or self.f0_ceil <= self.f0_floor:
            raise ValueError("F0 bounds must satisfy 0 < f0_floor < f0_ceil")
        if self.f0_estimation not in ("dio", "harvest"):
            raise ValueError("f0_estimation must be 'dio' or 'harvest'")
        if self.vocoder not in BACKENDS:
            raise ValueError(f"vocoder must be one of {BACKENDS}")


@dataclass(frozen=True)
class V2VAnalysis:
    """Frame-aligned *source observations* from :class:`V2VFrontend`.

    No phone, phoneme-context or target-acoustic fields are present by design.
    ``f0_hz`` uses the HMS/WORLD convention of 0.0 for unvoiced frames.
    ``lpc_coefficients`` stores ``[1, a1, ..., a_order]`` for the polynomial
    ``A(z) = 1 + a1 z^-1 + ...``. ``lpc_cepstra`` contains c1..cN for
    ``log(1/A(z))``; c0/gain is omitted. ``lpc_log_spectrum`` is a
    mean-centred log-*power* envelope, with frequency bins given by
    ``spectrum_frequencies_hz``.

    The ``candidate_features`` property is only a compact, inspectable matrix
    for classical-model experiments.  Its layout is documented by
    ``candidate_feature_names`` and it is intentionally **not** compatible
    with ``FeatureSpec`` or directly accepted by ``Synthesizer``.
    """

    sample_rate: int
    audio_samples: int
    frame_period_ms: float
    frame_times_s: np.ndarray
    center_sample_indices: np.ndarray
    backend: str
    f0_hz: np.ndarray
    energy_rms: np.ndarray
    lpc_coefficients: np.ndarray
    reflection_coefficients: np.ndarray
    prediction_error_fraction: np.ndarray
    lpc_cepstra: np.ndarray
    lpc_log_spectrum: np.ndarray
    spectrum_frequencies_hz: np.ndarray
    lpc_order: int
    cepstral_order: int
    frame_length_samples: int

    def __len__(self) -> int:
        return int(self.f0_hz.shape[0])

    @property
    def voiced(self) -> np.ndarray:
        """WORLD voicing decision, preserved as a Boolean frame mask."""
        return self.f0_hz > 0.0

    @property
    def duration_seconds(self) -> float:
        """Duration of the input waveform, distinct from the output frame grid."""
        return self.audio_samples / float(self.sample_rate)

    @property
    def candidate_feature_names(self) -> Tuple[str, ...]:
        """Names of the exploratory low-dimensional candidate feature columns."""
        cepstra = tuple(f"lpc_logamp_cepstrum_c{i}"
                        for i in range(1, self.cepstral_order + 1))
        return cepstra + ("absolute_log_f0_semitones_re_A4", "voiced_flag",
                          "log_rms_db")

    @property
    def candidate_features(self) -> np.ndarray:
        """Compact LPC + F0 + energy matrix for *experiments only*.

        F0 is absolute log-pitch in semitones re. A4, set to 0 on unvoiced
        frames; the following voicing flag disambiguates that value. RMS is in
        dB relative to full scale (silence is floored at -240 dB). The cepstral
        and log-spectrum arrays use a last-voiced edge hold only on a terminal
        unvoiced suffix; F0, voicing, energy and LPC coefficients stay measured.
        Empty/all-unvoiced analyses are unchanged. Frame timing remains in
        ``frame_times_s`` rather than being smuggled in as a feature.
        """
        voiced = self.voiced
        log_f0 = np.zeros(len(self), dtype=np.float64)
        if voiced.any():
            log_f0[voiced] = 12.0 * np.log2(self.f0_hz[voiced] / 440.0)
        log_rms_db = 20.0 * np.log10(np.maximum(self.energy_rms, 1e-12))
        return np.column_stack((self.lpc_cepstra, log_f0,
                                voiced.astype(np.float64), log_rms_db))


class V2VFrontend:
    """Extract experimental LPC/WORLD observations without changing HMS synthesis."""

    def __init__(self, config: Optional[V2VAnalysisConfig] = None,
                 vocoder: Optional[Vocoder] = None) -> None:
        self.config = config or V2VAnalysisConfig()
        self._provided_vocoder = vocoder
        self._cached_vocoder: Optional[Vocoder] = None
        self._cached_fs: Optional[int] = None

    def _vocoder_for_rate(self, fs: int) -> Vocoder:
        if self._provided_vocoder is not None:
            return self._provided_vocoder
        if self._cached_vocoder is None or self._cached_fs != fs:
            self._cached_vocoder = get_vocoder(
                self.config.vocoder, fs=fs,
                frame_period=self.config.frame_period_ms)
            self._cached_fs = fs
        return self._cached_vocoder

    def analyze(self, audio: np.ndarray, fs: int) -> V2VAnalysis:
        """Analyse mono float audio into source observations on one frame grid.

        WORLD (or HMS's configured fallback backend) supplies F0 and therefore
        the authoritative frame count.  LPC/RMS frames are centred on that
        backend's nominal time grid, and are padded with zeros at file edges.
        The waveform is not resampled; ``fs`` is the actual input rate.
        """
        raw = np.asarray(audio)
        if raw.ndim != 1:
            raise ValueError("V2VFrontend expects mono 1-D audio; downmix first "
                             "with hms.data.wavio.read_wav")
        if np.iscomplexobj(raw):
            raise ValueError("audio samples must be real-valued")
        try:
            fs_value = float(fs)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample rate must be a positive integer") from exc
        if not np.isfinite(fs_value) or fs_value <= 0 \
                or fs_value != int(fs_value):
            raise ValueError("sample rate must be a positive integer")
        fs = int(fs_value)
        signal = np.ascontiguousarray(raw, dtype=np.float64)
        if not np.isfinite(signal).all():
            raise ValueError("audio samples must contain only finite values")

        config = self.config
        backend_name = "not-run"
        if signal.size:
            vocoder = self._vocoder_for_rate(fs)
            f0, world_sp, world_ap = vocoder.analyze(
                signal, fs=fs, frame_period=config.frame_period_ms,
                f0_floor=config.f0_floor, f0_ceil=config.f0_ceil,
                f0_estimation=config.f0_estimation,
                refine_f0=config.refine_f0)
            f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
            sp = np.asarray(world_sp)
            ap = np.asarray(world_ap)
            if sp.ndim < 1 or ap.ndim < 1 \
                    or sp.shape[0] != len(f0) or ap.shape[0] != len(f0):
                raise ValueError("vocoder F0, spectral envelope and "
                                 "aperiodicity frame counts must agree")
            # The source spectral envelope is deliberately not retained or routed
            # into the target HMS model: this experiment's acoustic inputs are
            # LPC shape, F0, RMS and time.
            del world_sp, world_ap, sp, ap
            f0 = np.where(np.isfinite(f0) & (f0 > 0.0), f0, 0.0)
            backend_name = str(getattr(vocoder, "name", config.vocoder))
        else:
            f0 = np.zeros(0, dtype=np.float64)

        frame_times_s = (np.arange(len(f0), dtype=np.float64)
                         * (config.frame_period_ms / 1000.0))
        center_samples = np.rint(frame_times_s * fs).astype(np.int64)
        frame_length = max(int(round(fs * config.frame_length_ms / 1000.0)),
                           config.lpc_order + 2)
        if frame_length % 2 == 0:
            frame_length += 1
        n_frames = len(f0)
        order = int(config.lpc_order)
        cepstral_order = int(config.cepstral_order)
        spectrum_bins = int(config.spectrum_bins)

        energy = np.zeros(n_frames, dtype=np.float64)
        coefficients = np.zeros((n_frames, order + 1), dtype=np.float64)
        coefficients[:, 0] = 1.0
        reflection = np.zeros((n_frames, order), dtype=np.float64)
        prediction_error = np.zeros(n_frames, dtype=np.float64)
        cepstra = np.zeros((n_frames, cepstral_order), dtype=np.float64)
        log_spectrum = np.zeros((n_frames, spectrum_bins), dtype=np.float64)

        if n_frames:
            emphasized = signal.copy()
            if config.preemphasis and len(emphasized) > 1:
                emphasized[1:] = (signal[1:]
                                  - config.preemphasis * signal[:-1])
            window = np.hanning(frame_length)
            autocorrelation_fft_size = 1 << (2 * frame_length - 1).bit_length()
            spectrum_fft_size = 2 * (spectrum_bins - 1)
            tiny = np.finfo(np.float64).tiny

            for first in range(0, n_frames, config.batch_frames):
                last = min(first + config.batch_frames, n_frames)
                centers = center_samples[first:last]
                raw_frames = _centered_frames(signal, centers, frame_length)
                energy[first:last] = _row_rms(raw_frames)
                frames = _centered_frames(emphasized, centers, frame_length)
                windowed = frames * window[None, :]
                scale = np.max(np.abs(windowed), axis=1)
                active = ((energy[first:last] >= config.silence_rms)
                          & np.isfinite(scale) & (scale > tiny))
                if not active.any():
                    continue

                active_rows = np.flatnonzero(active)
                active_frames = windowed[active] / scale[active, None]
                spectrum = np.fft.rfft(active_frames,
                                       n=autocorrelation_fft_size, axis=1)
                power = spectrum.real * spectrum.real \
                    + spectrum.imag * spectrum.imag
                autocorrelation = np.fft.irfft(
                    power, n=autocorrelation_fft_size, axis=1)[:, :order + 1]
                zero_lag = autocorrelation[:, 0]
                good = np.isfinite(autocorrelation).all(axis=1) \
                    & np.isfinite(zero_lag) & (zero_lag > tiny)
                if not good.any():
                    continue
                local_rows = active_rows[good]
                full_rows = first + local_rows
                normalized_acf = (autocorrelation[good]
                                  / zero_lag[good, None])
                a, k, error = _levinson_durbin_batch(
                    normalized_acf, order, config.reflection_limit)
                coefficients[full_rows] = a
                reflection[full_rows] = k
                prediction_error[full_rows] = error
                cepstra[full_rows] = _lpc_cepstral_coefficients(
                    a, cepstral_order)

                response = np.fft.rfft(a, n=spectrum_fft_size, axis=1)
                log_power_shape = -2.0 * np.log(
                    np.maximum(np.abs(response), 1e-12))
                log_power_shape -= log_power_shape.mean(axis=1, keepdims=True)
                log_spectrum[full_rows] = np.clip(log_power_shape, -80.0, 80.0)

        # A zero/unvoiced suffix has no LPC shape of its own.  Edge-hold the
        # spectral descriptors after the last valid voiced frame so downstream
        # sequences do not see a fabricated drop to an all-zero envelope.  Keep
        # raw predictor coefficients, F0, voicing and RMS as measured; if there
        # was no valid voiced frame (silence/uninitialized input), do nothing.
        valid_voiced = np.flatnonzero(
            (f0 > 0.0) & (energy > 0.0) & (energy >= config.silence_rms))
        if valid_voiced.size:
            tail_start = int(valid_voiced[-1]) + 1
            if tail_start < n_frames:
                cepstra[tail_start:] = cepstra[tail_start - 1]
                log_spectrum[tail_start:] = log_spectrum[tail_start - 1]

        frequencies = np.linspace(0.0, fs / 2.0, spectrum_bins,
                                  dtype=np.float64)
        return V2VAnalysis(
            sample_rate=fs, audio_samples=int(signal.size),
            frame_period_ms=float(config.frame_period_ms),
            frame_times_s=frame_times_s,
            center_sample_indices=center_samples,
            backend=backend_name, f0_hz=f0, energy_rms=energy,
            lpc_coefficients=coefficients,
            reflection_coefficients=reflection,
            prediction_error_fraction=np.clip(prediction_error, 0.0, 1.0),
            lpc_cepstra=cepstra, lpc_log_spectrum=log_spectrum,
            spectrum_frequencies_hz=frequencies, lpc_order=order,
            cepstral_order=cepstral_order,
            frame_length_samples=frame_length)


def analyze_v2v(audio: np.ndarray, fs: int,
                config: Optional[V2VAnalysisConfig] = None,
                vocoder: Optional[Vocoder] = None) -> V2VAnalysis:
    """Convenience function for one V2V frontend analysis."""
    return V2VFrontend(config=config, vocoder=vocoder).analyze(audio, fs)


def _centered_frames(signal: np.ndarray, centers: np.ndarray,
                     frame_length: int) -> np.ndarray:
    """Extract zero-padded frames centred at exact sample indices."""
    offsets = np.arange(frame_length, dtype=np.int64) - frame_length // 2
    indices = centers[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < len(signal))
    frames = np.zeros(indices.shape, dtype=np.float64)
    if valid.any():
        rows, columns = np.nonzero(valid)
        frames[rows, columns] = signal[indices[rows, columns]]
    return frames


def _row_rms(frames: np.ndarray) -> np.ndarray:
    """Overflow-resistant RMS for each zero-padded row."""
    scale = np.max(np.abs(frames), axis=1)
    normalized = np.zeros_like(frames)
    np.divide(frames, scale[:, None], out=normalized,
              where=scale[:, None] > 0.0)
    return scale * np.sqrt(np.mean(normalized * normalized, axis=1))


def _levinson_durbin_batch(normalized_acf: np.ndarray, order: int,
                           reflection_limit: float
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Biased-autocorrelation LPC with Schur-stable reflection clipping.

    ``normalized_acf`` has r[0] == 1.  The returned predictor polynomial uses
    ``x[n] + sum(a[k] * x[n-k])`` as the one-step residual convention.
    """
    n_frames = normalized_acf.shape[0]
    coefficients = np.zeros((n_frames, order + 1), dtype=np.float64)
    coefficients[:, 0] = 1.0
    reflection = np.zeros((n_frames, order), dtype=np.float64)
    error = np.ones(n_frames, dtype=np.float64)

    # When a frame is nearly deterministic (a clean sustained tone is the
    # common case), its prediction error can collapse long before the requested
    # order.  Continuing the recursion after that point magnifies round-off in
    # the denominator, so leave the remaining reflection coefficients at zero.
    active = np.ones(n_frames, dtype=bool)
    error_stop = 1e-8
    for degree in range(1, order + 1):
        numerator = normalized_acf[:, degree].copy()
        if degree > 1:
            numerator += np.sum(
                coefficients[:, 1:degree]
                * normalized_acf[:, degree - 1:0:-1], axis=1)
        eligible = active & (error > error_stop)
        k = np.zeros(n_frames, dtype=np.float64)
        if eligible.any():
            raw_k = (-numerator[eligible]
                     / np.maximum(error[eligible], 1e-12))
            raw_k = np.nan_to_num(raw_k, nan=0.0,
                                  posinf=reflection_limit,
                                  neginf=-reflection_limit)
            k[eligible] = np.clip(raw_k, -reflection_limit, reflection_limit)
        if degree > 1:
            previous = coefficients[:, 1:degree].copy()
            coefficients[:, 1:degree] = previous + k[:, None] * previous[:, ::-1]
        coefficients[:, degree] = k
        reflection[:, degree - 1] = k
        error = np.clip(error * (1.0 - k * k), 0.0, 1.0)
        active &= error > error_stop

    return coefficients, reflection, np.clip(error, 0.0, 1.0)


def _lpc_cepstral_coefficients(coefficients: np.ndarray,
                               cepstral_order: int) -> np.ndarray:
    """Cepstrum c1..cN of the inverse LPC polynomial, without gain c0.

    The recursion is the power-series expansion of ``log(1 / A(z))``.  In the
    Fourier domain, the real log-amplitude response is
    ``Re(sum(c[m] * exp(-j*w*m)))``.  This avoids treating raw LPC coefficients
    as if they were already a well-conditioned cepstral feature vector.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    order = coefficients.shape[1] - 1
    cepstrum = np.zeros((len(coefficients), cepstral_order + 1),
                        dtype=np.float64)
    for degree in range(1, cepstral_order + 1):
        value = (-coefficients[:, degree].copy()
                 if degree <= order else np.zeros(len(coefficients)))
        for previous_degree in range(1, degree):
            lpc_index = degree - previous_degree
            if lpc_index <= order:
                value -= ((previous_degree / float(degree))
                          * cepstrum[:, previous_degree]
                          * coefficients[:, lpc_index])
        cepstrum[:, degree] = value
    return cepstrum[:, 1:]


__all__ = ["V2VAnalysisConfig", "V2VAnalysis", "V2VFrontend", "analyze_v2v"]
