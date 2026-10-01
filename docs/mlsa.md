# The MLSA vocoder backend

`vocoder="mlsa"` / `hms synth --vocoder mlsa` selects the second pure-numpy
backend, next to the `builtin` fallback.  It consumes exactly the same acoustic
representation as every other backend — `f0` (Hz, `0.0` = unvoiced), `sp`
(linear power spectral envelope on the WORLD grid) and `ap` (aperiodicity in
`[0, 1]`) — and turns it into a waveform with the classical **MLSA** (*mel log
spectrum approximation*) synthesis filter instead of a zero-phase magnitude
filter.  This page documents exactly which formulation is implemented, why it
is shaped the way it is, and what it does not do.

## 1. Formulation

### 1.1 The filter

The MLSA filter is the exponential of a mel-cepstral polynomial evaluated on a
warped frequency axis [Imai 1983; Fukada et al. 1992]:

```
H(z) = exp( sum_{m=0}^{M-1} a_m * z~^{-m} )          z~^{-1} = (z^{-1} - a) / (1 - a z^{-1})

|H(e^{jw})| = exp( sum_m a_m cos(m beta(w)) )
phase(H)    = -sum_m a_m sin(m beta(w))
beta(w)     = w + 2 atan2(a sin w, 1 - a cos w)
```

`z~^{-1}` is a first-order all-pass; `beta(w)` is the warped (mel-like)
frequency; `a` is the all-pass coefficient; `a_m` are the **log-amplitude**
mel-cepstral coefficients (`0.5 x` the log-power ones).  `|H|^2` is therefore
the smooth interpolation of the modelled log-power envelope across mel — the
"mel log spectrum approximation".

The coefficients are *not* a parallel parameter space.  HMS already models the
mel-cepstrum (`FeatureSpec` streams: `[c_0 ... c_{M-1}]` = DCT-II of the log
mel-band powers), and `mcep_to_power` reconstructs `sp` from it by interpolating
between band centres.  `mlsa_log_amplitude` converts those same DCT
coefficients into the Fourier coefficients of the same curve on the warped
axis; evaluated at the analysis knots, the MLSA series reproduces
`mcep_to_power` exactly (verified to machine precision in
`hms/tests/test_mlsa.py::test_filter_reproduces_the_projects_own_cepstral_reconstruction`).

### 1.2 Realization: exact response + per-frame FIR, overlap-add

SPTK realizes `H(z)` with an Imai ladder / continued fraction: exact to a
configured order, but a sequential recursion costing `O(M)` multiply-adds *per
sample* — in pure Python that is far slower than real time.  This backend
evaluates the same transfer function **exactly on an FFT grid** and realizes it
as a per-frame FIR (the IFFT of the response), applied by overlap-add:

* frame `t`'s excitation chunk is one hop long; it is convolved with the
  current impulse response and overlap-added;
* if the filter were constant this reproduces the true convolution sample for
  sample — the time-varying case is the standard partitioned-convolution
  approximation, with coefficient changes cross-faded over the filter length;
* the response decays like `a^n` (`a <= 0.6` for every HMS sample rate), so the
  FIR cut-off is chosen as `FILTER_PERIODS = 2` frame periods (256 taps at
  22.05 kHz, 512 at 44.1 kHz).  Truncation error is below -100 dB at the
  default length; it can be overridden with `MLSAVocoder(filter_length=...)`.

Because the filter is truncated rather than recursive, there is no filter state
and no stability question: the coefficients may jump arbitrarily between frames.

### 1.3 Warping coefficient

There is no closed form for the `a` whose all-pass image best matches a mel
scale, so `mel_warping_factor(fs)` fits it by least squares against the
project's own HTK-style mel scale (`1127.01 ln(1 + f/700)`), on a fixed grid,
and caches the result.  The fitted values are close to the conventional
SPTK/HTS choices:

| fs (Hz) | 8000 | 10000 | 16000 | 22050 | 44100 | 48000 |
|---|---|---|---|---|---|---|
| `a` | 0.362 | 0.394 | 0.459 | 0.502 | 0.585 | 0.595 |

The mel axis is not exactly a first-order all-pass image, so the warped
evaluation differs from `mcep_to_power`'s piecewise-linear interpolation
between knots by a sub-band amount — for the orders HMS uses (20-40), well
under a decibel on average.  The two agree exactly at the knots.

## 2. Excitation

MLSA shapes an excitation; the backend builds a mixed one:

* **voiced frames** — a band-limited pulse train (Hann-windowed sinc kernel,
  cut-off at 0.9 x Nyquist) plus white noise, mixed **per frequency bin** with
  `sqrt(1 - ap)` and `sqrt(ap)`.  That is the same convention WORLD's synthesis
  uses, and it is what preserves a frame's frequency-dependent voicing: ap that
  is low in the low band and high in the high band renders as such, instead of
  being averaged into one per-frame number.
* **unvoiced frames** (`f0 <= 0`) — pure noise, whatever `ap` says.
* **level** — pulse positions are tracked exactly (never accumulated from
  rounded samples) and placed through the fractional-shift kernel; each pulse
  carries exactly one period's worth of energy, so the train's mean square is 1
  at any F0.  The noise is unit-variance.  With `|H|^2 ~ sp`, the output power
  therefore tracks `sp` (~1.08x of `mean(sp)` on the demo corpus) without a
  global gain constant, and loudness does not follow pitch (within 0.3 dB over
  100-800 Hz in the test suite).
* **determinism** — the noise comes from `numpy.random.default_rng(seed)`,
  re-seeded for every `synthesize` call (`seed=12345` by default), so identical
  parameters give bit-identical output.
* **pulse phase** is re-locked to the frame grid at the start of every voiced
  run, as the builtin backend does.

## 3. Timing, level and memory conventions

* **Duration** is WORLD's: exactly `int(f0_length * frame_period * fs)`
  samples.  The excitation is generated past the last frame (replicating the
  last frame's F0, the same edge convention `add_dynamic_features` uses) so the
  trailing samples of a render keep their periodic excitation when the sample
  grid is longer than `n_frames * hop` (5 ms = 220.5 samples at 44.1 kHz).
* **No peak normalisation.**  Like the WORLD backends — and unlike the builtin
  fallback, whose absolute level is arbitrary and which applies `limit_peak` —
  the waveform is returned as produced; `wavio.write_wav` applies the headroom.
  A very periodic render can exceed `±1`, exactly as WORLD's can.
* **Blocked, bounded memory.**  Synthesis is processed in blocks of 256 frames
  (`MLSAVocoder._block`): the response, the per-bin weights and the FFTs are
  built per block, so no `(T, n_fft)` filter ever exists.  What remains is
  `O(n_samples)` for the pulse train, the noise and the output.  A 60 s render
  at 22.05 kHz peaks around 92 MB of Python heap (native WORLD: 77 MB).
* **Analysis** is delegated to `BuiltinVocoder`: MLSA is a synthesis filter,
  so `hms extract`/`hms train --vocoder mlsa` produce exactly the same
  parameters as `--vocoder builtin`.  (There is no SPTK-style mel-cepstral
  *analysis* here, and none is needed: the model's features already are the
  mel-cepstrum.)

## 4. What was measured

Reproduce with `python tools/bench_vocoder.py` (same parameters for every
backend, best of three calls, `tracemalloc` peak measured in a separate pass):

```
22.05 kHz, frame period 5 ms                    RTF        peak heap
  1 s   mlsa     0.0096  (104x real time)       5.5 MB
  1 s   builtin  0.0067  (150x)                 1.3 MB
  1 s   native   0.0208  ( 48x)                 1.3 MB
 20 s   mlsa     0.0081  (124x)                36.0 MB
 20 s   builtin  0.0067  (150x)                21.2 MB
 20 s   native   0.0207  ( 48x)                25.7 MB
 60 s   mlsa     0.0077  (130x)                91.9 MB
 60 s   builtin  0.0074  (136x)                63.7 MB
 60 s   native   0.0214  ( 47x)                77.0 MB
```

**Is MLSA faster or smaller than the existing builtin backend? No, neither**
— it is 1.0-1.4x slower and uses 1.4-4.2x the peak heap (it materialises the
pulse and noise excitations and works in vectorised blocks), and both backends
run ~100x faster than real time.  It is, however, **2.3-2.8x faster than the
native WORLD synthesis** at comparable memory.

What the extra time buys, measured on the demo model and score at 22.05 kHz
(re-analysing each render with native WORLD):

| backend | frames WORLD still calls voiced | median pitch error | pitch error p95 | LF/HF harmonic-to-noise* |
|---|---|---|---|---|
| mlsa | 0.74 (target 0.76) | 2.2 cents | 28.4 cents | 21.0 / 1.1 dB |
| builtin | 0.44 (target 0.76) | 15.6 cents | 68.3 cents | 5.3 / 0.7 dB |
| native | 0.75 (target 0.76) | 0.5 cents | 6.5 cents | 37.4 / 0.7 dB |

\* a synthetic frame with `ap = 0.02` below 2 kHz and `ap = 0.98` above it —
the per-bin mixing keeps ~16 dB more harmonic structure in the low band than
the builtin's per-frame scalar mixing does.

Spectral fidelity of the reconstructed envelope is similar to the builtin's
(re-analysis of a render correlates with the target `sp` at 0.988-0.992, mean
absolute error 0.6 dB); the difference is in excitation and voicing, not in the
envelope.

## 5. Limitations

* **Frame-wise frozen coefficients.**  As in every frame-based vocoder, the
  filter is held constant inside a frame and cross-faded by the overlap-add, so
  a parameter change faster than one frame is smoothed over the filter length
  (two frame periods by default).
* **Warping is fitted, not exact.**  The all-pass axis approximates the mel
  scale in least squares (Section 1.3), so the MLSA evaluation of the
  mel-cepstrum differs from `mcep_to_power` between the analysis knots by a
  sub-band amount.  At the knots it is exact.
* **No excitation phase model.**  Pulses are re-locked to the frame grid at
  every voiced-run boundary and the noise is a fresh (seeded) draw; there is no
  phase continuation across unvoiced gaps and no pulse-shape/glottal model.
* **Higher memory than the builtin fallback** (Section 4), because the
  excitation signals are materialised for the whole utterance.
* **No peak normalisation**, by design; the caller decides (see
  `Vocoder.synthesize`).
* **Analysis is shared**, not MLSA-specific: the backend does not implement
  SPTK-style mel-cepstral analysis, and `--vocoder mlsa` therefore extracts the
  same parameters as `--vocoder builtin`.
* **Not WORLD.**  It renders the modelled envelope with a different filter and
  a different excitation; it is not intended to be indistinguishable from
  WORLD's synthesis, only to be a better-motivated pure-numpy alternative to
  the builtin fallback (`vocoder="auto"` still resolves to WORLD when
  available, and to `builtin` otherwise — MLSA is opt-in).

## 6. References and tests

* Imai, S. (1983). *Cepstral analysis synthesis on the mel frequency scale.*
* Fukada, T., Tokuda, K., Kobayashi, T., Imai, S. (1992). *An adaptive
  algorithm for mel-cepstral analysis of speech.*
* Implementation: `hms/vocoder/mlsa.py`; contract and edge-case tests:
  `hms/tests/test_mlsa.py`; duration conventions shared with every backend:
  `hms/tests/test_vocoder.py`.
