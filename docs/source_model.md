# Source/excitation model (Phase 1)

HMS has always modelled the **filter**: a mel-cepstrum envelope per frame, learned
by the HMM/GMM. The **source** was left to the vocoder, which invents it from
scratch (a pulse train plus noise mixed by the aperiodicity). That is a good
synthesiser, but the excitation is a fixed recipe rather than something a model
can learn, and the recipe only knows how to be a voice.

This document covers **Phase 1 of a source-aware HMS**: a generic representation
of the excitation, a first (voice) backend that extracts it from real audio, a
NumPy PCA that compresses it, and the measurements that say how much survives.
It is deliberately *analysis only* — nothing in the trainer, the parameter
generator or the vocoder calls any of it, so the existing synthesiser is
unchanged bit for bit.

The eventual architecture, once a later phase learns a model over these
parameters, is

```
                 ┌── spectral model ──→ spectral envelope ──┐
    HMM output ──┤                                           ├─→ filter ─→ audio
                 └── source model ────→ excitation ─────────┘
```

i.e. the HMM output is *split*: one half stays what it is today (mel-cepstrum +
aperiodicity → spectral envelope), the other half describes the excitation and
drives a source generator in front of the same filter. Phase 1 builds the right
half's data path and proves it carries information; Phase 2 would give it
parameters a model can generate.

## Pipeline

```
audio ──► F0 (supplied, or estimated) ──► epochs ──► inverse filter ──► residual
                                                        │
                       one cycle per period ◄───────────┘
                                │
                       128-sample, unit-RMS vector
                                │
                        PCA coefficients (4–16)
```

* **F0** comes from the caller when there is one (`analyze(x, fs, f0=...)`): in
  HMS the vocoder or the score already knows the pitch, and a track from outside
  is authoritative, never smoothed. Without it the backend runs the same
  pure-NumPy autocorrelation + spectral-flatness estimator the fallback vocoder
  uses (`hms.core.dsp`), with a source-side variant that can trade time
  resolution for low-pitch robustness (`f0_window_periods`, see below).
  Non-finite, negative and out-of-range values become unvoiced or are clamped —
  a bad track costs cycle spacing, never a malformed vector.
* **Epochs** are found by *phase accumulation* over the per-sample period, not
  by picking periods independently: an F0 jump changes the slope of the phase,
  it cannot make epochs run backwards or jump a whole period. Each voiced run
  starts its own phase, so an unvoiced gap never drags a fractional phase into
  the next run, and a cycle is kept only when both its epochs come from the same
  run and its whole span lies inside the signal. Epochs are then snapped onto the
  strongest residual sample within ±`refine_ratio` periods, bounded by the
  neighbouring epochs and by the shortest allowed period (so a snap can never
  empty or double a cycle).
* **Inverse filtering** reuses the model's own envelope definition: the
  mel-cepstrum (`hms.core.features.power_to_mcep` / `mcep_to_power`), estimated
  per frame and divided out as a real, zero-phase gain
  (`hms.source.residual.whiten`). Inverting the *shape* (the envelope is
  normalised to unit mean per frame) keeps the residual in the audio's units, so
  a cycle's gain means something; the inversion gain is limited to ~30 dB, which
  is what stops a spectral valley from turning into amplified noise. No scipy,
  no librosa, no new envelope definition to keep in sync.
* **Fixed length**: every cycle is resampled from its measured period to
  `cycle_length` samples (128 by default) with a *periodic*, band-limited FFT
  resample — a cycle is one period of a periodic signal, so the resample wraps
  around instead of interpolating between two ends that were never adjacent.
  The result is then normalised to exactly unit RMS, and the level it lost is
  stored separately as `gain`.

The 128-sample length is a parameter, not part of the architecture
(`SourceModel(cycle_length=...)`), and it is the representation's bandwidth
limit: a 128-sample vector cannot carry source energy above 64 harmonics per
period. Everything downstream — PCA, placement, synthesis — uses the same
length.

## Representation

`hms/source/base.py` defines the backend-independent pieces:

| object | what it is |
|---|---|
| `SourceFrame` | one source unit: `f0`, `voiced`, `excitation`, `source_coefficients`, `noise_level`, `gain`, `epoch`, `period`, `index` |
| `SourceSequence` | one utterance: per-frame `f0`/`voiced`/`noise_level` on the HMS frame grid, plus per-unit `excitation`/`gains`/`epochs`/`periods` |
| `SourceModel` | the interface: `analyze` (backend specific) and the shared `encode`/`decode`/`synthesize` |

Nothing in `base.py` knows about glottal pulses, vowels or voices. A unit is one
pitch period for a periodic source and one analysis frame for an unpitched or
unknown one; `excitation` is always a fixed-length unit-RMS vector, `gain` the
level that was removed, and `noise_level` a cheap concentration measure (0 = one
clean excitation event, 1 = energy spread over the unit). An invalid unit is
*dropped*, never filled with a placeholder, so `excitation` only ever contains
finite, unit-RMS vectors.

Two backends ship with Phase 1 (`get_source_model(name)`):

| name | what it does | when it applies |
|---|---|---|
| `voice` | pitch-synchronous residual cycles: epoch tracking, one cycle per period | singing voice, any quasi-periodic source with a usable F0 |
| `residual` | frame-synchronous whitened residual, one unit per analysis frame | anything without a usable pitch (fricatives, breath, percussion, wind) |

The registry mirrors `hms.vocoder`: `SOURCE_BACKENDS`, `available_backends()`,
`get_source_model(name, **kwargs)`, `source_model_for(kind)`. A new physical
source type (lip-reed, bowed string, drum) only has to fill the same fields in
its own `analyze`; the PCA, the codec, the reconstruction measurement and the
eventual HMM coupling are shared.

## PCA

`hms/source/pca.py` is a NumPy-only PCA over the source vectors — no sklearn,
no covariance matrix. `SourcePCA.fit(cycles, n_components=8)` centres the data
and takes the SVD directly (better conditioned than an eigendecomposition of
`XᵀX`), fixes each component's sign so a saved basis is reproducible, and fills
any direction the data does not constrain with a deterministic orthonormal
vector so the basis is always complete and well formed. Fitting more components
than the data has rank is not an error: the extra components get zero
eigenvalue and are marked inactive, and `encode` zeroes their coefficients
rather than projecting noise onto them.

```
pca = SourcePCA.fit(cycles, n_components=8)
coeffs = pca.encode(cycles)               # (K, 128) -> (K, 8)
rebuilt = pca.decode(coeffs)              # (K, 8) -> (K, 128)
report = pca.report(held_out_cycles)      # mse, rmse, relative_rmse, p90, corr
pca.save("source.npz"); pca2 = SourcePCA.load("source.npz")
```

`report()` returns `n_vectors`, `mse`, `rmse`, `relative_rmse`,
`median_relative_error`, `p90_relative_error`, `mean_correlation`,
`explained_variance` (of the data it is given) and
`cumulative_explained_variance` (of the fitted basis). An empty input reports
NaN errors instead of raising — a benchmark with no usable cycles should print
"nothing", not crash.

## The experiment: 128 samples → *k* numbers → cycles

```bash
python tools/bench_source_pca.py                       # demo corpus, 22050 Hz
python tools/bench_source_pca.py --components 4 8 16 --out out/source
python tools/bench_source_pca.py --wav my_singing.wav --fs 44100
```

The benchmark renders the demo corpus, concatenates every phrase into one
32.11 s signal, extracts cycles, fits the basis on the first half and measures
on the second. With the defaults (`cycle_length=128`, `n_mcep=30`,
`refine_ratio=0.35`):

```
audio              : 32.11 s, 22050 Hz, 6441 frames @ 5 ms
voiced frames      : 5314 / 6441 (82.5 %)
valid source units : 8419 extracted cycles (8419 source vectors)
cycle period       : 11 .. 207 samples, median 67 (F0 106.5 .. 2004.5 Hz, median 329.1 Hz)

all cycles  (basis fitted on the first half, measured on the second)
   k  expl. var       MSE     RMSE  rel RMSE  median err  p90 err    corr
   1      0.232    0.6476   0.8047     0.805       0.796    0.991   0.537
   4      0.485    0.4457   0.6676     0.668       0.665    0.897   0.718
   8      0.663    0.3155   0.5617     0.562       0.514    0.786   0.813
  16      0.852    0.1597   0.3997     0.400       0.347    0.567   0.912
  32      0.983    0.0174   0.1319     0.132       0.063    0.212   0.991

per pitch (same period, basis fitted and measured within the group)
  F0 Hz  cycles                   k=4                   k=8                  k=16
  393.8     369     0.807/     0.493     0.919/     0.437     0.989/     0.337
  350.0     347     0.767/     0.572     0.892/     0.487     0.976/     0.342
  294.0     341     0.824/     0.560     0.911/     0.507     0.981/     0.396
  262.5     301     0.877/     0.594     0.961/     0.515     0.995/     0.403
  355.6     294     0.814/     0.590     0.926/     0.534     0.987/     0.421
  298.0     270     0.814/     0.549     0.916/     0.492     0.985/     0.319
  334.1     268     0.807/     0.391     0.929/     0.284     0.991/     0.150
  329.1     255     0.774/     0.466     0.925/     0.337     0.991/     0.200
(cells: explained variance / held-out relative RMSE)

longest steady stretch  (87 cycles at ~374 Hz)
   k  expl. var       MSE     RMSE  rel RMSE  median err  p90 err    corr
   4      0.862    0.0068   0.0823     0.082       0.076    0.111   0.997
   8      0.969    0.0037   0.0606     0.061       0.057    0.086   0.998
  16      0.998    0.0010   0.0311     0.031       0.025    0.047   1.000

cycle round trip   : correlation 1.0000, relative error 0.0059
                     (residual -> cycles -> residual, no PCA: the loss is the
                      fixed-length resampling)
```

How to read it:

* **The framework is not the bottleneck.** Cycles survive the round trip through
  the fixed-length representation almost exactly (correlation 1.0000, 0.6 %
  relative error), and 32 coefficients reach 0.13 relative RMSE on held-out
  data. The question is only how far *few* coefficients go.
* **A handful of coefficients is enough when the context is fixed.** On a
  steady stretch — one note, one vowel, no articulatory movement — 4
  coefficients reproduce the average cycle to 8 % relative error, 8 to 6 %, with
  correlation 0.998.
* **Across mixed material, 4–8 coefficients are a coarse sketch.** Pooled over
  every cycle in the corpus (different notes, vowels and consonants, half of
  them *unseen* by the fit) 8 coefficients give relative RMSE 0.56 /
  correlation 0.81; conditioning on the pitch alone (per-pitch groups, which is
  what a Phase 2 source model conditioning on the note would provide) improves
  that to 0.28–0.53 relative RMSE with explained variance 0.89–0.96. 16
  coefficients reach 0.40 pooled and 0.15–0.42 per pitch.
* **The residual loss is conditioning, not capacity.** The same 8 coefficients
  give 0.06 relative RMSE inside one steady stretch and 0.56 on the whole
  corpus. What separates the two is that the corpus mixes excitation shapes from
  different pitches, vowels and consonant transitions; a source model that
  conditions on those (an HMM over source coefficients, coupled to the spectral
  model) is exactly what Phase 2 would add.
* Note that `mean_correlation` and relative RMSE are different lenses on the
  same thing: a relative RMSE of 0.5 corresponds to a correlation of ~0.86
  (measured at k=12: 0.496 / 0.860 — relative RMSE is an energy error, the
  correlation is not).

## Measured design choices

These were tuned against that benchmark; each row is a measurement, not a guess.

**Inverse-filter gain limit** (`DEFAULT_GAIN_LIMIT = 30.0`, i.e. ~30 dB of
envelope inversion). The per-frame envelope is clipped to this dynamic range
before dividing. On a mixed-vowel test the adjacent-cycle correlation is 0.96 at
30 dB and 0.62 at 60 dB: inverting a spectral valley by more than ~30 dB mostly
amplifies the noise floor. Raise it for a very clean source, lower it for a
noisy one.

**Epoch refinement** (`refine_ratio`, how far an epoch may snap toward the
strongest residual sample in its neighbourhood). Held-out relative RMSE /
cycle-to-cycle correlation at k=8 on the corpus: off 0.713/0.795, 0.15
0.657/0.939, 0.25 0.603/0.950, **0.35 0.569/0.956**, 0.50 0.529/0.963. The
default is 0.35: larger windows keep improving the numbers because excitation
events are what the snap locks onto, but past half a period an epoch could reach
its neighbour's event, which is the failure the neighbour bound exists to
prevent. The refinement runs **per voiced run**: an epoch's neighbours, its local
period and the span it may move inside all come from its own run, never from a
neighbouring run or the unvoiced gap between them
(`hms/tests/test_source_cycles.py::test_epoch_refinement_never_uses_a_neighbour_from_another_voiced_run`).

**F0 analysis window** (`f0_window_periods`). The default 0 keeps the plain
`4 * hop` window the rest of HMS uses, which tracks pitch and vibrato closely.
Stretching the window to hold a few periods of `f0_floor` is what resolves low
pitches and formant lock-in (a 60 Hz synthetic note under a 735 Hz formant is
estimated at 735 Hz with the default window and at 60 Hz with
`f0_window_periods=3`), but it costs time resolution and can lock an octave
*low* on high-pitched material (900 Hz read as 459 Hz at 22.05 kHz). It is
therefore opt-in, and the honest summary is: **supply an F0 track when you have
one** — the fallback estimator is a convenience, not the intended source of
pitch.

**The gain convention.** A stored vector is exactly unit RMS; `gain` is the RMS
of the analysed cycle band-limited to `cycle_length` harmonics, so
`vector * gain * scale` restores the analysed level. The band limit is the only
level loss and it exists only for periods longer than `cycle_length` samples
(a 200-sample cycle resampled to 128 loses ~19 % of its RMS; see
`hms/tests/test_source_cycles.py`).

## What Phase 1 does not do

* No HMM is trained on the source coefficients, and the acoustic parameter model
  is untouched.
* Nothing in `hms.core.synthesizer`, `hms.core.generation` or the vocoders calls
  the source model. `SourceModel.synthesize` renders the *excitation* on the
  sample grid — it is a validation path, not a new vocoder.
* Unvoiced excitation is not modelled: unvoiced frames are marked (noise level
  1.0) but their excitation is still the vocoder's noise.
* The representation is band-limited to `cycle_length / 2` harmonics per period,
  and the residual is a zero-phase inverse-filter output, not a claimed glottal
  flow estimate. A minimum-phase inverse filter would not change what the PCA
  can represent here (measured: no gain over the zero-phase division).
* Conditioning the source model on pitch, phoneme or context is left to the
  phase that learns it — the numbers above say that is where the remaining
  error lives.

## Where the code lives

| path | contents |
|---|---|
| `hms/source/base.py` | `SourceFrame`, `SourceSequence`, `SourceModel`, `assign_frame_noise` |
| `hms/source/residual.py` | mel-cepstrum whitening / inverse filtering |
| `hms/source/cycles.py` | epoch tracking, cycle extraction, resampling, placement |
| `hms/source/voice.py` | `VoiceSourceModel`, `estimate_f0`, `smooth_f0` |
| `hms/source/generic.py` | `GenericResidualSourceModel` (pitch-free backend) |
| `hms/source/pca.py` | `SourcePCA` (fit / encode / decode / save / load / report) |
| `tools/bench_source_pca.py` | the experiment above |
| `hms/tests/test_source_cycles.py`, `test_source_pca.py`, `test_source_model.py` | the unit tests (85 tests: cycles, PCA, model API and reconstruction) |
