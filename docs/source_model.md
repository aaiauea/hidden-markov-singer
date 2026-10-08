# Source/excitation model (Phase 1 + Phase 2 + Phase 3)

HMS has traditionally modelled the **filter**: a mel-cepstrum envelope per
frame, learned by the acoustic HMM/GMM. The **source** was left to the vocoder,
which invents it from scratch (a pulse train plus noise mixed by the
aperiodicity). Phase 1 added a generic source representation, analysis backends
and NumPy PCA. Phase 2 added a separate statistical predictor over those
existing PCA coefficients. Phase 3 connects the two branches: the learned source
is rendered as an excitation and drives the vocoder's filter.

The completed architecture is

```
labels + explicit F0 ──► source HMM/GMM ──► PCA coefficients ──► excitation ─┐
                                                                          ├─► filter ─► audio
labels (+ score F0) ───► acoustic HMM/GMM ─► spectral envelope / AP ──────┘
```

The acoustic branch is untouched by all three phases. The source branch predicts
the Phase-1 representation independently from phoneme/context labels and an
explicit frame-level F0 trajectory, and Phase 3 is the seam where the two meet.

* [Phase 1](#phase-1-source-representation) — representation, backends, PCA.
* [Phase 2](#phase-2-context--f0-conditioned-coefficient-prediction) — the
  standalone predictor.
* [Phase 3](#phase-3-source-aware-synthesis) — rendering the prediction as an
  excitation and filtering it.

Phase 3 is opt-in and additive: `Synthesizer.synthesize(score)` without a
`source_model` is bit-identical to the synthesis HMS has always done, and no
acoustic model file, vocoder API or Phase-1/Phase-2 API changed meaning.

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
  The Phase-1 analyzer sanitizes non-finite/negative and out-of-range values
  for cycle extraction. The Phase-2 predictor is stricter: an explicit track
  must be finite and non-negative, is never clamped or resized, and uses the
  configured threshold only to derive voicing.
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
  correlation 0.81; conditioning on pitch alone (per-pitch groups, an oracle
  rather than a predictor) improves that to 0.28–0.53 relative RMSE with
  explained variance 0.89–0.96. Phase 2 adds an explicit F0-conditioned
  predictor, but these benchmark figures remain representation measurements,
  not a quality claim for the trained predictor. 16 coefficients reach 0.40
  pooled and 0.15–0.42 per pitch.
* **The residual loss is conditioning, not capacity.** The same 8 coefficients
  give 0.06 relative RMSE inside one steady stretch and 0.56 on the whole
  corpus. What separates the two is that the corpus mixes excitation shapes from
  different pitches, vowels and consonant transitions; Phase 2 now models those
  dependencies in a separate HMM/GMM over source coefficients, selected by
  phone context and conditioned on explicit per-frame F0. It remains independent
  of the acoustic HMM and is not yet coupled into full waveform synthesis.
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

## Phase 2: context- and F0-conditioned coefficient prediction

Phase 2 lives in `hms/source/trainer.py` and `hms/source/hmm.py`. It consumes
Phase-1 `SourceSequence` units and the existing `SourcePCA`; it does **not**
change source analysis, create a new representation, or modify the acoustic
HMS model.

### Training flow

1. `SourceTrainingExample` pairs one `Utterance`, one Phase-1 `SourceSequence`,
   and an optional explicit F0 track. If the track is omitted, the sequence's
   existing `source.f0` is used. `SourceTrainingExample.from_audio(...)` is only
   a convenience wrapper around the selected Phase-1 analyzer.
2. A supplied `SourcePCA` is reused unchanged. If none is supplied,
   `SourceTrainingConfig.pca_components` controls a call to the existing
   `SourcePCA.fit` (default: 8).
3. The PCA coefficients on source units are overlap-resampled to the analysis
   frame grid: each unit contributes according to its sample overlap with
   `[frame * hop, (frame + 1) * hop)`. Multiple fast units are averaged, a
   longer pitch cycle contributes to each covered frame, and uncovered frames
   stay invalid—there is no interpolation across missing units or unvoiced
   holes. Training uses only labelled frames with valid source coverage.
4. The normalized static coefficient stream and its optional HMS delta and
   delta-delta streams are collected by phone occurrence. The existing
   `LeftToRightHMM` / `DiagGMM` and `DurationModel` train the phone units. The
   sparse context tier reuses `hms.core.context`: observed exact triphones and,
   when enabled, one-sided diphones are kept only when their configured frame
   and occurrence thresholds are met. Selection falls back through triphone,
   best-supported diphone, phone, class backoff and optional global backoff.
5. Each HMM state gets component-gated, regularized linear regressions from the
   explicit F0 features to the static/dynamic PCA observation means. The F0
   input is normalized semitone log-F0 relative to `FeatureSpec.f0_ref_hz`, its
   first HMS delta, and an explicit voiced flag. Delta calculation is confined
   to voiced runs so an unvoiced gap cannot create a large artificial pitch
   jump. The source HMM's observation dimension is only
   `pca.n_components * number_of_dynamic_streams`; categorical context is used
   for HMM selection, not concatenated into the GMM vector.

The explicit F0 track is authoritative: it must be finite, non-negative, and
one value per source analysis frame; it is never estimated, clamped, stretched,
or silently resized by the Phase-2 trainer or generator. Use `0 Hz` for
unvoiced frames (values at or below the configured voiced threshold are also
treated as unvoiced). Sample rate and frame period must match the `FeatureSpec`; there
is no implicit resampling. When creating examples from audio, pass the F0 track
on the exact frame grid returned by that Phase-1 analyzer.

Example (the variables `utterances`, `waveforms`, `f0_tracks`, `spec`, and
`phoneme_set` come from the caller's existing data pipeline):

```python
from hms.source import (SourceTrainer, SourceTrainingConfig,
                        SourceTrainingExample, get_source_model)

analyzer = get_source_model(
    "voice", cycle_length=128, fs=spec.fs,
    frame_period=spec.frame_period, n_mcep=spec.n_mcep)
examples = [
    SourceTrainingExample.from_audio(
        utt, audio, analyzer, fs=spec.fs, frame_period=spec.frame_period,
        f0_hz=f0)
    for utt, audio, f0 in zip(utterances, waveforms, f0_tracks)
]
config = SourceTrainingConfig(
    pca_components=8, context_enabled=True,
    context_min_frames=100, context_min_occurrences=3)
model = SourceTrainer(spec, phoneme_set, config).fit(
    examples, pca=phase1_pca, name="singer-a")
model.save("models/singer-a-source")
```

### Generation and Phase-1 decoding

`SourceHMMModel.generate(utterance, f0_hz)` deterministically resolves the
labelled phone contexts, allocates each active phone's frames to HMM states by
`DurationModel` proportions, predicts the state/component moments from the
explicit F0 trajectory, and calls the existing banded MLPG solver for a smooth
coefficient path. It returns a `SourcePrediction` with frame-aligned
`frame_coefficients`, F0/voicing/active/state/phone tracks, and a Phase-1
`SourceSequence` in `.sequence`.

The sequence stores PCA-decoded vectors on the Phase-1 backend's native unit
grid: pitch cycles for `voice`, frame groups for `residual`. The `voice` backend
does not invent cycles in unvoiced frames; the existing downstream source or
vocoder path remains responsible for their noise. The frame-synchronous
`residual` backend can predict residual units in voiced and unvoiced labelled
regions. Generated unit gains are neutral (`1.0`): Phase 2 predicts source
shape coefficients, not the Phase-1 gain track or a separate amplitude model.

The returned vectors remain usable through the existing Phase-1 decoder API:

```python
prediction = model.generate(utterance, f0_hz=target_f0)
excitation = analyzer.synthesize(prediction.sequence, pca=model.pca)
# Or reload the standalone source tier later:
from hms.source import SourceHMMModel
loaded = SourceHMMModel.load("models/singer-a-source")
```

`excitation` here is only the source waveform on the sample grid; full filtered
audio is what [Phase 3](#phase-3-source-aware-synthesis) adds. The `seed`
argument is retained for API parity, but MLPG prediction is deterministic and
does not sample.

### Separate model format and current scope

A source tier saves as a standalone directory containing `source.yaml`,
`source_hmms.npz`, and `source_pca.npz`. YAML stores geometry, F0/coefficient
normalization, context/backoff indexes and training options; numeric HMM/GMM and
F0-regression arrays and the Phase-1 PCA basis are stored as NPZ without pickle.
This keeps singer-specific source data separate from the generic acoustic HMS
model.

The Phase-1 representation remains band-limited to `cycle_length / 2`
harmonics per period, and its residual is a zero-phase inverse-filter output,
not a claimed glottal-flow estimate. Phase 2 does not predict Phase-1 unit gains,
infer an F0 track when none is supplied, or add new pitch/vibrato modelling.
Connecting it to the acoustic envelope and the vocoder is Phase 3, below.

`hms/tests/test_source_hmm.py` covers overlap alignment, F0 validation and
pitch-change features, sparse context selection, configurable PCA size,
voiced/unvoiced behavior, deterministic MLPG generation, PCA decoder
compatibility, and YAML/NPZ save/load.

## Phase 3: source-aware synthesis

Phase 3 is an **integration** phase. It redefines no representation, retrains
nothing, and does not touch the acoustic HMM/GMM, the parameter generator or the
feature layout. It adds one module (`hms/source/synthesis.py`), one optional
capability on the vocoder interface, one optional argument on
`Synthesizer.synthesize`, and two CLI flags.

### The integration boundary

The seam is the **excitation waveform on the sample grid** — not PCA
coefficients, not WORLD parameters:

```
SourceHMMModel.generate(utterance, f0)          # Phase 2, unchanged
        │  .sequence  (Phase-1 units, PCA-decoded)
        ▼
hms.source.synthesis.render_source_excitation    # place_cycles + coverage
        │  excitation (n_samples,)  + support / weights / coverage
        ▼
Vocoder.synthesize_with_excitation(params, …)    # the backend's own filter
        │
        ▼
      audio
```

`hms/source/synthesis.py` owns the source: it places the units on the sample
grid with Phase 1's `place_cycles`, marks where they exist, and calibrates the
level. It knows nothing about WORLD, MLSA or mel-cepstra. The vocoder owns the
filter: a backend that generates its own excitation can also be handed one, and
that is the only place where filtering happens. Nothing forces a backend to
understand PCA coefficients, and `SourceHMMModel` gained no WORLD-specific code.

### What the learned source replaces

Only the **periodic** half of the excitation — the pulse train. The noise half
(aperiodicity) is untouched, which preserves two existing behaviours for free:

* **Unvoiced excitation is unchanged.** Every HMS backend forces `ap` to 1 on an
  unvoiced frame, i.e. pure noise, so the learned source is multiplied by zero
  there and the existing noise path is exactly what it was. The learned source is
  not asked to be a noise generator and does not have to be muted. For the
  `voice` backend there are no units in unvoiced frames anyway; for the
  frame-synchronous `residual` backend, units in unvoiced frames are
  inaudible for this reason (see limitations).
* **Loudness stays with the acoustic model.** `sp` and `ap` still decide how
  loud a frame is, as they did before.

Where the source does not cover the timeline — unvoiced gaps, the samples after
the last epoch of a run, the tail past the end of the frame grid — the backend's
own excitation is used, so nothing regresses relative to an ordinary render.

The transition is **continuous and frame-aligned**: the source layer builds a
per-sample weight from the sample-accurate support mask and smooths it with a
moving average one frame period long (`fade_weights`), so every boundary ramps
over `hop` samples instead of stepping. The ramp is computed from a cumulative
sum, so it is O(n) with no kernel and no scipy.

### Timing and alignment

The two branches are made to agree before anything is rendered:

| quantity | how agreement is enforced |
|---|---|
| sample rate | `source_model.spec.fs` must equal `model.spec.fs` exactly, else `ValueError` — HMS never resamples audio or a source prediction |
| frame period | same check on `frame_period` |
| frame count | the source branch is given the acoustic `plan`'s frame count, never its own |
| F0 | the render's own `parameters.f0` (Hz, `0` = unvoiced) is sliced per utterance and passed to `generate`, so pitch and voicing cannot disagree |
| duration / sample count | `render_length(n_frames, frame_period, fs)` — the same WORLD convention the backends use |
| sample layout | an utterance starting at frame `lo` is written at sample `lo * hop`, the mapping the vocoder itself uses, so no drift accumulates over a long score |

`plan()` records `(utterance, start_frame, stop_frame)` per score utterance as
internal state — the same pattern as its existing `_frame_units`, so the public
six-value return signature is unchanged — because Phase 2 is defined per
utterance and the frame grid is flat across the score.

A source model whose `voiced_threshold` or `f0_ref_hz` differs from the acoustic
model's is **reported** rather than rejected: the F0 is shared, so the two
branches derive voicing from the same track and the acoustic branch's decision
still reaches the vocoder. They cannot fight.

### Source gain

**Phase 2 predicts source shape, not amplitude.** Its generated units carry
`gains = 1.0`. Phase 3 therefore does not invent a gain model. What it does:

1. every generated unit is scaled to **unit RMS**. This is Phase 1's own storage
   convention — `SourceSequence.excitation` holds unit-RMS cycles and
   `SourceSequence.gains` holds the level — so for a Phase-1 analysis it is the
   identity, and for a generated one it removes the unmodelled amplitude the
   sampled PCA coefficients arrive with (measured spread on a fitted model: an
   order of magnitude, i.e. isolated spikes, a clipped filter, and a render
   several dB quiet after a backend's peak normalisation). Degenerate units
   — numerically zero, so carrying no shape either — are silenced rather than
   multiplied by `1/eps`;
2. `restore_gain=True` then applies the sequence's own `gains`, so a Phase-1
   analysis (or a future gain predictor) reaches the output through the existing
   representation with no new field;
3. the placed waveform is finally scaled to **unit RMS over the samples it
   covers** — exactly the calibration of the MLSA pulse train it replaces (that
   train is built to unit mean square at any F0 so the output level tracks
   `sp`). The measured RMS and the applied scalar are kept on
   `SourceExcitation` (`source_rms`, `applied_gain`) so the correction is
   inspectable, not hidden;
4. `source_gain` (default `1.0`) is a single deliberate scalar on top, exposed
   as `SynthesisConfig.source_gain`, `hms synth --source-gain`, and the
   `synthesis:` block of `parameters.yaml`.

Net effect: **loudness stays with the acoustic model.** Measured on the demo
corpus (32 s, same vocoder, only the excitation swapped), a broadband
whitened-residual-like source renders at **+7 % RMS** through `builtin` and
**−11 %** through `mlsa`, with the excitation's crest factor at ~9 against the
default pulse train's ~6. Without step 1 the same measurement gives −75 % and
a peak of 10 (clipping) instead of 2.2.

The honest limitation: equalising the *excitation* RMS does not equalise the
*output* level, because a learned source does not have a pulse train's flat
spectrum — a deliberately narrow-band source can still come out several dB
louder, and only a per-unit gain target in the source HMM would fix that. That
is future work, not something Phase 3 pretends to have.

### API

```python
from hms.core.model import HMSModel
from hms.core.synthesizer import Synthesizer, SynthesisConfig
from hms.source import SourceHMMModel

model = HMSModel.load("model")
source = SourceHMMModel.load("model-source")
score = labels.load("corpus/score.tsv")

# ordinary HMS synthesis -- unchanged, bit-identical to before Phase 3
plain = Synthesizer(model).synthesize(score)

# source-aware synthesis
result = Synthesizer(model, SynthesisConfig(vocoder="mlsa", source_gain=1.0)
                     ).synthesize(score, source_model=source)
result.source            # SourceExcitation: excitation, support, weights,
                         # coverage, voiced, n_units, source_rms, applied_gain
result.source.summary()  # one-line diagnostic
```

```bash
hms synth --model model --score score.tsv --out song.wav \
          --source-model model-source --vocoder mlsa [--source-gain 1.0]
```

`source_model=` follows the same convention as the existing `f0=`: it is
per-render data, not a voice setting, so it is a `synthesize` argument rather
than a `SynthesisConfig` field and it is not needed by anyone who does not train
a source model. `SynthesisResult.source` is an additive field that is `None`
for an ordinary render.

### Which backends are supported

| backend | source-aware synthesis | why |
|---|---|---|
| `builtin` | yes | pure-numpy: it builds its own pulse/noise excitation, so it can be handed one |
| `mlsa` | yes | same: fractional-position pulse train plus noise, mixed per bin |
| `native` (WORLD) | no | `Synthesis(f0, sp, ap)` takes no excitation |
| `pyworld` | no | same |

Support is an explicit capability, `Vocoder.supports_external_excitation`
(default `False`) plus `Vocoder.synthesize_with_excitation(...)`, which raises
`VocoderUnavailable` naming the backends that do support it. The synthesizer
turns a refusal into a `ValueError` before rendering anything rather than
silently dropping the caller's source model. A new backend only has to override
the flag and splice the waveform into whatever excitation it already builds.

### Tests

`hms/tests/test_source_synthesis.py` (32 tests) covers: the ordinary path
unchanged without a source model; both numpy backends rendering a tiny Phase-2
model; the learned source actually changing the audio; frame-count,
sample-rate and frame-period agreement; F0 identity between the branches;
rejection of a wrong-length / non-finite / negative F0 and of a render longer
than its frame grid; unit spans staying inside the utterance and never crossing
a voiced/unvoiced gap; an all-unvoiced score rendering **bit-identically** to
the ordinary path; the one-frame continuous transition; unit-RMS calibration;
amplitude staying in the same ballpark and tracking `source_gain`
monotonically; determinism; finite output of the expected sample count; refusal
by a backend that cannot filter an excitation; and an end-to-end
labels + F0 → acoustic model → source model → excitation → filter → waveform
render.

## Where the code lives

| path | contents |
|---|---|
| `hms/source/base.py` | `SourceFrame`, `SourceSequence`, `SourceModel`, `assign_frame_noise` |
| `hms/source/residual.py` | mel-cepstrum whitening / inverse filtering |
| `hms/source/cycles.py` | epoch tracking, cycle extraction, resampling, placement, unit support / frame coverage |
| `hms/source/voice.py` | `VoiceSourceModel`, `estimate_f0`, `smooth_f0` |
| `hms/source/generic.py` | `GenericResidualSourceModel` (pitch-free backend) |
| `hms/source/pca.py` | `SourcePCA` (fit / encode / decode / save / load / report) |
| `hms/source/trainer.py` | Phase-2 examples, alignment, PCA reuse, phone/context HMM/GMM training |
| `hms/source/hmm.py` | F0-conditioned prediction, MLPG, Phase-1 decoding, separate save/load |
| `hms/source/synthesis.py` | Phase-3 excitation rendering, coverage/weights, geometry checks |
| `hms/vocoder/base.py` | `Vocoder`, `render_length`, `blend_excitation`, the external-excitation capability |
| `tools/bench_source_pca.py` | the Phase-1 reconstruction experiment above |
| `hms/tests/test_source_cycles.py`, `test_source_pca.py`, `test_source_model.py` | Phase-1 cycle, PCA, API and reconstruction tests |
| `hms/tests/test_source_hmm.py` | Phase-2 alignment, context, F0, generation and serialization tests |
| `hms/tests/test_source_synthesis.py` | Phase-3 source-aware synthesis integration tests |
