# HMS architecture

This document explains what each part of HMS does and why it is built that way.
It mirrors the module layout: read it top to bottom and you have read the
system.

```
label file + WAVs
      │
      │  hms.core.labels            score/label parsing, frame alignment
      ▼
hms.vocoder.*                       WORLD (native | pyworld | builtin)
      │                             f0 (Hz), sp (power), ap (aperiodicity)
      │  hms.core.features          FeatureSpec.encode / decode
      ▼
compact feature vectors            [0] F0 vs. the sung note, [1:] mcep + ap bands
      │  hms.core.trainer           normalise, collect per phoneme, train HMMs
      ▼
hms.core.hmm + hms.core.gmm        left-to-right HMM, diagonal GMMs
      │
      ├── hms.core.duration         per phoneme log-normal durations + allocation
      ├── hms.core.pitch            note-relative pitch statistics + vibrato
      ▼
hms.core.model                     HMSModel.save / load  (model.yaml + npz)
```

Synthesis runs the same picture backwards:

```
score ─► hms.core.synthesizer.plan ─► (phoneme, state) per frame
      ─► per-state GMM statistics ─► hms.core.generation.mlpg ─► trajectories
      ─► note + deviation + vibrato ─► F0;  voicing mask ─► hms.vocoder.synthesize
```

## Why HMM/GMM in 2020s terms

An HMM with Gaussian mixtures is a *small, interpretable, trainable-from-almost-
nothing* sequence model. For singing synthesis the interesting decisions are
mostly prosodic (which note, how long, voiced or not), and the acoustic
realisation is smooth and slowly varying. A left-to-right HMM with 2–5 states
per phoneme captures exactly that, and its parameter count is proportional to
phonemes × states × components × features — measurable, tunable, and small.

The alternative (a neural acoustic model) is deliberately out of scope.

## 1. Vocoder layer (`hms/vocoder`)

`Vocoder` is an abstract class with three things: `analyze()` (waveform →
`f0`, `sp`, `ap`), `synthesize()` (`f0`, `sp`, `ap` → waveform) and the frame
geometry (`fft_size`, `fft_size_for(fs)`, `n_bins`). Everything above it is
vocoder-agnostic.

* `NativeWorldVocoder` — the real WORLD (DIO + StoneMask, CheapTrick, D4C,
  Synthesis) through a small C ABI (`tools/world_native/hms_world_capi.cpp`)
  loaded with `ctypes`. The shared library is built by `tools/build_world.sh`.
* `PyWorldVocoder` — uses `pyworld` if it is installed.
* `BuiltinVocoder` — pure numpy: normalised-autocorrelation F0 with a voicing
  decision from both the autocorrelation peak *and* spectral flatness,
  cepstrally liftered envelope, pitch-synchronous aperiodicity, and
  overlap-add synthesis with a zero-phase magnitude response.

Two details worth knowing:

1. **WORLD's FFT size follows the sample rate** (2048 at 44.1 kHz, 1024 at
   22.05 kHz). `fft_size_for(fs)` is the only correct way to ask; the native
   backend refuses an explicitly pinned size that WORLD cannot honour rather
   than silently writing the wrong number of bins per frame.
2. **D4C degenerates on noise-free signals** (it reports aperiodicity ≈ 1.0 for
   every voiced frame, i.e. "all of it is noise"). `analyze()` detects this and
   substitutes a pitch-synchronous harmonicity estimate, emitting a
   `RuntimeWarning`. Synthesised demo material hits this; real recordings
   normally do not.

## 2. Features (`hms/core/features.py`)

One `AcousticFrameSequence` (f0/sp/ap) is turned into a static feature vector
per frame plus dynamic (delta) features. `FeatureSpec` owns the geometry and the
`encode`/`decode` pair, so training and synthesis share one definition.

| slot | content | dim (default) |
|---|---|---|
| 0 | log-F0 in semitones **relative to the sung note** | 1 |
| 1 … n_mcep | mel-cepstrum c0…c29 of the WORLD spectral envelope | 30 |
| next n_band | mel-spaced aperiodicity bands | 5 |

* **Spectral envelope**: the filterbank band densities of `sp` are normalised by
  the filter weight sums (otherwise low bands come out ~10 dB high), log'd, and
  projected with an orthonormal DCT-II; the first `n_mcep` coefficients are
  kept. `decode` inverts this exactly, and the round-trip error measured in
  band-density space is ~0.2 dB for typical frames, i.e. the truncation is the
  only loss.
* **Aperiodicity**: 5 mel bands, interpolated back onto the spectrum by
  `decode`. 5 bands is enough because the ear is insensitive to the fine
  structure of aperiodicity; it is also 5 parameters instead of 1025.
* **Dynamic features**: `delta_coeffs(window=2)` (HTS-style regression), applied
  to the whole static vector. Deltas are effectively mandatory — MLPG needs
  them to generate a smooth trajectory — and delta-deltas are off by default
  because they roughly double the acoustic parameter count for modest gains
  unless the corpus is larger.

Unvoiced frames store 0.0 in slot 0 rather than NaN (Gaussians cannot be fitted
around NaN). Voicing is therefore carried separately, by the per-state voicing
probabilities, and is imposed at synthesis time.

## 3. Statistical core (`hms/core/gmm.py`, `hmm.py`)

**GMM** (`DiagGMM`): diagonal or *tied* (one shared covariance per state)
Gaussians, k-means++ initialisation, weighted EM, component pruning, and a
variance floor expressed relative to the data variance. `n_components=1` gives a
plain Gaussian, which is what most consonant states use. Data-efficiency comes
from three places: the variance floor (no runaway peaks on short segments),
pruning (useless components collapse), and tying (a state with 50 frames can
afford one covariance, not three).

**HMM** (`LeftToRightHMM`): states in a strict left-to-right chain with a
self-loop per state and an optional skip transition (off by default). Each state
carries its GMM, a log-duration mean/variance and a voicing probability.

* *Training* is embedded Viterbi (segmental k-means): segment each labelled
  phoneme occurrence with the current models, re-estimate the GMMs from the
  assigned frames, iterate. Baum-Welch is available via
  `training_method: baum_welch` and is a textbook forward-backward pass; it is
  slower and slightly less robust on tiny corpora.
* *Segmentation* (`segment`) returns a monotone state path and enforces a
  minimum occupancy (a quarter of an equal share) so that a long vowel cannot
  starve its own onset/transition states during training, which would otherwise
  produce a duration model that deletes those states at synthesis time.
* *Durations* are the per-state log of the run lengths observed in training. The
  self-loop probability is derived from them (`p = 1 − 1/mean duration`), so the
  transition matrix is never a free parameter that could drift out of sync.
* *Voicing* per state is the fraction of that state's frames whose F0 was above
  the voicing threshold.

**Alignment** lives in two places by design: `labels.py` maps label times to
frame indices, and `hmm.segment` maps frames to states. There is no separate
alignment module because there is no separate alignment problem.

## 4. Duration (`hms/core/duration.py`)

Two jobs:

1. `DurationModel.predict` — per phoneme log-normal duration statistics, used
   when the score carries no timing (`--duration-mode model`). Sampling from it
   (`variance_scale`) is how you get natural timing variation.
2. `DurationModel.allocate` — split a *known* duration across the phoneme's HMM
   states in proportion to their learned mean durations, with every state
   guaranteed at least one frame. This is what makes the state sequence
   duration-aware: a long note stretches the steady state, not the onset.

## 5. Pitch (`hms/core/pitch.py`)

The rule is **note first, statistics second**:

*Training*: `relative[t] = 12·log2(f0[t] / note_hz[t])`, with unvoiced gaps
linearly interpolated (they carry no pitch information; leaving raw zeros there
would teach the model nonsense). The acoustic HMM therefore learns *this
singer's habits around a note* — scoops, drift, how a phrase is approached —
independently of which note is sung.

*Synthesis*: `f0 = note + MLPG trajectory of feature 0 + optional vibrato`,
followed by clamping into the analysed range so that transposing a score beyond
the training range degrades audibly but does not silently drop the melody.

Vibrato is an explicit component (`Vibrato`: rate, depth, delay, attack,
randomness, waveform) rather than part of the HMM, because MLPG smooths away
exactly the fast oscillation that makes vibrato sound alive. Its defaults can be
*measured* from the training data (`estimate_vibrato` finds the dominant 3–9 Hz
component of the longest sustained note).

## 6. Parameter generation (`hms/core/generation.py`)

Classic MLPG: build the window matrix `W` that maps a static trajectory to the
static + delta feature sequence, then solve

```
(Wᵀ Σ⁻¹ W) x = Wᵀ Σ⁻¹ μ
```

`Σ` is diagonal, so `Wᵀ Σ⁻¹ W` is banded with bandwidth `2·window·(streams−1)`;
the solver is a banded Cholesky written directly in `numpy` (no scipy), checked
against a dense solve in the tests to ~1e-15.

* `variance_scale` relaxes the *dynamic* constraints (>1 = smoother, <1 =
  follows the deltas more literally). Scaling every precision by the same factor
  would leave the solution unchanged, which is why the knob acts on the delta
  streams only.
* `smooth=False` skips MLPG entirely and returns the state means (useful when
  debugging the acoustic model without the dynamics).

## 7. Training pipeline (`hms/core/trainer.py`)

```
read label file → analyse every utterance (WORLD) → per-frame phoneme/note labels
→ note-relative F0 → static features → deltas
→ per-dimension mean/std normalisation over the whole corpus
→ per phoneme: collect frames, train a LeftToRightHMM
→ pooled per phoneme-class backoff models for phonemes with too little data
→ duration model (per phoneme) → pitch model (per phoneme/state) → voicing
→ HMSModel (spec, hmms, backoff, normalisation, stats, metadata)
```

`min_phoneme_frames` (default 20) is the guard rail: a phoneme that appears less
than that goes to the backoff model instead of being fitted with an unreliable
Gaussian, and `hms inspect-model` reports the parameter budget so you can see
the effect.

## 8. Synthesis pipeline (`hms/core/synthesizer.py`)

1. `plan` — score segments → phoneme durations → per-state frame counts →
   `(phone, state)` per frame; gaps become silence; unknown phonemes are
   reported and routed to the backoff model.
2. `frame_statistics` — per-state GMM means/variances stacked into per-frame
   statistics (dominant component by default, or the mixture marginal).
3. `mlpg` — the trajectory.
4. `denormalize` → static features; voicing from the state/phone statistics;
   F0 from note + trajectory + vibrato; clamp; decode features to
   `(f0, sp, ap)`.
5. `vocoder.synthesize` → waveform (written by `wavio.write_wav`, 16-bit by
   default).

`SynthesisResult` keeps everything: audio, WORLD parameters, per-frame phones,
states, notes, `f0_semitones` and diagnostics. `hms synth --trace` writes the
same information as a text table, which is the fastest way to see what the model
decided.

## 9. Model format

```
model.yaml      human readable: format version, feature spec, normalisation
                (offset/scale), phoneme set, duration model, pitch model
                (including vibrato), HMM index, parameter budget, metadata
hmm.npz         arrays: GMM weights/means/variances, self-loops, durations,
                voicing probabilities
backoff.npz     the same for the pooled per-class models
```

YAML for anything a human might want to read or tweak, `.npz` for the arrays.
`HMSModel.save/load` is the only serialisation code in the project, and the
loader validates the format version.

## 10. Design trade-offs (what is deliberately missing)

| decision | why |
|---|---|
| per-phoneme HMMs, no state tying across phonemes | the inventory is small; backoff models cover rare phonemes with one pooled model per class |
| log-normal durations, no duration HMM | the score already carries the timing; the model only fills gaps |
| 5 aperiodicity bands | the fine structure of `ap` is perceptually unimportant compared to 1025 extra parameters |
| vibrato outside the HMM | MLPG would smooth it away; keeping it explicit makes it controllable |
| no spectral postfilter (GV etc.) | MLPG already yields slightly over-smoothed spectra, and a postfilter is a tuning surface better added later, deliberately |
| one speaker per model | adaptation/multi-speaker would complicate every stage; nothing in the format prevents adding it |

## 11. Extension points

* **New phonemes** — edit `phonemes.yaml` (or point `--phonemes` at your own).
* **New vocoder** — subclass `hms.vocoder.base.Vocoder` and register it in
  `hms/vocoder/__init__.py`.
* **Different features** — `FeatureSpec` is data; the trainer, model file and
  synthesizer read their geometry from it.
* **Different dynamics** — `window_matrix`/`mlpg` take the delta window as a
  parameter, so a different regression or a new dynamic feature only needs a
  matching `stream_taps`.
* **Expression** — `transpose`, `tempo`, `variance_scale`, `pitch_variation`
  and `Vibrato` are all synthesis-time knobs; adding another one means adding a
  field to `SynthesisConfig` and applying it in one place.
