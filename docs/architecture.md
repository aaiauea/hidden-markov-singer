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
compact feature vectors            [0] note-relative F0, [1:] mcep + ap bands
      │  hms.core.trainer           normalise, collect per phoneme, train HMMs
      ▼
hms.core.hmm + hms.core.gmm        left-to-right HMM, diagonal GMMs
      │
      ├── hms.core.duration         per-phoneme log-normal durations + allocation
      ├── hms.core.pitch            optional state-mean pitch statistics + vibrato
      ▼
hms.core.model                     HMSModel.save / load  (model.yaml + npz)
```

Synthesis keeps target musical F0 distinct from optional learned prosody:

```
score notes ─► target F0 ─────────────────────────────────────────────┐
score phones ─► hms.core.synthesizer.plan ─► HMM/GMM ─► MLPG ─► sp/ap ├─► WORLD
                                      └─ optional F0 deviation / vibrato ─► F0 ┘
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
| 0 | note-relative log-F0 in semitones (an optional learned-deviation feature; target F0 still comes from the score) | 1 |
| 1 … n_mcep | mel-cepstrum c0…c29 of the WORLD spectral envelope, sampled on 2·n_mcep mel bands | 30 |
| next n_band | mel-spaced aperiodicity bands | 5 |

* **Spectral envelope**: the filterbank band densities of `sp` are normalised by
  the filter weight sums (otherwise low bands come out ~10 dB high), log'd, and
  projected with an orthonormal DCT-II; the first `n_mcep` coefficients are
  kept.  `decode` inverts that projection, interpolating the band values back
  onto the spectrum, and the round-trip error measured in band-density space is
  ~0.2 dB for typical frames, i.e. the truncation is the only loss.

  The envelope is sampled on `mel_band_count(n_mcep) = 2 * n_mcep` mel bands
  (`FeatureSpec.n_mcep` on the model side; the analysis grid is part of the
  feature definition, see `hms.core.features.mel_band_count`).  The sampling
  grid matters more than it looks: each band value is an *average* over a wide
  triangle and the decoder interpolates between band centres, so structure
  narrower than the band spacing is flattened and a formant that falls between
  two knots can come back as a peak *on* a knot.  With one band per coefficient
  (the historical `n_mcep + 2` grid) the spacing between 1.8 and 3.2 kHz was
  307-421 Hz and a knot sat exactly at 2365 Hz, which is where every synthesised
  vowel used to show a spurious ~2.4 kHz peak.  Doubling the grid costs nothing
  in model size (the model still stores `n_mcep` coefficients) and cuts the
  2-4 kHz reconstruction error by roughly a third; beyond 2x the band spacing
  drops below the ~130 mel smoothing width of the truncated DCT and the error
  stops improving.
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

The rule is **target musical F0 first; learned prosody is optional**:

*Target F0*: the score supplies the MIDI note for each frame. In the default
`f0_source: score` mode, that note is used directly as target F0 (subject to the
phoneme voicing mask and configured F0 range). A trained pitch model is not
required for this path.

*Optional acoustic deviation*: training can encode
`relative[t] = 12·log2(f0[t] / note_hz[t])`, with unvoiced gaps interpolated for
finite acoustic features. The acoustic HMM can then learn this singer's
note-relative movement. `f0_source: acoustic` adds the MLPG trajectory of this
feature to the score target; `f0_source: state_means` instead uses the separate
per-phoneme/state pitch statistics stored in `PitchModel`. These are optional
ways to add deviations, not autopitch sources or prerequisites. Under-threshold
phones have no separate state-mean pitch statistics; that optional mode safely
uses zero deviation for them rather than inventing mandatory pitch prediction.

*Optional vibrato*: `Vibrato` is an explicit component (rate, depth, delay,
attack, randomness, waveform) rather than part of the HMM, because MLPG smooths
away exactly the fast oscillation that makes vibrato sound alive. Its defaults
can be *measured* from the training data (`estimate_vibrato` finds the dominant
3–9 Hz component of the longest sustained note). Vibrato is disabled by
default and can be added independently of the target note.

All sources are followed by clamping into the analysed F0 range so that
transposing a score beyond the training range degrades audibly but does not
silently drop the melody.

## 6. Parameter generation (`hms/core/generation.py`)

Classic MLPG: build the window matrix `W` that maps a static trajectory to the
static + delta feature sequence, then solve

```
(Wᵀ Σ⁻¹ W) x = Wᵀ Σ⁻¹ μ
```

`Σ` is diagonal, so `Wᵀ Σ⁻¹ W` is banded with bandwidth `2·window·(streams−1)`;
the solver is a banded Cholesky written directly in `numpy` (no scipy), checked
against a dense solve in the tests to ~1e-15.

* `variance_scale` scales the *dynamic* (delta) variances: >1 trusts the
  deltas less, so the trajectory follows the per-frame means more literally
  (livelier, more detail), while <1 strengthens them and flattens the
  trajectory.  Scaling *every* precision by the same factor would leave the
  solution unchanged (the normal equations are homogeneous), which is why the
  knob has to act on the delta streams only.
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

`min_phoneme_frames` (default 20) is the guard rail: a phoneme below it does
not get a dedicated HMM. Class backoff HMMs are trained directly from the raw
feature sequences of *all* known phones in that class, including under-threshold
phones; each frame contributes once, so a common phone naturally contributes
more than a rare one. Backoff GMM components are never formed by averaging
component indices from separately trained phone models. The parameter report
shows dedicated and backoff budgets separately.

## 8. Synthesis pipeline (`hms/core/synthesizer.py`)

1. `plan` — score segments → phoneme durations → per-state frame counts →
   `(phone, state)` per frame; gaps become silence; unknown phonemes are
   reported and routed to the backoff model.
2. `frame_statistics` — per-state GMM means/variances stacked into per-frame
   statistics (dominant component by default, or the mixture marginal).
3. `mlpg` — the trajectory.
4. `denormalize` → static features; voicing from HMM/phoneme statistics;
   target F0 from score notes, with optional acoustic/state-mean deviation and
   optional vibrato; clamp; decode features to `(f0, sp, ap)`.
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
