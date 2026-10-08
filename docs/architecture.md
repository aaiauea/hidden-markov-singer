# HMS architecture

This document explains what each part of HMS does and why it is built that way.
It mirrors the module layout: read it top to bottom and you have read the
system.

```
label file + WAVs
      │
      │  hms.core.labels            score/label parsing, frame alignment
      ▼
hms.vocoder.*                       WORLD (native | pyworld)
      │                             pure numpy (builtin | mlsa)
      │                             f0 (Hz), sp (power), ap (aperiodicity)
      │  hms.core.features          FeatureSpec.encode / decode
      ▼
compact feature vectors            [0] note-relative F0, [1:] mcep + ap bands
      │  hms.core.trainer           normalise, collect per phoneme, train HMMs
      │                             (+ optional tiers: phone contexts, and the
      │                              same units per pitch bin of the scored note)
      ▼
hms.core.hmm + hms.core.gmm        left-to-right HMM, diagonal GMMs
      │
      ├── hms.core.duration         per-phoneme log-normal durations + allocation
      ├── hms.core.pitch            optional state-mean pitch statistics + vibrato
      ├── hms.core.pitch_condition  optional deterministic pitch bins (selection)
      ▼
hms.core.model                     HMSModel.save / load  (model.yaml + npz)
```

Synthesis keeps target musical F0 distinct from optional learned prosody, and an
external trajectory overrides both:

```
score notes ─► target F0 ─────────────────────────────────────────────┐
score phones ─► hms.core.synthesizer.plan ─► HMM/GMM ─► MLPG ─► (GV) ─► sp/ap ├─► WORLD
   │                          └─ optional F0 deviation / vibrato ─► F0 ┘
   │                          └─ external F0 (synthesize(f0=…)) ─► F0 ─┘
   │                             (authoritative: replaces, not added)
   └─► optional pitch bin (score note, not generated F0) ─► *which* HMM/GMM
                                                              is used
```

Phase 2 adds a **separate** source-parameter branch over the Phase-1 source
representation. It is not called by the current synthesis pipeline:

```
labelled phones/context + explicit frame F0
                 └─► source HMM/GMM ─► MLPG ─► PCA coefficients
                                                └─► Phase-1 PCA decoder ─► source units
```

The source tier has its own singer-specific files and strict frame/F0 checks.
It does not alter `HMSModel`, acoustic HMMs, `Synthesizer`, `FeatureSpec`'s
acoustic dimensions, or vocoder backends.

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
* `MLSAVocoder` — pure numpy, opt-in (`vocoder="mlsa"`): the same shared
  analysis, but synthesis goes through the MLSA (mel log spectrum
  approximation) filter, `H(z) = exp(sum_m a_m z~^{-m})` on the mel-warped
  axis.  The mel-cepstral coefficients are the ones the feature layer already
  models (the DCT-II coefficients, converted to the warped Fourier basis), the
  transfer function is evaluated exactly on the FFT grid and realized as a
  per-frame FIR with overlap-add, and the excitation is a fractional-position
  pulse train plus noise mixed *per frequency bin* by the aperiodicity.  It is
  a synthesizer, not a new parameterisation: duration, `fft_size_for` and
  analysis conventions are identical to `BuiltinVocoder`.  Full formulation,
  benchmarks and limitations: [mlsa.md](mlsa.md).

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

* **Input validation with line numbers.** `labels.py` reports every problem
  instead of failing downstream or staying silent: unusable rows are dropped
  with a line-numbered diagnostic (missing columns, empty utterance id or
  phoneme, non-numeric or non-finite times, negative times, offset before
  onset, zero-duration segments, non-numeric notes, notes outside the MIDI
  range 0-127) and unrecognised extra columns (everything after the note
  column must be `key=value` context) are ignored with a diagnostic. Overlaps
  *and* gaps between consecutive segments of an utterance are reported per
  utterance; both stay tolerated (a gap is silence in synthesis and a carried
  note in training) but are no longer silent. Legitimate labels — contiguous
  rows, boundary silences, unnoted segments, fractional MIDI detuning,
  `key=value` context — produce no diagnostics.

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

This module is everything about *what pitch is produced*. The optional
pitch-conditioned acoustic models (§7) are the opposite direction and never
touch it: they use the scored note to choose *which spectral HMM* a phone is
rendered from, leave dimension 0 (note-relative F0) and the whole F0 path
exactly as they are, and introduce no second pitch predictor. Enabling both
together means the note still comes from the score while the envelope comes from
the pitch region the corpus observed that note in.

### Out-of-training-range F0

`f0_floor`-`f0_ceil` bound the F0 **analyser** (DIO/Harvest) that produced the
training features, not what the synthesizer may be asked for. They describe
where the model has observations; they are not a playback limit. So an F0
outside that range is handled by separating what the model knows from what the
caller asked for:

* **The pitch is never adjusted.** The requested F0 — score note, or the
  external trajectory — reaches `FeatureSpec.decode` and the vocoder verbatim.
  `decode` unvoices a frame only when it has no pitch at all (`NaN`, or below
  `voiced_threshold`), so MIDI 96 (~2.1 kHz) is synthesised at 2.1 kHz and
  MIDI 24 (~32.7 Hz) at 32.7 Hz.
* **The spectral parameters are the nearest trained ones.** HMS's acoustic
  model is not indexed by F0: a state's statistics are a mel-cepstrum envelope
  and a band aperiodicity, i.e. a description of the vocal tract, learned from
  whatever frames happened to sing that phone. There is no second envelope at
  2.1 kHz to interpolate towards, so the boundary region it does have is
  reused unchanged — Sinsy's "use the closest observed F0's acoustic
  parameters", reduced to what this architecture can express, with no neural
  model and no explicit formant parameters. What you hear is a trained vowel
  excited at a pitch the model never heard it at.
* **It is reported, not clamped.** `plan()` names out-of-range notes before
  rendering, and `Synthesizer._out_of_range_pitch_diagnostics` adds one
  message per render with the requested frequencies, e.g. *"100 of 100 voiced
  frame(s) request F0 outside the model's trained F0 range 71-800 Hz (up to
  2093.0 Hz); the boundary acoustic statistics for those frames are reused
  unchanged and the requested F0 is preserved"*.

The one case that *is* normalised is a note that is not a MIDI note at all
(outside 0-127; reachable only through `transpose`, a `default_note` override
or a programmatic score, since parsed labels are already checked). Such a note
names no musical pitch, so it is rendered at the nearest valid MIDI note and
reported — which also keeps every rendered frequency below Nyquist. Valid MIDI
numbers are never moved, however far their frequency is from the trained
range. Training warns separately about *label* notes outside the analysis
range, because there the extractor really cannot produce a matching F0 and the
note-relative pitch learned on those segments is off.

*External F0 override*: `Synthesizer.synthesize(score, f0=…)` (and the CLI's
`hms synth --f0-file`) accepts a frame-level F0 trajectory, in Hz, with one
value per *synthesis frame* of the whole score and `0.0` marking unvoiced
frames — the same convention `FeatureSpec.encode` and WORLD use. The insert
point is deliberately narrow: `external_f0_to_semitones` converts the array
into the representation `PitchModel.generate` returns (semitones re.
`f0_ref_hz`, `NaN` where unvoiced) and the synthesizer then treats it exactly
like a generated contour — same `FeatureSpec.decode`, same out-of-range
reporting, same vocoder, and the same rule that the values are used exactly as
supplied. Because it replaces the generated trajectory rather than feeding into
it, an external F0 automatically receives no score pitch, no learned deviation,
no generated vibrato and no `pitch_smoothing`, and nothing about `sp`/`ap`,
timing or the vocoder interface changes.

The trajectory is never resampled: it must be one value per frame, and anything
else (wrong length or shape, empty input, non-numeric values, `NaN`/`inf`,
negative frequencies) is a `ValueError` naming the offending frame index. That
validation lives in one place, `external_f0_to_semitones`, precisely so the
vocoder is never handed a trajectory it can only fail on later.

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

### Experimental Global Variance (`hms/core/gv.py`)

MLPG's maximum-likelihood trajectory often has too little across-utterance
variation: the static means and delta constraints smooth rapid changes toward
their average. GV is an optional *second optimization* on **static** MLPG output,
not on the stacked delta streams and not a simple gain applied to the output.
`FeatureSpec.stream_sizes = (static_dim,) * n_streams`; `stack_streams` takes
feature columns in the order [static, delta, delta-delta] and puts complete
streams down the row axis. `mlpg` returns `(T, static_dim)` and the optional GV
optimizer runs on that array **before** `model.denormalize` and `spec.decode`.
The F0 slot is note-relative in training space, so in score-F0 mode the score's
requested pitch remains authoritative.

For feature `d`, training estimates the target `v*_d` as the arithmetic mean of
utterance-level population variances of the **normalized static** frames. Each
utterance's variance is measured about *that utterance's own mean*, not the
corpus mean; single-frame utterances are excluded. The cached training features
are read one utterance at a time, with only `[:, :static_dim]` used. This gives
one target per static coefficient (no GV for a derivative stream and no extra
corpus-sized training arrays). Silence/unvoiced frames still contribute; this
first experiment has no per-phone, voiced-only or duration-conditioned target.

Starting at `c0 = mlpg(μ, Σ, ...)`, for each feature separately the optimizer
minimizes the following **approximate GV objective** over the whole trajectory:

```
v(c)      = (1/T) Σ_t (c_t − mean(c))²
∇_c v(c)  = (2/T) (c − mean(c))
q         = max(v*, v(c0), 1e-6)
E(c)      = (1/(2T)) [W(c − c0)]ᵀ P [W(c − c0)]
            + (gv_weight/2) [(v(c) − v*)/q]²
```

`W` has the *same* clipped delta/delta-delta windows as MLPG (including
utterance boundaries), and `P` has its variance floor and `variance_scale` on
the dynamic streams. When `smooth=True`, `c0` solves the MLPG normal equations,
so the first term equals the increase in acoustic negative log-likelihood up to
solve rounding; with `smooth=False`, it is a local quadratic trust anchor around
the un-smoothed static means instead. Scaling by `T` makes the weight independent
of sequence length and `q` keeps flat features well-conditioned, but `q` is
**not** a learned GV uncertainty. The gradient combines `Wᵀ P W(c − c0)/T`
with `gv_weight · ((v(c) − v*)/q²) · 2(c − mean(c))/T`. Iterations take diagonally
preconditioned steps, bounded in feature units and accepted by a per-feature
backtracking line search only if the objective decreases. This applies the
model's local static/dynamic constraints throughout; it is *not* a post-hoc
variance multiplier and requires no extra Cholesky solve or dense W matrix.

`SynthesisConfig.gv_enabled=False` is the default (also `synthesis.gv_enabled`
in YAML or `hms synth --gv` / `--no-gv`). `gv_weight=1.0` and `gv_iterations=20`
are adjustable (CLI: `--gv-weight`, `--gv-iterations`; zero means no updates).
The optimized MLPG band assembly, Cholesky and solve are untouched. If the model
has no stored GV section, disabled synthesis still loads/runs unchanged; an
explicitly enabled GV render raises a clear error instead of making up targets.
Constant features have no variance gradient and are left constant; T < 2 and
non-finite/invalid targets are handled safely. Only a mean per-feature target
is learned, not the distribution/covariance of GV found in classic GV-MLPG,
and the weight is a tuning choice rather than a trained parameter. Listen and
compare on your own corpus; more variance need not mean better audio.

## 7. Training pipeline (`hms/core/trainer.py`)

```
read label file → analyse every utterance (WORLD) → per-frame phoneme/note labels
→ note-relative F0 → static features → deltas
→ per-dimension mean/std normalisation over the whole corpus
→ optional static GV target statistics from normalized utterance-level frames
→ per phoneme: collect frames, train a LeftToRightHMM
→ pooled per phoneme-class backoff models for phonemes with too little data
→ optional (context.enabled): sparse HMMs for the observed phone contexts,
  plus an optional pooled global backoff
→ optional (pitch_conditioning.enabled): the same units re-bucketed by pitch
  bin of the scored note
→ duration model (per phoneme) → pitch model (per phoneme/state) → voicing
→ HMSModel (spec, hmms, backoff, contexts, pitch models, normalisation, GV, stats)
```

`min_phoneme_frames` (default 20) is the guard rail: a phoneme below it does
not get a dedicated HMM. Class backoff HMMs are trained directly from the raw
feature sequences of *all* known phones in that class, including under-threshold
phones; each frame contributes once, so a common phone naturally contributes
more than a rare one. Backoff GMM components are never formed by averaging
component indices from separately trained phone models. The parameter report
shows dedicated, context, pitch-conditioned, class-backoff and global-backoff
budgets separately.

Both optional tiers are separate passes over the same disk-backed cache: one
utterance is memory-mapped at a time and every bucket keeps *views* into it
rather than copies, so neither splitting a corpus by phone context nor by pitch
bin puts the corpus in RAM. The caches are released before the duration and
pitch models reopen them.

### Optional sparse phone contexts (`context.enabled: true`)

Off by default; `hms train --context` (or `context.enabled: true` in
`parameters.yaml`) adds a *sparse* context tier on top of the phoneme models:

* **Observed contexts only.** For every labelled segment the trainer records
  its exact `(pre_phone, curr_phone, future_phone)` triphone and — with
  `context.partial: true` — the one-sided diphone contexts `(pre, curr, _)`
  and `(_, curr, post)`. There is no full triphone inventory: a context is a
  candidate only if it occurs in the corpus. Utterance boundaries use the
  inventory's existing `sil` symbol as the neighbour; there are no BOS/EOS
  tokens.
* **Thresholds and a cap.** A context earns its own HMM only with at least
  `context.min_frames` pooled frames and `context.min_occurrences`
  occurrences; beyond the `context.max_models` cap the best-supported contexts
  win. Selection sorts by (frames, occurrences, key), so the sparse set is
  deterministic for a given corpus.
* **Trained from pooled raw sequences.** Exactly like the class backoffs, each
  context HMM is fitted directly to the feature sequences of its occurrences —
  never by averaging separately trained phone models. The state/component
  budget follows the *current* phone's definition, so a context of a vowel
  spends like a vowel. Context seeds are offset past the backoffs', so
  enabling the feature does not perturb the other tiers.
* **Optional global backoff.** `context.global_backoff: true` additionally
  trains one small pooled HMM over every frame, as the last safety net for
  phones nothing else covers.
* **Zero models is said out loud.** When context modelling is enabled but no
  observed context clears the thresholds (or the cap is 0), training emits an
  explicit diagnostic that 0 context models were created and that synthesis
  will rely on the normal phone/backoff hierarchy; context HMMs are never
  force-created on insufficient data.

Resolution at synthesis time (`HMSModel.resolve_unit`) follows a fixed
hierarchy: exact triphone → best-supported one-sided diphone (ties favour the
left context) → dedicated current-phone HMM → phone-class backoff → optional
global backoff. The resolved unit drives state allocation, acoustic statistics
and voicing; the `plan()` API keeps its six return values (the per-frame unit
choice travels alongside, internally). Score notes remain the base F0 path;
learned pitch deviations remain optional exactly as in the context-free case.
With contexts disabled every code path is bit-identical to the classic
pipeline, and models stay format-compatible (contexts are an additive format-3
payload; format-2 models load with an empty context tier).

When pitch conditioning is also enabled, three conditioned rungs are tried
*before* this hierarchy (next subsection); with conditioning disabled
`resolve_unit` ignores its optional `pitch_bin` argument and behaves exactly as
above.

### Optional pitch-conditioned acoustic models
(`pitch_conditioning.enabled: true`, experimental)

Off by default; `hms train --pitch-conditioning [--pitch-bin-size N]` (or the
`pitch_conditioning:` block in `parameters.yaml`) adds a tier of acoustic HMMs
for *(unit, pitch bin)* pairs. The motivation is that a pooled phone model mixes
observations from every pitch the corpus contains, so the envelope it emits for
`a` is an average across the voice's range. This feature lets the corpus say how
`a` looked *in that pitch region*, without changing anything about how pitch is
generated.

**The condition (`hms/core/pitch_condition.py`).** A bin is
`floor(MIDI note / bin_size)` with `bin_size` in semitones (default 6, i.e. a
tritone — two bins per octave). The computation is exact integer arithmetic:
the note is quantised to cents (`round(note * 100)`) before the division, so
`pitch_bin(66.0) == 11` for every representation of that note, equal notes
always share a bin, and bin edges never depend on floating-point rounding.
MIDI 0-127 maps to bins `0 .. n_pitch_bins(bin_size) - 1` (0-21 at the default
width); `bin_note_bounds` recovers each bin's note range for the parameter
report and for `model.yaml`. A value that is not a real number (including a
numeric string) has no condition rather than being coerced, and `bin_size` is
validated as a whole number of semitones in 1-128 — a width wider than the MIDI
range would silently switch the feature off, so the configuration says so
instead.

**Which frames carry a condition.** The condition is the *scored note* —
`Segment.note` — and one function, `segment_pitch_bin`, is used at training and
synthesis time alike:

| segment | condition |
| --- | --- |
| note present, phone is not the inventory silence | that note's bin |
| note present, phone *is* the silence symbol | **none** |
| no note (`None`, a rest or an unlabelled gap) | **none** |
| unvoiced frames inside a noted segment | the segment's bin |

So a phone keeps a single condition across a note even where the frames are
unvoiced or the pitch wobbles, silence is never pitch-conditioned, and
`default_note` is never substituted for a missing note: `effective_note`
distinguishes "no note" from "a note that happens to be the default", which a
caller cannot do after the fact. Frames without a condition train and synthesise
exactly as before. Unvoiced frames keep the note's condition deliberately — the
condition describes the musical context, not an instantaneous measurement.

**Synthesis never re-derives the condition from generated audio.** `plan()`
computes the bin from the note it was *given* — the score note plus
`--transpose`, clipped into MIDI range — via `segment_pitch_bin`, before any F0
exists. With an external F0 trajectory the same scored note is used: the
trajectory is honoured for pitch, and it is not treated as a second source of
conditioning. There is no F0 estimator, no pitch regression and no new pitch
predictor in this path.

**Buckets are structured keys, not phonemes.** A bucket key is
`(unit, bin)` where `unit` is a phone symbol *or* an existing context key
(`pre^curr^post`, one-sided diphones included) and `bin` is an `int`. The
phoneme inventory, the duration model, the note-conditioned pitch model
(`hms/core/pitch.py`) and the feature normalisation are untouched; nothing like
`a@60` is ever created, and a loaded model's `pitch_index` records `kind`,
`unit`, `curr`, `pitch_bin`, `note_min`, `note_max` and support counts per
bucket.

**Training.** One extra pass after the context tier, over the same disk-backed
cache (`_collect_cached_pitch_data` maps one utterance at a time and keeps
views). For each labelled frame with a condition the pass accumulates the frame
into its phone's bucket and, for every context unit that *earned* its own HMM in
this run, into that context's bucket — contexts that fell short of the support
thresholds are not conditioned either, because their bins would hold even less
data. `select_pitch_models` then applies the thresholds of the tier a bucket
conditions: `min_phoneme_frames` for a phone bucket, `context_min_frames` and
`context_min_occurrences` for a context bucket. Because a bin holds a *subset*
of its unit's frames, earning a conditioned model is never easier than earning
the unconditioned one; a bucket that falls short is simply not created.
Selection sorts by (frames desc, occurrences desc, unit, bin), so the sparse set
is deterministic. `train_pitch_models` reuses `LeftToRightHMM.train` on the
bucket's pooled raw sequences — the same Baum-Welch, the same budget taken from
the *current* phone's definition, no duplication — with seeds offset past the
context block (`seed + 10_000 * (len(backoff) + 1)`) so enabling the feature
does not perturb any other tier. When conditioning is enabled but no bucket
clears the thresholds, training says so explicitly (0 pitch-conditioned models,
synthesis stays on the ordinary hierarchy) instead of force-creating models.

**Resolution.** `resolve_unit(pre, curr, post, pitch_bin=None)` tries, in
order, the conditioned rungs and then the whole classic hierarchy:

```
exact triphone + bin  →  best-supported one-sided diphone + bin (ties favour
the left context)  →  phone + bin
→  exact triphone  →  best diphone  →  phone  →  class backoff  →  global backoff
```

A conditioned model therefore outranks an unconditioned one *only for the same
neighbourhood*: when a bin has no model, the frame drops to the ordinary
hierarchy, and the tiers are named `triphone+pitch`, `left+pitch`,
`right+pitch`, `phone+pitch` in the traces. Nothing borrows a model from a
*different* pitch region — a wrong bin would be worse than the pooled one — so
the fallback never fails and never misconditions. A model with no conditioned
tier ignores the argument entirely.

**F0 output is unchanged.** Conditioned HMMs emit the same full feature vector,
including dimension 0 (note-relative F0 in semitones): conditioning selects
*which* acoustic model is used, it does not generate absolute pitch. The
`pitch_source`/vibrato machinery, MLPG, the vocoder, the frame shift and the
label format are all untouched.

**Reporting.** `hms synth` diagnostics say how many frames were
pitch-conditioned, how many asked for a bin that has no trained model and which
bins those were — and say so plainly instead when the model carries no
conditioned models at all, or when no segment carried a scored note.
`hms evaluate` adds `pitch_conditioned_frames` and `pitch_fallback_frames`
(and never warns about the pitch-conditioning configuration fields, which
describe a model rather than a corpus). `hms inspect-model` reports the pitch
tier's parameter count separately and lists each bucket with its bin, note
range and support.

**Compatibility.** The tier is an additive format-4 payload: `model.yaml` gains
a `pitch_conditioning:` section (the definition: `enabled`, `bin_size`,
`bin_unit`, `n_bins`) and a `pitch_index:` section (per-bucket metadata), and
`pitch.npz` holds the arrays under `"{unit}/{bin}/"` prefixes. All three are
written only when conditioning produced models. Format-2 and format-3 models
load unchanged with conditioning recorded as disabled — no migration — and a
saved model carries its own bin width, so synthesis never needs the training
configuration.

With the default configuration nothing changes: same training, same synthesis,
same diagnostics. A model trained with the feature off writes exactly the files
it wrote before, with `format_version: 4` and two extra fields
(`pitch_conditioning_enabled: false`, `pitch_conditioning_bin_size: 6`) in the
recorded training configuration, where `context_enabled` already sits; checked
against the previous release on one corpus, its `hmm.npz` and `backoff.npz`
arrays, its state allocation, its F0 trajectory and its rendered audio are
bit-identical, and its parameter report grows by two lines (`pitch models : 0`,
`pitch HMM params : 0`).

**What this is not.** Not a quality claim. Splitting a unit's observations
across bins gives each bin less data than the pooled model it augments, so
whether a voice benefits is a property of the corpus (its range, and how evenly
that range is covered), measured with `hms evaluate` against an unconditioned
baseline — the thresholds decide what is modelled, and a bin with too few frames
is left unmodelled rather than fitted to a handful of frames.

## 7b. Model evaluation (`hms/core/evaluate.py`, `hms evaluate`)

`hms evaluate` compares trained models on an evaluation corpus without
rendering audio and **without an aggregate quality score**: it reports
separate objective metrics per model — evaluation-corpus log-likelihood (total
and per frame, under the units synthesis would select, contexts and pitch bins
included), voicing agreement, duration prediction MAE, backoff-routed frames,
and (for a pitch-conditioned model) `pitch_conditioned_frames` vs.
`pitch_fallback_frames` — so the reader sees *what* differs, not just a fused
number. It does not enforce that the evaluation corpus is disjoint from the
models' training data; the metrics describe the corpus they were measured on,
nothing more.

The likelihood is measured with the same note-per-segment policy as training, so
a conditioned model is scored on the frames it actually claims. A corpus whose
notes fall outside a model's trained bins therefore shows up as fallback frames
rather than as a mysterious likelihood change.

Because those metrics are only comparable between like-for-like models,
evaluation first cross-checks them: the feature spec must match exactly (hard
error), and differences in phoneme inventory, training method, seed, training
corpus paths, and any other non-context training setting are surfaced as
warnings. Context and pitch-conditioning settings are deliberately exempt —
comparing a model against its own baseline is the usual reason to run it, and
that is also how the experimental pitch tier is meant to be judged.

## 8. Synthesis pipeline (`hms/core/synthesizer.py`)

1. `plan` — score segments → phoneme durations → per-state frame counts →
   `(phone, state)` per frame; gaps become silence; unknown phonemes are
   reported and routed to the backoff model. With context models present, each
   segment's unit is resolved through the context hierarchy (exact triphone →
   best-supported diphone, ties left → phone → class backoff → global
   backoff), which changes the state allocation for that segment; the six-value
   return signature is unchanged. With pitch-conditioned models present, each
   segment first asks for the bin of the note it is being rendered at
   (`_segment_pitch_bin`: score note + transpose, clipped into MIDI range, never
   a generated F0) and the conditioned rungs are tried ahead of that hierarchy —
   silence and unnoted segments ask for nothing.
2. `frame_statistics` — per-state GMM means/variances stacked into per-frame
   statistics (dominant component by default, or the mixture marginal).
3. `mlpg` — the normalized static trajectory; optionally run the experimental
   GV optimizer on its static coefficients, using the model's per-feature GV
   targets (only when explicitly enabled).
4. `denormalize` → static features; voicing from HMM/phoneme statistics;
   target F0 from score notes, with optional acoustic/state-mean deviation and
   optional vibrato — or, when `f0=` is passed, that external trajectory
   instead (validated and converted by `external_f0_to_semitones`); report
   out-of-trained-range F0; decode features to `(f0, sp, ap)`.
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
                (including vibrato), optional static GV targets, HMM index,
                context index (when context modelling was on), pitch
                conditioning definition and pitch index (when pitch
                conditioning was on), parameter budget, metadata
hmm.npz         arrays: GMM weights/means/variances, self-loops, durations,
                voicing probabilities
backoff.npz     the same for the pooled per-class models
context.npz     the same for the sparse phone-context models and the optional
                global backoff (only written when any of them exist)
pitch.npz       the same for the pitch-conditioned models (only written when
                any exist)
```

YAML for anything a human might want to read or tweak, `.npz` for the arrays.
`HMSModel.save/load` is the serialization path for the acoustic tier; its
loader validates the format version. (The separate Phase-2 source predictor
uses its own `source.yaml` + NPZ format, documented below.) The current acoustic
format is 4; each version is
additive over the previous one (2 = baseline, 3 = the context tier, 4 = the
pitch-conditioned tier), so format-2 and format-3 models keep loading — with an
empty tier and, for pitch conditioning, `enabled: false` recorded. The additive
`global_variance` YAML section is optional in format 4: older models keep it
absent rather than receiving fabricated values, and GV is never enabled just
because the section exists. No version bump or new `.npz` payload is needed:

```yaml
global_variance:
  space: normalized_static
  utterances: 18
  target_variance: [0.72, 0.91, 0.35]  # actual list has static_dim entries
```

Malformed, non-finite, negative or wrong-dimension targets are rejected on
load. The list is indexed exactly like FeatureSpec's static vector, not like
its stacked delta streams. Context keys
are three `^`-joined phone symbols (`a^i^sil`); a reserved wildcard marks the
unmodelled side of a one-sided diphone (`s^a^_` = `a` given left neighbour `s`,
`_^a^i` = `a` given right neighbour `i`), which keeps the two diphone pools of
one bigram distinct.

Pitch-conditioned models reuse those same unit keys plus an integer bin, so the
condition stays structured rather than encoded in a name. Two YAML sections are
written together with `pitch.npz`, and only when the feature produced models:

```yaml
pitch_conditioning:      # the definition -- how to read a bin index
  enabled: true
  bin_size: 6
  bin_unit: semitones
  n_bins: 22
pitch_index:             # per-bucket metadata, keyed by unit then bin
  a:
    10: {kind: phone, unit: a, curr: a, pitch_bin: 10, note_min: 60,
         note_max: 65, frames: 480, occurrences: 9, ...}
```

Arrays inside `pitch.npz` are prefixed `"{unit}/{bin}/"` — e.g.
`a/10/means` — the same shape as the other payload files. Because the model
records its own `bin_size` (and `bin_unit`), synthesis needs no training
configuration to interpret a bin, and a bucket can be checked against the notes
it stands for straight from the file.

## Phase 2 source model (standalone)

The source predictor is implemented under `hms/source/`, alongside—not inside—
the generic acoustic HMM tier. `SourceTrainer` consumes Phase-1
`SourceSequence` objects, the existing `SourcePCA`, a labelled `Utterance`, and
an optional authoritative F0 track. It overlap-aligns per-cycle/per-frame PCA
coefficients onto the source analysis frame grid, derives the configured HMS
static/delta streams, then reuses `LeftToRightHMM`, `DiagGMM`, `DurationModel`,
`hms.core.context` and the banded `mlpg` solver.

The source observation vector is the PCA coefficient stream (plus its dynamic
streams). When enabled, phone context selects a sparse triphone/diphone HMM;
context symbols are not appended as an unbounded numeric feature vector. Each
state has a small mixture-gated ridge regression from three per-frame F0
features—normalized
semitone log-F0, its within-voiced-run delta, and a voiced flag—to its source
statistics. This makes F0 trajectories explicit while retaining the shared
HMM/GMM/MLPG infrastructure. A supplied track is never resized or pitch-clamped;
its sample rate, frame period and frame count must match the source data.

`SourceHMMModel.generate` returns both frame-aligned PCA coefficients and a
Phase-1 `SourceSequence` decoded through the saved PCA basis. It is deterministic
and does not sample. `voice` maps the trajectory to pitch-cycle units and leaves
unvoiced excitation to the existing downstream noise path; `residual` maps it to
frame-synchronous units and can model unvoiced residual shapes. The current
result is an excitation/source waveform only, not filtered audio. Gains are
neutral; the source predictor does not model amplitude.

The separate source directory contains `source.yaml`, `source_hmms.npz`, and
`source_pca.npz`. It does not change `HMSModel.save/load`, acoustic model files,
the synthesizer, or vocoder APIs. Training controls, usage and remaining scope
are documented in [source_model.md](source_model.md); focused tests live in
`hms/tests/test_source_hmm.py`.

## 10. Design trade-offs (what is deliberately missing)

| decision | why |
|---|---|
| per-phoneme HMMs, no state tying across phonemes | the inventory is small; backoff models cover rare phonemes with one pooled model per class |
| phone contexts are opt-in, sparse and capped | a full triphone inventory would spend parameters on contexts the corpus never shows; observed-only contexts with thresholds and a model cap keep the budget honest, and the fixed fallback hierarchy means nothing can fall through |
| no BOS/EOS tokens for context boundaries | the inventory's `sil` already marks utterance edges in labels and scores, so contexts reuse it instead of inventing parallel symbols |
| pitch conditioning is opt-in, binned and off by default | the score already fixes the pitch; conditioning the *spectral* model on it is an experiment about envelope-vs-pitch, not a fix for pitch generation. Fixed semitone bins keep the condition deterministic and inspectable (no regression, no per-frame F0), and a bin that does not clear its tier's support threshold simply does not exist, so the pooled model is used instead of one fitted to a handful of frames |
| a missing bin never borrows a neighbouring one | a model from the wrong pitch region is worse than the pooled one, so the fallback goes down the existing hierarchy rather than sideways across bins |
| `hms evaluate` reports metrics separately, no fused quality score | collapsing likelihood, voicing and duration errors into one number would hide which part of the model a change actually moved |
| log-normal durations, no duration HMM | the score already carries the timing; the model only fills gaps |
| 5 aperiodicity bands | the fine structure of `ap` is perceptually unimportant compared to 1025 extra parameters |
| MLSA is opt-in; `auto` still prefers WORLD, then `builtin` | adding a backend must not silently change what an existing configuration renders with, and the MLSA filter is a different synthesis model rather than a bug fix for the fallback |
| MLSA is opt-in, `auto` still prefers WORLD then `builtin` | adding a backend must not silently change what an existing configuration renders with; the MLSA filter is a different synthesis model, not a bug fix for the fallback |
| vibrato outside the HMM | MLPG would smooth it away; keeping it explicit makes it controllable |
| GV is opt-in, not a mandatory spectral postfilter | MLPG can over-smooth; the experimental static GV optimizer offers an inspectable, tunable variance target while leaving the normal render bit-identical when disabled |
| one speaker per model | adaptation/multi-speaker would complicate every stage; nothing in the format prevents adding it |

## 11. Extension points

* **New phonemes** — edit `phonemes.yaml` (or point `--phonemes` at your own).
* **New vocoder** — subclass `hms.vocoder.base.Vocoder`, register the name in
  `hms/vocoder/__init__.py` (`_BACKENDS`, `available_backends`, `get_vocoder`)
  and add it to the `vocoder` validators in `hms/core/synthesizer.py` and
  `hms/core/trainer.py` plus the `--vocoder` help text in `hms/cli/main.py`;
  `hms doctor` lists whatever the registry holds.  Keep `auto`'s precedence
  unless the new backend is meant to become the default.
* **Different features** — `FeatureSpec` is data; the trainer, model file and
  synthesizer read their geometry from it.
* **Different dynamics** — `window_matrix`/`mlpg` take the delta window as a
  parameter, so a different regression or a new dynamic feature only needs a
  matching `stream_taps`.
* **Expression** — `transpose`, `tempo`, `variance_scale`, experimental
  `gv_enabled`/`gv_weight`/`gv_iterations`, `pitch_variation` and `Vibrato` are
  synthesis-time knobs; a future GV optimizer can replace `hms/core/gv.py`
  without touching MLPG's banded solver.
* **External F0** — `synthesize(score, f0=…)` (CLI: `hms synth --f0-file`)
  replaces the generated contour. It is deliberately not a `SynthesisConfig`
  field: the trajectory is render-specific data, not a voice setting, and the
  whole override is one conversion function plus one branch in `synthesize`.
* **The source / excitation** — `hms/source` keeps Phase-1 analysis/PCA and the
  Phase-2 source HMM/GMM predictor separate from the acoustic tier.
  `SourceModel.analyze` extracts pitch-synchronous cycles
  (`VoiceSourceModel`) or frame-synchronous residual units
  (`GenericResidualSourceModel`), `SourcePCA` compresses them, and
  `SourceTrainer` / `SourceHMMModel` predict those coefficients from phone
  context and explicit F0. The standalone predictor is not called by the current
  full-audio synthesis pipeline; see [docs/source_model.md](source_model.md) for
  training, generation, tests, and the remaining source/vocoder integration work.
* **A different conditioning variable** — the pitch bins are one instance of a
  general shape: `hms/core/pitch_condition.py` owns the mapping from a scored
  segment to an integer condition, `HMSModel.resolve_unit(…, pitch_bin=…)` owns
  the extra rungs, and the trainer owns one collection pass plus a threshold
  check. Conditioning on another label attribute (dynamics, phonation, speaker)
  would mean a second such module and a second optional tier, not a rewrite of
  the acoustic model.
