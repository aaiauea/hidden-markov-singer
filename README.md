# HMS — Hidden Markov Singer

A lightweight, **non-neural** singing synthesiser: HMM/GMM acoustic modelling of
a compact WORLD parameterisation, with the musical score driving pitch, timing
and voicing.

HMS is a *modernised classical* singing synthesizer. It is deliberately small:
one speaker, a handful of phonemes, a few minutes of labelled audio, a model you
can read with a text editor and train on a laptop in seconds. There is no neural
network anywhere in the system — the acoustic model is a left-to-right HMM with
diagonal-covariance Gaussian mixtures, and parameter generation is the classic
maximum-likelihood parameter generation (MLPG) over static + delta features.

```
score (phones, notes, times) ──► duration/state plan ──► HMM state sequence
                                  │                          │
                                  │                          └─► GMM statistics ─► MLPG ─► optional GV ─► sp/ap
                                  │                                                       │
                                  └─► target musical F0 ────────────────────────────────┤
                                      optional learned deviation / vibrato ─► F0 ────────┤
                                  external F0 override (synthesize(f0=…)) ─► F0 ─────────┤
                                                                                         ▼
                                                        WORLD (f0, sp, ap) ──► waveform
```

## Design principles

* **No neural networks.** HMM/GMM + WORLD, nothing else.
* **Small data first.** The demo voice trains on **32 seconds** of audio to
  about **19,500 total free parameters** (≈3.0 per training frame, including
  class backoffs). Per phoneme
  budgets, tied covariances, variance floors, parameter sharing and class
  backoff models are all there to keep that possible.
* **Compact, statistically modelable features.** Mel-cepstrum spectral envelope
  (from WORLD's `sp`, sampled on a mel grid twice the model order so formants
  are not rounded onto the grid knots), mel-band aperiodicity (from `ap`), and
  a note-relative F0 feature for the optional learned-deviation mode, plus
  deltas. No hand-written formant tables anywhere in the engine.
* **Score F0 is the base path.** Ordinary synthesis uses the requested MIDI
  notes directly as target F0 and does not need learned pitch statistics.
  Note-relative F0 deviations from the acoustic HMM, separate state-mean pitch
  statistics, and explicit vibrato remain optional prosody extensions; absolute
  training-speaker F0 is never replayed. Callers can also supply their own
  frame-level F0 trajectory (`synthesize(..., f0=...)`, `hms synth --f0-file`),
  which replaces the generated contour instead of being mixed into it.
* **The requested pitch is the pitch you get.** `f0_floor`–`f0_ceil` bound the
  F0 *analyser* that produced the training features, not the synthesizer: a
  valid MIDI note above the ceiling (or below the floor) is rendered at its own
  frequency, with the model's nearest trained spectral envelope, and reported
  in `result.diagnostics`. Nothing is silently moved to the edge of the
  training range — see [Out-of-training-range F0](docs/architecture.md).
* **Every component is replaceable.** Vocoder backends, phoneme inventory,
  feature set, model files and the CLI are all thin layers over plain data.
* **Readable models.** A trained voice is a `model.yaml` you can inspect plus
  two compressed `.npz` array files.

## Install

```bash
pip install -r requirements.txt        # numpy + PyYAML (+ pytest for tests)
./tools/build_world.sh                 # optional: real WORLD via a small C++ shim
```

The build step compiles the vendored WORLD sources plus a C ABI wrapper into
`hms/vocoder/_native/libhms_world.so` (needs `g++`). It is **optional**: without
it HMS falls back to a pure-numpy vocoder, so the whole pipeline still runs.
There are two pure-numpy backends: `builtin` (the default fallback, a zero-phase
magnitude filter) and `mlsa`, an MLSA (mel log spectrum approximation) filter
with a per-frequency-band mixed excitation — see
[the MLSA vocoder backend](docs/mlsa.md) for the exact formulation, the
benchmark numbers and the trade-offs. `--vocoder mlsa` selects it; `auto` keeps
resolving to WORLD when available and to `builtin` otherwise.

`hms doctor` tells you which backends are available:

```
$ hms doctor
hms version      : 0.1.0
vocoder backends :
  pyworld   unavailable
  native    available
  builtin   available
  mlsa      available
```

## Quickstart

For the smallest explicit train → save → load → synthesize workflow (one
0.8-second generated recording, no native WORLD), see
[the minimal end-to-end example](examples/minimal/README.md).

One command generates an example corpus with a synthetic singer, trains on it
and sings it back:

```bash
hms demo --out hms-demo
# 1/5 reading corpus ... 5/5 duration, pitch and voicing models
#   model written to hms-demo/model (19,500 total free parameters)
# 3/3 synthesising the corpus back
#   wrote hms-demo/demo.wav
```

Listen to `hms-demo/demo.wav`. Then try the individual steps:

```bash
# 1. training: labels.tsv + a directory of WAVs -> a model directory
hms train --labels corpus/labels.tsv --wav-dir corpus/wav --out model

# 2. synthesis: score + model -> WAV
hms synth --model model --score corpus/score.tsv --out song.wav

# 3. what did I train? (phoneme stats, duration/pitch models, parameter budget)
hms inspect-model --model model --phoneme a

# 4. WORLD parameters only (useful for debugging or other vocoders)
hms extract --labels corpus/labels.tsv --wav-dir corpus/wav --out params --features
```

Useful flags: `hms train --covariance tied` (share one covariance per state —
fewer parameters for very small corpora), `hms train --iterations 10 --evaluate`,
`hms train --context` (optional sparse phoneme-context models, see below),
`hms train --pitch-conditioning --pitch-bin-size 6` (optional pitch-binned
acoustic models, experimental — see below),
`hms synth --transpose 5 --vibrato --variance-scale 2`,
`hms synth --gv --gv-weight 2 --gv-iterations 20` (experimental static GV;
see below), `hms synth --trace trace.tsv` (frame-by-frame phoneme/state/note/F0
table), `hms synth --f0-file contour.txt` (sing with your own F0 trajectory
instead of the generated one).

## Comparing models: `hms evaluate`

`hms evaluate` scores one or more trained models against an *evaluation
corpus* (a label file plus its WAVs) and prints the numbers side by side —
evaluation-corpus log-likelihood, voicing agreement, duration error and
backoff usage. It deliberately reports the metrics **separately**: there is no
aggregate quality score, because collapsing them would hide what actually
changed. The command does not require the evaluation corpus to be disjoint
from the models' training data; it measures whatever corpus it is given
(using a corpus the models have not seen is what makes the numbers read as
generalisation).

```bash
hms evaluate --labels eval/labels.tsv --wav-dir eval/wav \
    --model model-baseline --model model-context --json report.json
```

Before comparing, evaluation checks that the comparison is fair: the feature
specs must match exactly (a hard error), and differences in phoneme inventory,
training method, seed, training corpus paths, or any other non-context
training setting are reported as warnings. (Context settings are expected to
differ — that is usually what you are comparing.)

## Optional phoneme-context modelling

By default HMS models *phonemes*. Optionally — `context.enabled: true` in
`parameters.yaml` or `hms train --context` — it also learns HMMs for the phone
contexts that actually occur in the corpus. The design stays sparse and
data-efficient:

* only **observed** contexts are modelled — never a full triphone inventory;
* each context is the exact `(pre_phone, curr_phone, future_phone)` triple,
  plus one-sided diphone contexts `(pre, curr, _)` / `(_, curr, post)` where
  supported; utterance boundaries use the existing `sil` symbol (no BOS/EOS);
* a context gets its own HMM only past configurable support thresholds
  (`context.min_frames`, `context.min_occurrences`) and within the
  `context.max_models` cap — the best-supported contexts win deterministically;
* context HMMs are trained directly from the pooled raw feature sequences of
  their occurrences (like the class backoffs), and they take over state
  allocation, acoustic statistics and voicing for the segments they cover;
* resolution falls back gracefully: exact triphone → best-supported one-sided
  diphone (ties favour the left context) → dedicated phone HMM → phone-class
  backoff → optional pooled global backoff (`context.global_backoff`);
* the score notes remain the base F0 path; learned pitch deviations stay
  optional exactly as before.

With `context.enabled: false` (the default) nothing changes: same training,
same files, same synthesis. Context models are stored in `context.npz`
(model format 3; format-2 models still load, without contexts), and
`hms inspect-model` reports contextual, dedicated-phone, class-backoff and
global-backoff parameter counts separately.

## Optional pitch-conditioned acoustic models (experimental)

By default a phone's acoustic HMM pools **every** observation of that phone,
whichever note it was sung on: the spectral envelope the model emits for `a` is
an average of all the pitches the corpus contains. Optionally —
`pitch_conditioning.enabled: true` in `parameters.yaml` or
`hms train --pitch-conditioning` — HMS additionally learns the same units *per
pitch bin of the scored note*, so a phone can be rendered from the distribution
observed in that pitch region:

* a bin is `floor(MIDI note / bin_size)` with `bin_size` in semitones
  (`pitch_conditioning.bin_size`, default **6** — a tritone, two bins per
  octave), computed in exact integer arithmetic, so equal notes always share a
  bin and boundaries never depend on floating-point rounding; MIDI 0-127 maps
  to bins 0-21 at the default width;
* the condition is the **scored note**, not a measured F0: a phone keeps one
  condition across a note even where frames are unvoiced or the pitch wobbles,
  and silence, rests and unnoted segments carry no condition at all
  (`default_note` is never substituted for a missing note);
* training and synthesis call the same function (`hms.core.pitch_condition`),
  and synthesis asks for the bin of the note it was *given* (score note +
  `--transpose`), before any F0 exists — never for the bin of a generated F0;
* buckets are keyed by `(unit, bin)` — a phone symbol *or* a context key plus an
  integer — so the pitch condition stays structured metadata: the phoneme
  inventory, the duration model and the note-conditioned pitch model are
  untouched, and there are no synthetic phonemes like `a@60`;
* F0 generation is **not** part of this: the score still drives pitch, the
  optional learned deviation and vibrato are unchanged, and no second pitch
  predictor is introduced. Feature normalisation also stays the single
  corpus-wide one, so bins remain comparable and MLPG sees one geometry;
* sparse data is expected and handled by the existing threshold philosophy: a
  bucket becomes a model only past the support threshold of the tier it
  conditions (`training.min_phoneme_frames` for a phone, `context.min_frames` /
  `context.min_occurrences` for a trained context). Resolution prefers a
  conditioned model and otherwise falls back — exact context + bin → partial
  context + bin → phone + bin → the ordinary hierarchy (context → phone → class
  backoff → global backoff). A missing bin never fails and never borrows a
  model from a *different* pitch region; `hms synth` reports how many frames
  took each path, and `hms evaluate` reports `pitch_conditioned_frames` vs.
  `pitch_fallback_frames`.

With `pitch_conditioning.enabled: false` (the default) nothing changes: same
training, same synthesis, same diagnostics, and the same files on disk — a
model trained without the feature differs from a format-3 model only in
`format_version: 4` and in two recorded training fields
(`pitch_conditioning_enabled: false`, `pitch_conditioning_bin_size: 6`), next
to where `context_enabled` already sits. Conditioned models are stored in
`pitch.npz` with a `pitch_conditioning` / `pitch_index` section in `model.yaml`
(model format 4; format-2 and format-3 models still load, with the feature
recorded as disabled and no migration needed). The saved model records its own
bin width, so synthesis does not need the training configuration.

This is an **experimental modelling option**, not a quality claim: whether
conditioning helps a given voice is exactly what the option lets you measure
(`hms evaluate --model baseline --model conditioned`), and a corpus that only
covers a narrow range will simply train few or no buckets.

## Optional Global Variance generation (experimental)

MLPG chooses a smooth static acoustic trajectory from state means and delta
constraints, but its maximum-likelihood average can suppress the utterance-wide
variation present in the recordings. GV is an **opt-in post-MLPG optimization**
of those static trajectories, not a fixed variance multiplier. During training,
HMS stores the mean of each utterance's **within-utterance population variance**
per normalized static feature (one target each for note-relative F0, mel-cepstral
coefficients and aperiodicity bands; none for deltas). Utterances with fewer than
two frames are skipped. The targets are in the optional `global_variance` section
of `model.yaml` with the number of contributing utterances, not in a new model
file; the model format remains version 4. A pre-GV model has **no targets** and
loads/synthesizes unchanged with GV off; asking it for GV reports an error
rather than inventing statistics.

For each static feature, the optimizer starts at the usual MLPG solution `c0`
and trades off the *increase in MLPG negative log-likelihood* from moving away
from `c0` against the squared difference between the new trajectory's variance
`mean((c - mean(c))²)` and its training target. The trade-off is controlled by
`gv_weight` (default 1.0); `gv_iterations` (default 20) caps the gradient steps.
Steps use the exact variance gradient `2(c - mean(c))/T`, MLPG's own dynamic
windows and precisions, a diagonal preconditioner and a per-feature line search
that only accepts objective improvements. Scaling the penalty by the larger of
the starting/target variance (with a small floor) keeps flat features safe.
Constant trajectories have zero variance gradient and stay constant rather than
acquire invented noise; one-frame trajectories also stay unchanged. This is an
**approximate GV penalty**, not a reproduction of a classic GV-MLPG system with
a learned distribution of global variances or a perceptual quality guarantee.
The targets include silence/unvoiced frames; very different utterance lengths,
small corpora or mismatched score statistics may call for tuning or disabling GV.

GV is **disabled by default**, even on newly trained models. Compare a render
with and without it (with `--vocoder builtin` if WORLD is unavailable):

```bash
hms synth --model model --score score.tsv --out plain.wav
hms synth --model model --score score.tsv --out expressive.wav \
    --gv --gv-weight 2 --gv-iterations 20
```

```python
from hms.core.synthesizer import SynthesisConfig, Synthesizer

plain = Synthesizer(model, SynthesisConfig(gv_enabled=False)).synthesize(score)
expressive = Synthesizer(model, SynthesisConfig(
    gv_enabled=True, gv_weight=2.0, gv_iterations=20)).synthesize(score)
```

Or set `synthesis.gv_enabled: true`, `synthesis.gv_weight` (non-negative; zero
means no updates) and `synthesis.gv_iterations` (non-negative; zero means no
updates) in `parameters.yaml`; `--no-gv` overrides the YAML setting. Only the
normalized **static** output of MLPG changes; the optimized MLPG solve and
existing `variance_scale` remain untouched. In default `f0_source: score` mode,
the requested note/F0 is still taken from the score; GV of the note-relative F0
feature only affects the optional acoustic-deviation mode. For a synthetic
performance/variance check, run `python tools/bench_gv.py --frames 1000 --dim 30`.

## Source models (Phase 1 representation + Phase 2 predictor)

HMS models the filter with the acoustic HMM/GMM and leaves source/vocoder
integration unchanged. Phase 1 provides pitch-synchronous or frame-synchronous
source units and a NumPy PCA; Phase 2 adds a standalone HMM/GMM predictor over
those existing PCA coefficients, optionally selected by sparse phone context
and conditioned on an explicit frame-level F0 trajectory. It can train,
save/load and generate a Phase-1 `SourceSequence`, but does not yet connect to
full-audio synthesis or predict source gain. Use `SourceTrainer`,
`SourceTrainingExample` and `SourceHMMModel` from `hms.source`; explicit F0
must match the source analysis frame grid and is never silently resized.
`python tools/bench_source_pca.py`
measures representation reconstruction (a steady note: ~6 % relative error
with 8 coefficients; mixed material: ~56 %). See
[docs/source_model.md](docs/source_model.md) for the training/generation flow,
serialization format, limitations and Phase-1 measurements.

## Data format

Everything is a tab-separated text file; no database, no binary labels:

```
# utt_id  onset   offset  phone   note
scale_01  0.000   0.121   s       -
scale_01  0.121   0.180   i       60
scale_01  0.180   0.520   i       60
scale_01  0.520   0.560   l       62
```

* one audio file per utterance id (`wav/<utt_id>.wav`),
* times in seconds (`time_unit: frames` in `parameters.yaml` if you prefer
  analysis frames),
* `phone` is a symbol from `phonemes.yaml`,
* `note` is a MIDI note number, or `-` for "no pitch" (silence, unvoiced-only
  segments). Repeating the note across consecutive rows lets the phoneme
  sequence be as fine-grained as you like.

The same format is used for *training labels* (real recordings) and for
*synthesis scores*: a score is just a label file you do not have audio for.
`hms synth --duration-mode model` will even invent the timing if your score has
none.

## Configuration

`hms/config/parameters.yaml` holds every tunable (sample rate, frame period,
feature sizes, training method, covariance type, the optional context and
pitch-conditioning tiers, synthesis knobs, vibrato), with comments explaining
which way to turn each one. `hms/config/phonemes.yaml` *is*
the phoneme inventory:

```yaml
defaults:
  vowel: {n_states: 5, n_components: 2, voiced: true, can_hold_note: true}
  voiced_consonant: {n_states: 3, n_components: 1, voiced: true}
  unvoiced_consonant: {n_states: 2, n_components: 1, voiced: false}
  silence: {n_states: 1, n_components: 1, voiced: false}
phonemes:
  a: {type: vowel}
  m: {type: voiced_consonant}
  s: {type: unvoiced_consonant}
aliases: {pau: sil, A: a}
```

Adding a phoneme is one line here — nothing in the engine changes. A phoneme
that appears in a score but not in the model degrades gracefully to a pooled
"backoff" model for its phoneme class (and says so in the diagnostics).

## Python API

The CLI is a thin shell over the library, so the same pipeline is available
directly:

```python
from hms.core import labels
from hms.core.model import HMSModel
from hms.core.synthesizer import Synthesizer

model = HMSModel.load("hms-demo/model")
score = labels.load("hms-demo/corpus/score.tsv")
result = Synthesizer(model).synthesize(score)

from hms.data import wavio
wavio.write_wav("song.wav", result.audio, model.spec.fs)
```

```python
from hms.core.trainer import Trainer, TrainingConfig

config = TrainingConfig(label_file="corpus/labels.tsv", wav_dir="corpus/wav",
                        fs=44100, n_iterations=5)
model = Trainer(config).train()          # -> HMSModel
model.save("model")
```

## External F0 override

Synthesis normally takes its pitch from the score (plus the optional learned
deviation and vibrato). Pass `f0=` to `synthesize` — or `hms synth --f0-file` —
to supply the F0 trajectory yourself, for example a contour you edited,
imported from another singer, or generated by an external model:

```python
import numpy as np

# one value per synthesis frame, in Hz; 0.0 marks an unvoiced frame
frames = len(Synthesizer(model).synthesize(score).params.f0)
trajectory = 440.0 * np.ones(frames)
result = Synthesizer(model).synthesize(score, f0=trajectory)
```

```bash
# .npy, or a text file with one value per line (# starts a comment)
hms synth --model model --score score.tsv --out song.wav --f0-file contour.npy
```

* The trajectory is **authoritative**: the score note, the learned deviation and
  the generated vibrato are *not* added on top of it, and nothing else about the
  render changes (same `sp`/`ap`, same timing, same vocoder).
* It is aligned with the **final synthesis frame sequence** of the whole score
  (what `Synthesizer.plan` produces), not with individual phonemes. The frame
  count is `len(result.params.f0)` of a normal render of the same score (or
  `duration_in_seconds × 1000 / frame_period_ms`); `hms synth --trace` prints
  one row per frame, which is the easiest way to check the alignment.
* Unvoiced frames use the project's existing convention: **0.0 Hz** (anything
  below `spec.voiced_threshold`, 5 Hz by default, counts as unvoiced) — the same
  convention `AcousticFrameSequence.f0` and WORLD use. Internally that becomes
  `NaN` in `result.f0_semitones`.
* HMS never resamples, interpolates, truncates or pads an explicit trajectory:
  a wrong length, a wrong shape, an empty array, non-numeric values, `NaN`/`inf`
  or negative frequencies raise `ValueError` naming the offending frame, and the
  CLI turns that into a one-line error. Values outside the model's trained F0
  range (`f0_floor`–`f0_ceil`) are *not* an error and are not altered: they are
  synthesised at the supplied frequency and reported in `result.diagnostics`,
  exactly as for score-driven F0.

## Model format

```
model/
├── model.yaml     # format version, feature spec, phoneme set, duration and
│                  # pitch models, normalisation, optional GV static targets,
│                  # context + pitch indexes, parameter budget -- readable
├── hmm.npz        # GMM weights/means/variances and transition stats
├── backoff.npz    # per phoneme-class pooled models
├── context.npz    # sparse phone-context HMMs + optional global backoff
│                  # (only written when context modelling was enabled)
└── pitch.npz      # pitch-conditioned HMMs, keyed by (unit, pitch bin)
                   # (only written when pitch conditioning produced models)
```

## Tests

```bash
HMS_NO_AUTO_BUILD=1 python -m pytest
# 536 passed, 9 optional skips without WORLD (at the time of this change)
```

The suite covers the numerical core (banded Cholesky, MLPG against a dense
solve), the statistical models (GMM/HMM behaviour, duration allocation), the
feature transforms (round-trip accuracy in the model's own space), the vocoder
contract every backend shares (including the exact rendered length at both the
window-dominated and frame-grid-dominated regimes), the MLSA backend (its
filter reproducing the project's own mel-cepstrum at the analysis knots, the
truncation error of the default filter length, deterministic seeded synthesis,
silence, voiced tones, unvoiced noise, per-bin aperiodicity mixing, gliding and
stepping F0, one-frame and zero-frame utterances, non-contiguous views,
NaN/Inf and absurd envelopes, and end-to-end synthesis through a trained model),
model serialisation (including the format-3
context payload, the format-4 pitch-conditioning payload, optional GV targets
and format-2/3 compatibility), the sparse context feature, the optional pitch-conditioned
acoustic models (bin arithmetic and validation, the scored-note/silence policy,
bucket separation, the resolution ladder and its fallbacks, serialisation,
older model formats, determinism, coexistence with contexts, evaluation
reporting and the CLI), optional GV (variance/gradient numerics, independent
static features, optimizer behavior, legacy models and end-to-end generation),
model evaluation, the CLI, out-of-training-range F0 (in
both directions, from the score and from an external trajectory, asserted on
the parameters handed to the vocoder), the external F0 override (trajectory
preservation, independence from learned deviation and vibrato, and its
validation), and an end-to-end train→synthesise run that checks the rendered
notes really are the requested ones.

## How it works

See [docs/architecture.md](docs/architecture.md) for the pipeline, the feature
layout, the HMM/GMM/training design, the note-conditioned pitch model, and the
trade-offs behind each choice, and [docs/source_model.md](docs/source_model.md)
for the Phase 1 source/excitation representation and Phase 2 standalone source
predictor (still not wired into the synthesizer). `hms/model/…` is a small system on purpose; the
documentation tries to explain *why* each piece looks the way it does.

## Limitations

* One speaker per model; no voice conversion or adaptation yet.
* The bundled inventory is a small demo set (open vowels and the consonants
  that carry a melody) — extend `phonemes.yaml` for real lyrics.
* No explicit duration HMM: state durations come from the score (or per-phoneme
  log-normal statistics when the score has no timing).
* The `builtin` vocoder is a fallback: it uses a zero-phase magnitude response
  instead of WORLD's minimum-phase impulse response, so build the native
  backend (or install `pyworld`) for the real thing.  The optional `mlsa`
  backend renders the same parameters through a mel-log-spectrum exponential
  with a per-bin mixed excitation (better pitch/voicing fidelity, ~1.0-1.4x the
  time and 1.4-4.2x the peak heap of `builtin` — neither faster nor smaller,
  and still ~100x faster than real time; see [docs/mlsa.md](docs/mlsa.md)).
* WORLD's synthesis is returned unscaled (see `Vocoder.synthesize`); it can
  overshoot `[-1, 1]` on very periodic material and `write_wav` applies the
  headroom.  A trained model is tied to the feature definition that produced
  it: `model.yaml` records `format_version: 4` (format-2 and format-3 models
  still load).
* Pitch conditioning is an optional modelling tier, not an improvement claim:
  it splits the acoustic observations of a unit across pitch bins, so each bin
  sees less data than the pooled model it augments. Whether a voice benefits
  depends on the corpus (its pitch range, and how much of it sits in each bin),
  and the support thresholds decide — a bin with too few frames simply does not
  get a model and its notes fall back. Bins are keyed by the scored note, so a
  voice that was recorded softly at pitch is not conditioned on how it was
  actually sung at that moment, and one bin size serves the whole model.
* `hms evaluate` compares models with separate objective metrics
  (evaluation-corpus likelihood, voicing agreement, duration error, backoff
  usage) and by design reports no aggregate quality score; it also does not
  enforce that the evaluation corpus is disjoint from the training data.
* Synthesis uses a single optimized MLPG solve, optionally followed by the
  experimental iterative GV penalty on static features. GV is a global target
  for each feature, not a phoneme-/note-dependent expression model; there is
  no broader prosody editing beyond `--transpose`, `--tempo`,
  `--variance-scale`, vibrato and an externally supplied F0 trajectory
  (`--f0-file` / `synthesize(f0=...)`).
* Notes outside the F0 range the model was trained on are rendered at the
  requested pitch using the nearest trained spectral envelope, so they are
  audible but are not *natural* for this voice: the envelope comes from frames
  the model never saw at that pitch.
