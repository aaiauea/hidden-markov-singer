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
                                  │                          └─► GMM statistics ─► MLPG ─► sp/ap
                                  │                                                       │
                                  └─► target musical F0 ────────────────────────────────┤
                                      optional learned deviation / vibrato ─► F0 ────────┤
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
  training-speaker F0 is never replayed.
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
it HMS falls back to a pure-numpy vocoder (`builtin`), so the whole pipeline
still runs — the spectral quality is just lower. `hms doctor` tells you which
backends are available:

```
$ hms doctor
hms version      : 0.1.0
vocoder backends :
  pyworld   unavailable
  native    available
  builtin   available
```

## Quickstart

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
`hms synth --transpose 5 --vibrato --variance-scale 2`, `hms synth --trace
trace.tsv` (frame-by-frame phoneme/state/note/F0 table).

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
feature sizes, training method, covariance type, synthesis knobs, vibrato), with
comments explaining which way to turn each one. `hms/config/phonemes.yaml` *is*
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

## Model format

```
model/
├── model.yaml     # format version, feature spec, phoneme set, duration and
│                  # pitch models, normalisation, context index, parameter
│                  # budget -- readable
├── hmm.npz        # GMM weights/means/variances and transition stats
├── backoff.npz    # per phoneme-class pooled models
└── context.npz    # sparse phone-context HMMs + optional global backoff
                   # (only written when context modelling was enabled)
```

## Tests

```bash
HMS_NO_AUTO_BUILD=1 python -m pytest
# 221 tests: 213 passed, 8 optional skips without WORLD
```

The suite covers the numerical core (banded Cholesky, MLPG against a dense
solve), the statistical models (GMM/HMM behaviour, duration allocation), the
feature transforms (round-trip accuracy in the model's own space), the vocoder
contract for both backends, model serialisation (including the format-3
context payload and format-2 compatibility), the sparse context feature,
model evaluation, the CLI, and an end-to-end train→synthesise run that checks
the rendered notes really are the requested ones.

## How it works

See [docs/architecture.md](docs/architecture.md) for the pipeline, the feature
layout, the HMM/GMM/training design, the note-conditioned pitch model, and the
trade-offs behind each choice. `hms/model/…` is a small system on purpose; the
documentation tries to explain *why* each piece looks the way it does.

## Limitations

* One speaker per model; no voice conversion or adaptation yet.
* The bundled inventory is a small demo set (open vowels and the consonants
  that carry a melody) — extend `phonemes.yaml` for real lyrics.
* No explicit duration HMM: state durations come from the score (or per-phoneme
  log-normal statistics when the score has no timing).
* The `builtin` vocoder is a fallback: it uses a zero-phase magnitude response
  instead of WORLD's minimum-phase impulse response, so build the native
  backend (or install `pyworld`) for the real thing.
* WORLD's synthesis is returned unscaled (see `Vocoder.synthesize`); it can
  overshoot `[-1, 1]` on very periodic material and `write_wav` applies the
  headroom.  A trained model is tied to the feature definition that produced
  it: `model.yaml` records `format_version: 3` (format-2 models still load).
* `hms evaluate` compares models with separate objective metrics
  (evaluation-corpus likelihood, voicing agreement, duration error, backoff
  usage) and by design reports no aggregate quality score; it also does not
  enforce that the evaluation corpus is disjoint from the training data.
* Synthesis is a single-pass MLPG render; no prosody/expression editing beyond
  `--transpose`, `--tempo`, `--variance-scale` and vibrato.
