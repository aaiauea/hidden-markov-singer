# Minimal HMS: tiny corpus → trained model → WAV

This is a plumbing example, **not a usable singing voice or a pretrained model**.
It generates one 0.8-second synthetic recording, learns an HMM/GMM model from
that recording, saves it, then loads it in a separate CLI process to sing a
1.2-second, two-note score. No downloads of recordings or native WORLD needed.

## 1. Setup

Use Python 3.9+ and run all commands **from the repository root**, not from this
directory. A virtual environment avoids system-package permission conflicts:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export HMS_NO_AUTO_BUILD=1
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell and use
`$env:HMS_NO_AUTO_BUILD = "1"`. Runtime dependencies are just NumPy and PyYAML;
the `dev` extra installs pytest for the regression test. No compiler, pyworld,
SciPy, or external audio tools are required. Do **not** run `tools/build_world.sh`.

The commands below use `python -m hms.cli.main`, the module entry point for the
existing `hms` CLI, so they use the same Python environment as setup.

## 2. Generate, train, synthesize

Run these three commands in order (shell line continuations are shown):

```sh
python examples/minimal/generate_corpus.py
python -m hms.cli.main train \
  --labels examples/minimal/labels.tsv \
  --wav-dir examples/minimal/out/wav \
  --config examples/minimal/parameters.yaml \
  --phonemes examples/minimal/phonemes.yaml \
  --vocoder builtin --name minimal \
  --out examples/minimal/out/model
python -m hms.cli.main synth \
  --model examples/minimal/out/model \
  --score examples/minimal/score.tsv \
  --config examples/minimal/parameters.yaml \
  --vocoder builtin \
  --out examples/minimal/out/song.wav
```

In PowerShell, put each command on one line instead of using `\` continuations.
Training performs two Viterbi iterations. Both train and synth explicitly
select `--vocoder builtin`, even if WORLD is installed. The configuration also
sets `builtin` separately for analysis and synthesis; selecting it for training
alone does not force the synthesis backend. `HMS_NO_AUTO_BUILD=1` is an extra
safeguard against native auto-builds, not a backend selector. The trainer's
“analysing (WORLD)” stage name refers to the parameter representation, not to
a native dependency; synthesis should print `backend : builtin`.

Listen to **`examples/minimal/out/song.wav`**. Expect a short, robotic/buzzy
“ah … ah”, on C4 (MIDI 60, about 261.6 Hz), then E4 (MIDI 64, about 329.6 Hz),
with brief quiet gaps. The builtin vocoder is an approximation, and this tiny
corpus cannot yield realistic speech or expressive singing. The example shows
that acoustic statistics can be trained and persisted, then used with a new
score's pitch and timing—not that one recording makes a production voice.

The generator uses the existing `hms.data.demo_singer` utility with a fixed
seed and no pitch jitter, scoop, drift, or vibrato. It reads the checked-in HMS
labels; it neither converts lyrics/scores into labels nor trains a model.
Re-running it overwrites the same WAV deterministically. Training and synthesis
also use fixed seeds (model metadata includes the training timestamp).

## Files and directory structure

After running the workflow:

```text
examples/minimal/
├── README.md
├── generate_corpus.py        # deterministic synthetic recording generator
├── labels.tsv               # training alignment for training_a.wav
├── score.tsv                # desired output; no matching recording needed
├── parameters.yaml          # ordinary HMS acoustic/training/synthesis config
├── phonemes.yaml            # ordinary HMS inventory: a (vowel), sil (silence)
└── out/                     # generated, ignored by Git; safe to delete
    ├── wav/
    │   └── training_a.wav    # 0.8 s, 22050 Hz, mono 16-bit PCM training audio
    ├── model/
    │   ├── model.yaml       # feature spec, inventory, duration/pitch stats,
    │   │                    # normalization, metadata and model structure
    │   ├── hmm.npz          # learned per-phone HMM/GMM arrays
    │   └── backoff.npz      # learned pooled phoneme-class HMM/GMM arrays
    └── song.wav             # about 1.2 s, 22050 Hz, mono 16-bit PCM output
```

Only the six source files above `out/` are checked in. There is no pretrained
model. The recording is generated locally (about 35 KB); all models and audio
stay under `out/`, using the repository's existing ignore rule.

### Label and score format

Both TSV files use HMS's **existing five-column label format**, with literal
tabs separating fields. A line beginning with `#` is a comment/header:

```text
# utt_id    onset    offset    phone    note
training_a  0.000    0.100     sil      -
training_a  0.100    0.700     a        60
training_a  0.700    0.800     sil      -
```

- `utt_id`: utterance name. Training looks for `<wav-dir>/<utt_id>.wav`.
  Here every training row belongs to `training_a.wav`.
- `onset`, `offset`: segment start/end in **seconds** from the start of that
  utterance, not sample indices or milliseconds. Rows here are contiguous,
  ordered, start at zero, and cover the whole recording.
- `phone`: inventory symbol. `a` is the sustained “ah” vowel; `sil` is silence.
- `note`: MIDI note number (not frequency in Hz); `-` means no pitch for `sil`.

`labels.tsv` aligns 100 ms silence, 600 ms of `a` at MIDI 60, then 100 ms
silence with the generated recording. `score.tsv` uses the same format but
names the output utterance `song`: 100 ms silence, 450 ms `a` at 60, 100 ms
silence, 450 ms `a` at 64, 100 ms silence. No `song.wav` input is needed.
Synthesis gets the phone inventory and acoustic feature spec from the saved
model; it does not need `--phonemes`, training labels, or the training WAV.

### Why this small corpus works

HMS does not require multiple utterances. With 5 ms frames, this clip gives
roughly 120 vowel frames and 40 silence frames, both above the unchanged
20-frame minimum for a dedicated phone model. Keeping a 600 ms vowel gives
the builtin pitch analyser a stable periodic region rather than just edges.
The inventory uses three vowel states, one silence state, and one Gaussian
per state. The config reduces features to 12 mel-cepstral coefficients and
three aperiodicity bands, with deltas enabled, at 22050 Hz. This is a small
practical smoke corpus, not a claim of the absolute minimum number of samples.
Only these two phones are demonstrated; there is no consonant or lyric coverage.

## Regression test

From the repository root with the environment active:

```bash
HMS_NO_AUTO_BUILD=1 python -m pytest hms/tests/test_minimal_example.py
```

The test copies the example to a temporary directory, runs the three workflow
commands directly from this README, checks deterministic corpus generation,
loads the saved model, and checks the rendered WAV. It uses builtin throughout
and leaves no generated files in the checkout.

## Troubleshooting

- **`No module named hms`, missing NumPy/YAML, or `hms` not found:** activate the
  virtual environment and repeat `python -m pip install -e '.[dev]'` from the
  repository root. Use that environment's `python` for every command.
- **File not found / missing audio / no training data:** run from the repository
  root and generate the corpus first. Confirm
  `examples/minimal/out/wav/training_a.wav` exists; the basename must match
  `training_a` in the labels. Do not train on `score.tsv`.
- **Sample-rate mismatch:** use the supplied `--config` for training. HMS does
  not resample this corpus automatically; its WAV and config must both be
  22050 Hz. Regenerate the WAV if you changed the acoustic sample rate.
- **Native WORLD build/library errors:** keep `--vocoder builtin` on **both**
  commands, use the supplied config, and set `HMS_NO_AUTO_BUILD=1`. There is no
  need to install or compile WORLD for this example.
- **Phones skipped / unexpectedly silent result after editing inputs:** keep
  seconds as the time unit, `a`/`sil` as inventory symbols, MIDI numbers on vowel
  rows, and sufficient frames per phone. Restore the supplied files and rerun
  all three commands to rule out mismatched/stale artifacts.
- **Buzzy or unnatural audio:** expected for the builtin backend and one tiny
  artificial recording. This tests the pipeline, not voice quality. Start
  playback at low volume.
