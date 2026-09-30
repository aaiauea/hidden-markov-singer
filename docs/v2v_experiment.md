# Experimental LPC/WORLD frontend for V2V

**Status: analysis-only prototype.** This is not a voice converter and does not
change HMS model files, training, score-driven synthesis, or the acoustic
trajectory generator. It tests whether compact source-audio measurements might
be useful to a future small V2V conditioning/recognition component.

## What the existing HMS model actually consumes

The current training and synthesis path is supervised by phone labels/scores;
it is not an audio-to-audio mapping:

1. `Trainer._analyse_utterance` reads mono WAV audio, asks the configured
   `Vocoder.analyze_to_sequence` for `(f0, sp, ap)`, and aligns the resulting
   frames to **labelled phoneme spans**.
2. `FeatureSpec.encode` converts each WORLD frame into the static layout
   `[F0 feature, mel-cepstrum c0..c(n_mcep-1), aperiodicity bands]`. The default
   static dimension is `1 + 30 + 5 = 36`. During training the trainer replaces
   slot 0 with **note-relative** log-F0; unvoiced F0 is stored as zero, with
   voicing supervised separately.
3. `add_dynamic_features` appends the configured delta and optional
   delta-delta streams (HTS-style regression, edge-replicated). Corpus
   normalization is applied before phoneme/state HMM/GMM fitting. Phone labels
   and scored notes/times provide the alignment and supervision; WORLD does not
   infer the phones.
4. A trained `HMSModel` stores its `FeatureSpec`, normalization, HMM/GMMs,
   phone inventory and optional context/pitch/GV sections. `HMSModel.load`
   reads these model-directory files.
5. `Synthesizer.synthesize(score, f0=None)` plans phone/state frames from an
   HMS `Score`, runs MLPG over target-model state statistics, chooses a target
   F0 contour/voicing, decodes the target features to `(f0, sp, ap)`, then calls
   the vocoder. The optional `f0=` argument replaces **only** the F0 contour
   and must have exactly one value per planned synthesis frame. It is not a
   source-audio feature interface.

The target HMM therefore generates the target singer's learned spectral
statistics only after a phone/context sequence and durations have been chosen.
A source LPC vector cannot be passed to the current `Synthesizer`; it has the
wrong semantics and there is no phone recognizer or acoustic-feature mapper in
HMS. In particular, assigning guessed/fake phones from F0 or LPC would hide the
main unsolved part of V2V, so this prototype does not do that.

The current optional global-variance implementation is also unrelated to
source analysis: training estimates per-static-feature variance targets, and
synthesis may apply its experimental static-space optimizer **after MLPG**.
GV stays off by default and was not changed.

## Prototype API and representation

The new module is `hms/core/v2v.py`:

```python
from hms.core.v2v import V2VAnalysisConfig, V2VFrontend
from hms.data.wavio import read_wav

source_audio, fs = read_wav("source.wav")
analysis = V2VFrontend(V2VAnalysisConfig(
    vocoder="native",       # auto | native | pyworld | builtin
    frame_period_ms=5.0,
    frame_length_ms=25.0,
    lpc_order=16,
    cepstral_order=20,
)).analyze(source_audio, fs)

# Continuous observations and frame times; neither contains phone labels.
features = analysis.candidate_features
frame_times = analysis.frame_times_s
source_f0_hz = analysis.f0_hz  # 0.0 is unvoiced
```

`V2VAnalysis` exposes the source observations independently of timing:

| Field | Shape | Meaning |
|---|---:|---|
| `f0_hz` | `(T,)` | Existing HMS vocoder's F0; non-finite/non-positive estimates are represented as 0 Hz. |
| `energy_rms` | `(T,)` | RMS of the original, not pre-emphasized, centred frame. |
| `lpc_coefficients` | `(T, order + 1)` | `[1, a1, ...]` for `A(z) = 1 + a1 z^-1 + ...`; diagnostic/source observation, not an HMS acoustic vector. |
| `reflection_coefficients` | `(T, order)` | Levinson-Durbin reflection coefficients, clipped to a strict interior limit. |
| `prediction_error_fraction` | `(T,)` | Normalized one-step LPC residual power from the windowed/pre-emphasized autocorrelation. |
| `lpc_cepstra` | `(T, cepstral_order)` | c1..cN of the causal log-amplitude response `log(1/A(z))`; c0/gain is kept separate. |
| `lpc_log_spectrum` | `(T, spectrum_bins)` | Mean-centred log-power envelope evaluated from the stable all-pole filter. |
| `frame_times_s`, `center_sample_indices` | `(T,)` | Nominal vocoder frame grid and nearest input-sample centres. |
| `candidate_features` | `(T, cepstral_order + 3)` | Cepstra + absolute log-F0 in semitones re A4 + voiced flag + log-RMS dB; a classical-model probe layout only. |

LPC uses a Hann window, configurable pre-emphasis (default 0.97), biased
autocorrelation and batched Levinson-Durbin recursion. Coefficients are scaled
from normalized autocorrelation, so their shape does not depend on signal gain.
Reflection clipping plus an early stop when residual power collapses avoids
continuing a numerically ill-conditioned recursion on nearly periodic or silent
frames. Silent frames retain the identity LPC polynomial. With no voiced
history, shape features remain zero. If valid voiced material precedes a
terminal unvoiced suffix, the last cepstral and log-spectrum shape is edge-held
across that suffix to avoid a synthetic drop in downstream sequences. The raw
LPC coefficients, F0, voicing and RMS remain measured; empty or all-unvoiced
inputs have no edge value to hold and remain unchanged. The log spectral
envelope is mean-centred, so energy remains in the separate RMS feature.

The frontend asks the current `Vocoder.analyze` interface for F0 and discards
its returned source `sp`/`ap` arrays. This keeps source WORLD spectral
parameters out of the candidate target path, but the present vocoder interface
still computes those arrays internally; see the memory result and limitations
below. Use `vocoder="native"` or `"pyworld"` for real WORLD F0 where available.
`"builtin"` is the existing NumPy fallback: useful for portable tests, but not
real WORLD and less accurate for pitch. The fallback autocorrelation window now
covers at least 2.5 periods at the configured F0 floor while retaining the
original hop and output frame count; this reduces octave errors for low F0.

## Reproducible experiment

No real labelled audio files or large dataset are bundled in the checkout. The
repository does include `hms/data/demo_singer.py`, a deterministic synthetic
source-filter singer with known segment labels and varied notes; the tests also
create its small corpus in temporary directories. The benchmark reuses that
generator and writes no dataset into the repository:

```sh
python tools/bench_v2v.py --backend native --seconds 2 --repeat 7 --warmup 2
python tools/bench_v2v.py --backend builtin --seconds 2 --repeat 7 --warmup 2
```

The second probe is a five-vowel nearest-centroid classifier: centroids are
built from LPC cepstra for synthetic vowels at MIDI 60 and unit gain, then
scored on 20 clips at MIDI 48/72 and gain scales 0.25/2.0. It deliberately uses
no F0/energy in the classifier. The timed section is frontend analysis only;
no target HMS model is trained and no synthesis call is made. Memory is the
`tracemalloc` peak for one frontend call (not process RSS or a complete native
allocator profile), and retained bytes count the output arrays.

### Measured run in this checkout

Environment: Linux, Python 3.11.2, NumPy 2.4.6; 22.05 kHz source-filter test
waveform, 5 ms frame period, 25 ms LPC frame, order 16, 20 cepstra, 129
log-spectrum bins, two warm-ups and seven timed repeats. `native` selected HMS's
ctypes C-API WORLD backend; `builtin` selected the NumPy fallback.

| Measurement | Native WORLD backend | Builtin fallback |
|---|---:|---:|
| Backend frames for 2.000 s | 401 (0.000–2.000 s) | 405 (0.000–2.020 s) |
| Median frontend extraction time | 161.9 ms | 103.6 ms |
| Throughput | 12.4x real time | 19.3x real time |
| `tracemalloc` peak | 12.52 MiB | 34.42 MiB |
| Retained output arrays | 0.57 MiB | 0.58 MiB |
| Candidate feature matrix | 72.1 KiB | 72.8 KiB |
| Known 220 Hz tone, median estimated F0 | 219.75 Hz (-2.0 cents) | 225.0 Hz (+38.9 cents) |
| Median normalized LPC prediction-error fraction | 2.80e-5 | 2.80e-5 |
| Median LPC log-envelope / periodogram correlation | 0.878 | 0.878 |
| Median centred log-power envelope RMSE (natural-log units) | 5.40 | 5.40 |
| Synthetic vowel probe | 16/20 (80%) | 16/20 (80%) |

A separate builtin autocorrelation stress signal with a 110 Hz fundamental and
an eight-times-stronger second partial tracked at 110.8 Hz (instead of the
former short-window octave estimate near 220 Hz). The same test setup tracked
165 Hz and 220 Hz signals at 167.0 Hz and 222.7 Hz, respectively. These are
steady synthetic signals, not natural-voice pitch accuracy results.

Timings are a small local run, not a hardware-independent performance claim.
The builtin backend's 405-frame grid extends 20 ms past the 2 s input because
its existing analysis frame geometry differs from WORLD; this prototype retains
the backend's frame count and expresses its nominal 5 ms grid rather than
silently resampling or truncating it. With Native WORLD, the measured frame
count/time grid matches the expected 401 frames through 2 s.

The LPC prediction residual is a signal-model diagnostic, **not** a voice
reconstruction-quality score. The spectral correlation is between a
mean-centred LPC-derived log-power envelope and the frame periodogram; its
0.878 value says the representation retains a substantial amount of source
spectral shape. That can help separate synthetic vowels, but it is also exactly
why LPC should not be assumed to remove source-speaker identity. The 80% vowel
result only shows separability for this one artificial singer under the tested
pitch/gain changes. There is no held-out speaker, natural speech/singing, noisy
recording benchmark, context recognition score or target-voice conversion
measurement here.

Focused tests additionally cover deterministic coefficients, shapes, silence,
noise/unvoiced F0, voiced-to-noisy transitions, non-contiguous arrays, tiny
signals, short LPC windows, coefficient finiteness/stability, frame count and
time alignment, gain separation, and F0 passthrough from the selected backend.

## What the measurements do—and do not—say

**Potentially useful:** LPC smooths fine harmonic structure and yields compact
formant/spectral-shape cues; cepstra are a more natural small-model input than
raw LPC coefficients. WORLD contributes established F0/voicing analysis, while
RMS, explicit frame times and local spectral shape provide energy and temporal
observations. The same-singer synthetic vowel probe remains useful after the
tested pitch and gain changes.

**Still source-dependent / missing:** LPC is an all-pole approximation of the
observed signal. It retains vocal-tract/formant shape and can be affected by
pitch harmonics, excitation, noise, microphone/channel and phonetic context.
F0 is explicitly source prosody. C0/gain was removed from the LPC shape, but
that does not make the remaining coefficients speaker-invariant. WORLD F0 plus
LPC is not a phoneme recognizer, and continuous frame features alone do not
provide a canonical phone sequence, context labels, word boundaries or target
HMM state durations. The current target model's acoustic HMMs are indexed by
labelled phones/context, not by these observations.

To attempt arbitrary V2V, the next required experiment is a labelled
source-content component: for example a small classical phone/context classifier
and sequence decoder trained/evaluated on paired or multi-speaker labelled audio,
plus a timing/alignment and source-to-target prosody policy. HMS would then need
an explicit bridge from those predictions to its phone/context/duration input
(or a separately evaluated acoustic mapping model). No synthetic labels should
be substituted for audio inference.

Compared conceptually, **WORLD + LPC + a small classical model** is lightweight,
inspectable and cheap to deploy, but must learn phonetic invariance from limited
labels and has no language/phonotactic prior. **HuBERT/wav2vec-style encoders**
bring large self-supervised neural representations and typically stronger
content abstraction, at the cost of large pretrained weights, neural runtime
and training/integration complexity. This prototype does not implement either
a classical recognizer or a neural encoder, and the measurements do not show
that the LPC representation is sufficiently content-oriented for cross-speaker
V2V.

## Safety / compatibility boundary

- `V2VFrontend` is a standalone analysis module; normal HMS training and
  `Synthesizer.synthesize(score)` are untouched.
- It does not alter `FeatureSpec`, model serialization/loading, duration/context
  resolution, pitch policy, GV or the existing WORLD synthesis parameters.
- It does not call MLPG and does not modify the optimized MLPG assembly,
  banded-Cholesky or banded-solve code.
- A passing LPC fit, a nonzero F0 track or the synthetic vowel probe is not
  evidence of working voice conversion. There is no source-content-to-target
  synthesis path in this prototype.
