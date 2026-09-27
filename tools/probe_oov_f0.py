"""Diagnostic probe: where does an unexpected measured F0 come from?

Renders a held C4 note and reports three *different* F0s, so a post-hoc pitch
estimator's answer can be compared with what HMS actually asked the vocoder for:

  1. internal  -- ``SynthesisResult.f0_semitones`` (semitones re. f0_ref)
  2. vocoder   -- ``SynthesisResult.params.f0`` (Hz, what the backend consumes)
  3. measured  -- a fresh F0 estimate on the rendered waveform

Run: ``python3 tools/probe_oov_f0.py [note] [transpose]``
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hms.core import labels as labels_module           # noqa: E402
from hms.core.dsp import autocorrelation_f0             # noqa: E402
from hms.core.features import semitone_to_hz            # noqa: E402
from hms.core.phonemes import PhonemeSet                 # noqa: E402
from hms.core.synthesizer import SynthesisConfig, Synthesizer  # noqa: E402
from hms.core.trainer import Trainer, TrainingConfig     # noqa: E402
from hms.data.demo_singer import SingerConfig, make_dataset  # noqa: E402

FS, FFT = 22050, 1024


def main() -> int:
    note = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    transpose = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    directory = Path(tempfile.mkdtemp(prefix="hms_probe_"))
    dataset = make_dataset(directory, fs=FS, singer=SingerConfig(fs=FS, seed=3),
                           label_jitter_ms=8.0, seed=3)
    config = TrainingConfig(label_file=dataset["labels"], wav_dir=dataset["wav_dir"],
                            fs=FS, fft_size=FFT, n_mcep=20, n_band=5,
                            use_delta=True, n_iterations=2,
                            min_phoneme_frames=10, seed=0)
    model = Trainer(config, PhonemeSet.default()).train()

    score = labels_module.Score([labels_module.Utterance("held", [
        labels_module.Segment("sil", 0.0, 0.1),
        labels_module.Segment("a", 0.1, 0.6, note=note),
        labels_module.Segment("sil", 0.6, 0.7)])])
    synth = Synthesizer(model, SynthesisConfig(vibrato=False, seed=0,
                                                vocoder="builtin"))
    result = synth.synthesize(score, default_note=note)

    internal = np.asarray(result.f0_semitones, dtype=float)
    vocoder_f0 = np.asarray(result.params.f0, dtype=float)
    voiced = np.isfinite(internal)
    measured = autocorrelation_f0(result.audio, FS, 4 * int(FS * 0.005),
                                  int(FS * 0.005), model.spec.f0_floor,
                                  model.spec.f0_ceil)
    measured_voiced = measured[measured > 0]

    requested = labels_module.midi_to_hz(note + transpose)
    print(f"requested note            : MIDI {note + transpose:g}"
          f" = {requested:.2f} Hz")
    print(f"model trained F0 range    : {model.spec.f0_floor:g}-"
          f"{model.spec.f0_ceil:g} Hz")
    print(f"frames voiced             : {int(voiced.sum())}/{len(internal)}")
    print(f"1. internal F0 (semitones): "
          f"{np.nanmin(internal[voiced]):.2f} .. {np.nanmax(internal[voiced]):.2f}"
          f"  -> {semitone_to_hz(internal[voiced], model.spec.f0_ref_hz).min():.2f}"
          f" .. "
          f"{semitone_to_hz(internal[voiced], model.spec.f0_ref_hz).max():.2f} Hz")
    voiced_f0 = vocoder_f0[vocoder_f0 > 0]
    print(f"2. vocoder params.f0 (Hz) : {voiced_f0.min():.2f} .. "
          f"{voiced_f0.max():.2f} Hz  (n={voiced_f0.size})")
    if measured_voiced.size:
        print(f"3. measured on waveform   : {np.median(measured_voiced):.2f} Hz "
              f"(median of {measured_voiced.size} voiced frames, "
              f"{measured_voiced.min():.1f}-{measured_voiced.max():.1f} Hz)")
        ratio = np.median(measured_voiced) / np.median(voiced_f0)
        print(f"   measured / vocoder     : {ratio:.4f}"
              f"  ({12 * np.log2(ratio):+.2f} semitones)")
    print("diagnostics:")
    for message in result.diagnostics:
        print(f"   - {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
