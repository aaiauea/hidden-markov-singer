"""Render the checked-in HMS labels as tiny, deterministic training audio.

Uses HMS's existing demo-only synthetic singer, not a pretrained voice. All
paths are relative to this file; the generated WAV lives under ignored out/.
"""

from pathlib import Path

from hms.config import load_parameters, training_config_from_parameters
from hms.core import labels
from hms.data.demo_singer import DemoSinger, SegmentSpec, SingerConfig
from hms.data.wavio import write_wav


def main() -> None:
    directory = Path(__file__).resolve().parent
    config = training_config_from_parameters(
        load_parameters(directory / "parameters.yaml"))
    corpus = labels.load(directory / "labels.tsv", time_unit=config.time_unit,
                         frame_period=config.frame_period)
    singer = DemoSinger(SingerConfig(
        fs=config.fs, seed=0, jitter=0.0, vibrato_semitones=0.0,
        scoop_semitones=0.0, drift_semitones=0.0))
    for utterance in corpus:
        # These labels are contiguous and start at zero; render each duration.
        script = [SegmentSpec(segment.phone,
                              (segment.end - segment.start) * 1000.0,
                              segment.note)
                  for segment in utterance.segments]
        path = directory / "out" / "wav" / f"{utterance.name}.wav"
        write_wav(path, singer.render(script), config.fs)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
