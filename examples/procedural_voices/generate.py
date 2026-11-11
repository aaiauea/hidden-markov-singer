"""Deterministic, source-only synthetic HMS example corpora.

Run from the repository root: python examples/procedural_voices/generate.py
Uses the repository's existing DemoSinger and HMS five-column labels; generated
WAVs live in ignored output directories. No external audio is used.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import yaml

from hms.data.demo_singer import DemoSinger, SegmentSpec, SingerConfig
from hms.data.wavio import write_wav

ROOT = Path(__file__).resolve().parent
FS = 22050
# HMS's DemoSinger currently has a small articulatory inventory (a/e/i/o/u,
# m/n/l/r/v/z/s/f/k/t/p/h, sil); retain these atomic units rather than morae.
INVENTORY = {
    "ja": ["a", "i", "u", "e", "o", "m", "n", "r", "s", "f", "k", "t", "p", "h", "sil"],
    "en": ["a", "e", "i", "o", "u", "m", "n", "l", "r", "v", "z", "s", "f", "k", "t", "p", "h", "sil"],
}
VOICES = {
    "ja_bright": ("ja", 0, 75, -15),
    "ja_dark": ("ja", 1, 55, -5),
    "en_bright": ("en", 2, 75, -15),
    "en_dark": ("en", 3, 55, -5),
}

def script(phones, seed, base):
    rng = np.random.default_rng(seed)
    result = []
    # Coverage schedule crosses every phone with pitches and varied durations.
    voiced = [p for p in phones if p != "sil"]
    for block in range(8):
        result.append(SegmentSpec("sil", 110, None))
        order = voiced[block % len(voiced):] + voiced[:block % len(voiced)]
        for j, phone in enumerate(order):
            # Every phone recurs at several notes; contexts vary through rotation.
            note = base + ((j * 3 + block * 5) % 19) - 9
            duration = float((180, 260, 420, 310)[(j + block) % 4])
            result.append(SegmentSpec(phone, duration, note))
            if j % 3 == 1:
                result.append(SegmentSpec("sil", 45, None))
    result.append(SegmentSpec("sil", 180, None))
    return result

def main():
    for name, (language, seed, base, tilt) in VOICES.items():
        out = ROOT / "generated" / name
        wav_dir = out / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        phones = INVENTORY[language]
        # Independent source/noise/vibrato/tilt and register settings.
        config = SingerConfig(fs=FS, seed=seed, tilt_db_per_octave=tilt,
                              vibrato_semitones=.28 if "bright" in name else .18,
                              jitter=.003 if "bright" in name else .006,
                              scoop_semitones=-.45 if "bright" in name else -.2)
        singer = DemoSinger(config)
        rows = ["# utt_id\tonset\toffset\tphone\tnote"]
        total = 0.
        # Split into four phrase boundaries / utterances.
        full = script(phones, seed, base)
        chunks = np.array_split(np.arange(len(full)), 4)
        for q, ids in enumerate(chunks):
            segments = [full[int(i)] for i in ids]
            utt = f"take_{q+1}"
            audio = singer.render(segments)
            write_wav(wav_dir / f"{utt}.wav", audio, FS)
            cursor = 0.
            for seg in segments:
                start = cursor / 1000
                cursor += seg.duration_ms
                note = "-" if seg.note is None else f"{seg.note:g}"
                rows.append(f"{utt}\t{start:.4f}\t{cursor/1000:.4f}\t{seg.phone}\t{note}")
            total += len(audio) / FS
        (out / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (out / "voice.json").write_text(json.dumps({"voice": name, "language": language,
            "seed": seed, "sample_rate": FS, "singer_config": config.to_dict(),
            "seconds": round(total, 3), "phones": phones}, indent=2) + "\n")
        digest = hashlib.sha256((out / "labels.tsv").read_bytes()).hexdigest()
        print(f"{name}: {total:.1f}s, {len(rows)-1} exact-timing labels, labels sha256={digest}")

if __name__ == "__main__":
    main()
