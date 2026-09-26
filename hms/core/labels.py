"""Label files: the score, and the supervision for training.

One format serves both roles, which keeps the toolchain small::

    # utt_id   onset   offset  phone   note
    scale_01   0.000   0.121   s       -
    scale_01   0.121   0.180   i       60
    scale_01   0.180   0.520   i       60
    ...

* ``onset``/``offset`` are in seconds by default (``time_unit: seconds`` in
  `parameters.yaml`); ``time_unit: frames`` interprets them as analysis frames,
  which is exact and convenient when labels were derived from the same frame
  grid.
* ``phone`` is a phoneme symbol from `phonemes.yaml`.
* ``note`` is a MIDI note number, or ``-`` (or empty) for "no pitch": silence
  and unvoiced-only segments.  A note may be repeated across several
  consecutive rows to keep the phoneme sequence fine-grained while the note
  stays put, as above.

Rows are grouped by utterance id; ``utterance_order`` is preserved from the
file.  Overlaps, gaps and malformed rows are tolerated but reported in
`Score.diagnostics <hms.core.labels.Score>`; a gap in the middle of an
utterance keeps the previous note and the utterance's first phoneme, because
the label layer has no notion of silence -- callers that do (`Trainer`,
`Synthesizer`) insert explicit silence for uncovered frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

#: Symbols that mean "no note".
NO_NOTE = {"-", "", "nan", "none", "rest"}


def midi_to_hz(note: float) -> float:
    """MIDI note number -> Hz (A4 = 440 Hz = MIDI 69)."""
    return 440.0 * 2.0 ** ((np.asarray(note, dtype=np.float64) - 69.0) / 12.0)


def hz_to_midi(f: float | np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-9) / 440.0)


@dataclass
class Segment:
    """One labelled segment: a phoneme held over a time span, on one note."""

    phone: str
    start: float
    end: float
    note: Optional[float] = None
    utterance: str = ""
    context: Dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def note_or(self, fallback: float) -> float:
        return fallback if self.note is None else float(self.note)


@dataclass
class Utterance:
    """An ordered list of segments sharing an utterance id."""

    name: str
    segments: List[Segment] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def start(self) -> float:
        return min((s.start for s in self.segments), default=0.0)

    @property
    def end(self) -> float:
        return max((s.end for s in self.segments), default=0.0)

    def phones(self) -> List[str]:
        return [s.phone for s in self.segments]

    def notes(self) -> List[Optional[float]]:
        return [s.note for s in self.segments]

    def frame_labels(self, frame_period: float, default_note: float,
                     n_frames: Optional[int] = None) -> np.ndarray:
        """Per-frame phoneme index + note, as an (T,) object-free structure.

        Returns a tuple ``(phones, notes, starts)`` where ``phones`` is a list
        of phoneme symbols per frame (gaps filled with the first phoneme) and
        ``notes`` a float array of MIDI notes (carried forward across gaps).
        """
        hop = frame_period / 1000.0
        total = int(round((self.end - self.start) / hop)) if n_frames is None \
            else int(n_frames)
        total = max(total, 1)
        phones: List[str] = [self.segments[0].phone if self.segments else "sil"] * total
        notes = np.full(total, float(default_note))
        covered = np.zeros(total, dtype=bool)
        for segment in self.segments:
            lo = int(round((segment.start - self.start) / hop))
            hi = int(round((segment.end - self.start) / hop))
            lo, hi = max(0, min(lo, total - 1)), max(1, min(hi, total))
            for t in range(lo, hi):
                phones[t] = segment.phone
                notes[t] = segment.note_or(default_note)
            covered[lo:hi] = True
        if not covered.all():                      # gaps: carry the previous note
            last = float(default_note)
            for t in range(total):
                if covered[t]:
                    last = notes[t]
                else:
                    notes[t] = last
        return phones, notes, self.start


class Score:
    """A parsed label file: named utterances in file order."""

    def __init__(self, utterances: List[Utterance],
                 diagnostics: Optional[List[str]] = None) -> None:
        self.utterances = utterances
        self.diagnostics = diagnostics or []

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self) -> Iterable[Utterance]:
        return iter(self.utterances)

    def __getitem__(self, key: str) -> Utterance:
        for utterance in self.utterances:
            if utterance.name == key:
                return utterance
        raise KeyError(key)

    @property
    def phone_count(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for utterance in self.utterances:
            for segment in utterance.segments:
                counts[segment.phone] = counts.get(segment.phone, 0) + 1
        return counts

    @property
    def total_duration(self) -> float:
        return float(sum(u.end - u.start for u in self.utterances))

    def to_lines(self) -> List[str]:
        lines = ["# utt_id\tonset\toffset\tphone\tnote"]
        for utterance in self.utterances:
            for segment in utterance.segments:
                note = "-" if segment.note is None else f"{segment.note:g}"
                lines.append(f"{utterance.name}\t{segment.start:.4f}\t"
                             f"{segment.end:.4f}\t{segment.phone}\t{note}")
        return lines

    def save(self, path) -> None:
        Path(path).write_text("\n".join(self.to_lines()) + "\n", encoding="utf-8")


def parse(text: str, time_unit: str = "seconds") -> Score:
    """Parse label text into a :class:`Score`."""
    if time_unit not in ("seconds", "frames"):
        raise ValueError("time_unit must be 'seconds' or 'frames'")
    utterances: List[Utterance] = []
    by_name: Dict[str, Utterance] = {}
    diagnostics: List[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 4:
            diagnostics.append(f"line {lineno}: expected at least 4 columns, "
                               f"got {len(parts)}")
            continue
        name, onset, offset, phone = parts[0], parts[1], parts[2], parts[3]
        note_raw = parts[4].strip().lower() if len(parts) > 4 else "-"
        try:
            start, end = float(onset), float(offset)
        except ValueError:
            diagnostics.append(f"line {lineno}: bad time columns {onset!r}, "
                               f"{offset!r}")
            continue
        if end <= start:
            diagnostics.append(f"line {lineno}: non-positive duration "
                               f"({end - start:+.4f}) -- dropped")
            continue
        note: Optional[float]
        note = None if note_raw in NO_NOTE else float(note_raw)

        context: Dict[str, str] = {}
        for extra in parts[5:]:
            if "=" in extra:
                key, value = extra.split("=", 1)
                context[key] = value

        utterance = by_name.get(name)
        if utterance is None:
            utterance = Utterance(name=name)
            by_name[name] = utterance
            utterances.append(utterance)
        utterance.segments.append(Segment(phone=phone, start=start, end=end,
                                          note=note, utterance=name,
                                          context=context))

    for utterance in utterances:
        utterance.segments.sort(key=lambda s: s.start)
        for previous, current in zip(utterance.segments, utterance.segments[1:]):
            if current.start < previous.end - 1e-9:
                diagnostics.append(
                    f"{utterance.name}: overlap between {previous.phone} "
                    f"({previous.start:.3f}-{previous.end:.3f}) and "
                    f"{current.phone} ({current.start:.3f}-{current.end:.3f})")
    return Score(utterances, diagnostics)


def load(path, time_unit: str = "seconds") -> Score:
    return parse(Path(path).read_text(encoding="utf-8"), time_unit=time_unit)


def note_sequence(utterance: Utterance, frame_period: float,
                  default_note: float,
                  n_frames: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
    """(per-frame phone symbols, per-frame MIDI notes) for one utterance."""
    phones, notes, _ = utterance.frame_labels(frame_period, default_note,
                                             n_frames=n_frames)
    return phones, notes


def segment_boundaries(utterance: Utterance, frame_period: float,
                       n_frames: Optional[int] = None
                       ) -> List[Tuple[str, int, int, Optional[float]]]:
    """Segments as (phone, start_frame, end_frame, note), clipped to the file."""
    hop = frame_period / 1000.0
    out: List[Tuple[str, int, int, Optional[float]]] = []
    for segment in utterance.segments:
        lo = int(round((segment.start - utterance.start) / hop))
        hi = int(round((segment.end - utterance.start) / hop))
        if n_frames is not None:
            lo, hi = max(0, min(lo, n_frames)), max(0, min(hi, n_frames))
        if hi > lo:
            out.append((segment.phone, lo, hi, segment.note))
    return out
