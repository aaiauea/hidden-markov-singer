"""Optional pitch-conditioned acoustic model selection.

HMS's *pitch* is already note-driven: the score supplies target F0 and the
acoustic model only carries an optional note-relative deviation
(`hms.core.pitch`).  What the score does **not** currently influence is which
acoustic distribution a phone is rendered from: every observation of a phone is
pooled into one HMM, whichever note it happened to be sung on.  For a voice
recorded across a wide range -- or for an instrument-like source whose spectral
envelope follows its pitch -- that pooling averages envelopes that were never
produced together.

This module owns the *condition* itself and nothing else: one deterministic
integer per scored note, plus the policy deciding when a segment carries a
condition at all.  It is deliberately tiny, because the point of the feature is
model **selection**, not a new pitch model:

* training groups each observation into the bucket ``(unit, pitch_bin)`` of the
  note it was sung on (see `Trainer.train_pitch_models`),
* synthesis asks for the bucket of the note it was *given*
  (see `HMSModel.resolve_unit`),
* both sides call the same function in this module, so a bin index always means
  the same semitone range.

Binning convention
------------------
A pitch bin is a fixed-width band of the MIDI note range::

    pitch_bin = floor(note / bin_size)          (bin_size in semitones)

computed in exact integer arithmetic on a fixed-point copy of the note
(`NOTE_RESOLUTION`, hundredths of a semitone), never on a floating-point
division.  With the default ``bin_size = 6`` (a tritone, two bins per octave):

    ==========================  =========================================
    MIDI note                   bin
    ==========================  =========================================
    0 .. 5                      0
    60 (C4) .. 65               10
    66 (F#4) .. 71              11
    127 (G9)                    21
    ==========================  =========================================

Properties this buys, and the reasons for them:

* **equal notes always map to the same bin** -- the mapping is a pure function
  of the note and the bin size, with no state, no corpus statistics and no
  randomness;
* **boundaries are exact** -- a note that is a multiple of ``bin_size`` is
  never pushed into the bin below by binary representation error.  A note is
  quantised to the nearest cent (`NOTE_RESOLUTION` steps per semitone) and the
  division is integer floor division, so two notes within half a cent of each
  other always share a bin and no bin edge depends on floating-point rounding;
* **the whole MIDI range stays supported** -- bins run from 0 at MIDI 0 to
  ``n_pitch_bins(bin_size) - 1`` at MIDI 127;
* **anything that is not a musical pitch has no condition** -- a missing note,
  a non-finite value or a note outside MIDI 0-127 returns ``None``, which means
  "use the ordinary, unconditioned model hierarchy" rather than "invent a bin".

What gets a condition
---------------------
`segment_pitch_bin` is the single policy used by both training and synthesis:

* the note is the segment's **scored note** (the label/score column), never a
  measured F0.  A phone sung on one note therefore keeps one condition even
  where its frames are unvoiced, its F0 wobbles, or the analyser failed: the
  musical note is the stable variable, and instantaneous F0 would fragment one
  note's data across bins;
* a segment with **no note** (``-`` in the label file: silence, unvoiced-only
  segments, rests, and the silence HMS inserts for label gaps) carries no
  condition.  ``default_note`` is *not* substituted for it: it is a guess used
  to render pitch, not an observation about the recording, and conditioning on
  it would invent a pitch region nobody sang;
* the inventory's **silence** phone is never conditioned, even if a label gives
  it a note: its acoustic content is not a function of the sung pitch.

Everything else (vowels, voiced and unvoiced consonants sung on a scored note)
is conditioned, which keeps the rule general rather than tuned to one voice.

Status
------
Experimental and **off by default** (`pitch_conditioning.enabled: false`).
Pitch-conditioned data is naturally sparse -- a voice that never sings a phone
above A4 has no high bin for it -- so the feature is only as good as the
corpus, and missing buckets fall back through the ordinary hierarchy instead of
failing.  Nothing here claims that conditioning improves audio quality; it
makes the pitch region an explicit, inspectable part of model selection so that
the question can be measured.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hms.core.labels import MIDI_NOTE_MAX, MIDI_NOTE_MIN

#: Default pitch-bin width, in semitones (a tritone: two bins per octave).
DEFAULT_BIN_SIZE = 6

#: Largest accepted bin width.  A bin wider than the whole MIDI range would put
#: every note in one condition, i.e. silently switch the feature off; the
#: configuration says so instead.
MAX_BIN_SIZE = 128

#: Notes are quantised to this many steps per semitone before the (integer)
#: bin division, so fractional detuning cannot straddle a boundary because of
#: floating-point representation error.  100 steps = one cent.
NOTE_RESOLUTION = 100

#: Unit ``bin_size`` is measured in; recorded in the model file so a saved
#: model can be interpreted without the configuration that produced it.
BIN_UNIT = "semitones"

#: Tier name of a dedicated-phone model, as used by `HMSModel.resolve_unit`.
KIND_PHONE = "phone"

#: Suffix that marks a tier as pitch-conditioned
#: (``phone`` -> ``phone+pitch``), so diagnostics and evaluation can tell the
#: two apart without a second return value.
PITCH_TIER_SUFFIX = "+pitch"


def pitch_tier(kind: str) -> str:
    """Tier name of the pitch-conditioned version of ``kind``."""
    return f"{kind}{PITCH_TIER_SUFFIX}"


def is_pitch_tier(tier: Optional[str]) -> bool:
    """Whether a resolved tier name came from a pitch-conditioned model."""
    return bool(tier) and str(tier).endswith(PITCH_TIER_SUFFIX)


def validate_bin_size(bin_size: Any) -> int:
    """A checked pitch-bin width in semitones.

    Accepts anything that is exactly a whole number of semitones in
    ``1 .. MAX_BIN_SIZE``; everything else is a configuration error and says so
    rather than being rounded into a bin width nobody asked for.
    """
    try:
        numeric = float(bin_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("pitch_conditioning bin_size must be a number of "
                         f"semitones, got {bin_size!r}") from exc
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ValueError("pitch_conditioning bin_size must be a whole number "
                         f"of semitones, got {bin_size!r}")
    size = int(numeric)
    if size < 1 or size > MAX_BIN_SIZE:
        raise ValueError("pitch_conditioning bin_size must be between 1 and "
                         f"{MAX_BIN_SIZE} semitones, got {bin_size!r}")
    return size


def n_pitch_bins(bin_size: int = DEFAULT_BIN_SIZE) -> int:
    """How many bins cover the MIDI range 0-127 at this bin width."""
    return int(MIDI_NOTE_MAX) // validate_bin_size(bin_size) + 1


def pitch_bin(note: Optional[float], bin_size: int = DEFAULT_BIN_SIZE
              ) -> Optional[int]:
    """The deterministic pitch bin of a MIDI note (``None`` when it has none).

    ``floor(note / bin_size)`` over the MIDI range, evaluated as integer floor
    division on the note quantised to `NOTE_RESOLUTION` steps per semitone, so
    the result never depends on floating-point rounding at a bin boundary.
    Equal notes always give the equal bin, and the function is pure: the same
    ``(note, bin_size)`` gives the same answer in training and in synthesis.

    Returns ``None`` -- "no pitch condition", i.e. use the unconditioned model
    hierarchy -- for a missing note, a value that is not a real number, a
    non-finite value, or a note outside MIDI 0-127.  It never raises for the
    note itself, because an unusable pitch is a normal case (silence, rests,
    unvoiced segments), not a programming error; an invalid ``bin_size`` *is* a
    configuration error and raises `ValueError`.
    """
    width = validate_bin_size(bin_size) * NOTE_RESOLUTION
    if not isinstance(note, numbers.Real):
        return None
    try:
        value = float(note)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value < MIDI_NOTE_MIN or value > MIDI_NOTE_MAX:
        return None
    return int(round(value * NOTE_RESOLUTION)) // width


def bin_note_bounds(index: int, bin_size: int = DEFAULT_BIN_SIZE
                    ) -> Tuple[int, int]:
    """Inclusive integer MIDI notes covered by one bin.

    Documentation/inspection helper: the bin of `pitch_bin`, read back as the
    notes it stands for.  The top bin is clipped to MIDI 127 when the range is
    not an exact multiple of ``bin_size``.
    """
    size = validate_bin_size(bin_size)
    low = max(int(index) * size, int(MIDI_NOTE_MIN))
    high = min(low + size - 1, int(MIDI_NOTE_MAX))
    return low, high


def effective_note(note: Optional[float], transpose: float = 0.0
                   ) -> Optional[float]:
    """The MIDI note a scored segment is actually rendered at.

    Score note plus transposition, clipped into the valid MIDI range exactly
    like `Synthesizer._valid_midi_notes` does for the rendered pitch, so the
    condition is computed from the note the listener hears and not from a value
    HMS is about to correct.  ``None`` (no note) stays ``None``.
    """
    if note is None:
        return None
    value = float(note) + float(transpose)
    if not math.isfinite(value):
        return value
    return min(max(value, MIDI_NOTE_MIN), MIDI_NOTE_MAX)


def is_silence(phoneme_set, phone: str) -> bool:
    """Whether a phone is the inventory's silence (by symbol or by class)."""
    canonical = phoneme_set.canonical(phone)
    if canonical == phoneme_set.silence:
        return True
    definition = phoneme_set.resolve(canonical)
    return definition is not None and definition.type == "silence"


def segment_pitch_bin(phone: str, note: Optional[float], phoneme_set,
                      bin_size: int = DEFAULT_BIN_SIZE) -> Optional[int]:
    """The pitch condition of one labelled/scored segment, or ``None``.

    The single policy shared by training and synthesis (see the module
    docstring): a segment is conditioned on its **scored note** when it has one
    and its phone is not silence; everything else -- unnoted segments, rests,
    inserted silence, unvoiced-only spans, invalid notes -- has no condition
    and resolves through the ordinary, unconditioned hierarchy.
    """
    if note is None or is_silence(phoneme_set, phone):
        return None
    return pitch_bin(note, bin_size)


@dataclass(frozen=True)
class PitchConditioning:
    """How a model's pitch conditions are defined.

    Stored in ``model.yaml`` so a saved model can interpret itself: synthesis
    never needs the training configuration to know whether conditioning was on
    or how wide a bin is.
    """

    enabled: bool = False
    bin_size: int = DEFAULT_BIN_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bin_size", validate_bin_size(self.bin_size))
        object.__setattr__(self, "enabled", bool(self.enabled))

    @property
    def active(self) -> bool:
        """Whether this model asks for pitch-conditioned lookups at all."""
        return bool(self.enabled)

    def bin_of(self, note: Optional[float]) -> Optional[int]:
        """The bin of a note under *this* model's bin size."""
        return pitch_bin(note, self.bin_size)

    def n_bins(self) -> int:
        return n_pitch_bins(self.bin_size)

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": bool(self.enabled), "bin_size": int(self.bin_size),
                "bin_unit": BIN_UNIT, "n_bins": self.n_bins()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PitchConditioning":
        data = dict(data or {})
        unknown = set(data) - {"enabled", "bin_size", "bin_unit", "n_bins"}
        if unknown:
            raise ValueError(f"unknown pitch conditioning options: "
                             f"{sorted(unknown)}")
        return cls(enabled=bool(data.get("enabled", False)),
                   bin_size=int(data.get("bin_size", DEFAULT_BIN_SIZE)))

    def describe(self) -> str:
        """One-line description for logs and diagnostics."""
        if not self.enabled:
            return "pitch conditioning: disabled"
        return (f"pitch conditioning: {self.n_bins()} bins of "
                f"{self.bin_size} semitone(s) over MIDI "
                f"{int(MIDI_NOTE_MIN)}-{int(MIDI_NOTE_MAX)}")
