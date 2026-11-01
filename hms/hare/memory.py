"""A small, optional store for individual HMS acoustic observations.

The values are the *existing*, unnormalized static mel-cepstral and
aperiodicity slices of :class:`hms.core.features.FeatureSpec`.  Pitch and
dynamic streams are deliberately not part of this memory.  See
``docs/hare_memory.md`` for the API, archive schema, and its limitations.
"""

from __future__ import annotations

import json
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Optional

import numpy as np

from hms.core.features import FeatureSpec

_FORMAT = "hms.hare.acoustic_memory"
_VERSION = 1
_MEMBERS = ("meta", "ids", "phonemes", "states", "mcep", "aperiodicity", "contexts")

# Version 1 records the whole existing FeatureSpec, not just its dimensions:
# fs, FFT size and mcep_f_min, for example, also determine what the values mean.
_SPEC_INTS = ("fs", "fft_size", "n_mcep", "n_band", "pitch_bins_per_semitone",
              "delta_window")
_SPEC_FLOATS = ("frame_period", "f0_ref_hz", "f0_floor", "f0_ceil",
                "voiced_threshold", "mcep_f_min")
_SPEC_BOOLS = ("use_delta", "use_delta2", "refine_f0")
_SPEC_STRINGS = ("f0_estimation",)
_SPEC_FIELDS = set(_SPEC_INTS + _SPEC_FLOATS + _SPEC_BOOLS + _SPEC_STRINGS)
_INT64_MAX = int(np.iinfo(np.int64).max)
_MAX_STATIC_DIM = int(np.iinfo(np.intp).max) // np.dtype("<f8").itemsize


def _text(value, name: str) -> str:
    """Nonempty identifiers/symbols that survive NumPy's Unicode padding."""
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{name} must be a nonempty string without NUL")
    return str(value)


def _state(value) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError("state must be a non-negative int64 HMM state index")
    # Convert first: older NumPy can round uint64/int64 comparisons to float.
    value = int(value)
    if not 0 <= value <= _INT64_MAX:
        raise ValueError("state must be a non-negative int64 HMM state index")
    return value


def _vector(values, name: str) -> np.ndarray:
    """Validate a real numeric vector and convert to the archive's float64."""
    try:
        array = np.asarray(values)
        if array.ndim != 1 or not array.size or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a nonempty 1-D real numeric vector")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(array, dtype="<f8")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite float64 values")
        return array
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real numeric vector") from exc


def _context(value) -> Mapping[str, str]:
    """Snapshot the string context supported by HMS's label layer."""
    if not isinstance(value, Mapping):
        raise ValueError("context must be a mapping of nonempty string keys to strings")
    if any(not isinstance(k, str) or not k or not isinstance(v, str)
           for k, v in value.items()):
        raise ValueError("context must be a mapping of nonempty string keys to strings")
    # Sort keys so iteration and serialization are independent of input order.
    return MappingProxyType(dict(sorted(value.items())))


def _spec_from_data(data: Mapping) -> FeatureSpec:
    """Validate and copy the exact version-1 FeatureSpec schema."""
    if not isinstance(data, Mapping) or set(data) != _SPEC_FIELDS:
        raise ValueError("feature_spec must contain exactly the version-1 fields")
    clean = {}
    for key in _SPEC_INTS:
        value = data[key]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise ValueError(f"feature_spec {key} must be an int64")
        value = int(value)
        if not -_INT64_MAX - 1 <= value <= _INT64_MAX:
            raise ValueError(f"feature_spec {key} must be an int64")
        clean[key] = value
    for key in _SPEC_FLOATS:
        value = data[key]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError(f"feature_spec {key} must be a finite number")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"feature_spec {key} must be a finite number") from exc
        if not np.isfinite(value):
            raise ValueError(f"feature_spec {key} must be a finite number")
        clean[key] = value
    for key in _SPEC_BOOLS:
        if not isinstance(data[key], (bool, np.bool_)):
            raise ValueError(f"feature_spec {key} must be a boolean")
        clean[key] = bool(data[key])
    for key in _SPEC_STRINGS:
        if not isinstance(data[key], str):
            raise ValueError(f"feature_spec {key} must be a string")
        clean[key] = str(data[key])
    spec = FeatureSpec(**clean)  # also enforces HMS's feature geometry rules
    _dimensions(spec)
    return spec


def _copy_spec(spec: FeatureSpec) -> FeatureSpec:
    if not isinstance(spec, FeatureSpec):
        raise ValueError("spec must be an HMS FeatureSpec")
    return _spec_from_data(spec.to_dict())


def _dimensions(spec: FeatureSpec) -> tuple[int, int]:
    """Check the dimensions needed to slice a frame, including mutated specs."""
    if not isinstance(spec, FeatureSpec):
        raise ValueError("spec must be an HMS FeatureSpec")
    dims = (spec.n_mcep, spec.n_band)
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
           for d in dims):
        raise ValueError("FeatureSpec n_mcep and n_band must be integers >= 2")
    n_mcep, n_band = map(int, dims)
    if n_mcep < 2 or n_band < 2 or n_mcep > _INT64_MAX or n_band > _INT64_MAX:
        raise ValueError("FeatureSpec n_mcep and n_band must be integers >= 2")
    if 1 + n_mcep + n_band > _MAX_STATIC_DIM:
        raise ValueError("FeatureSpec static dimensions exceed NumPy's addressable size")
    return n_mcep, n_band


@dataclass(frozen=True, eq=False)
class AcousticObservation:
    """One aligned frame's raw static acoustic features and label metadata.

    ``state`` is the zero-based index within the HMM used for this phoneme.
    ``context`` carries available string metadata (e.g. ``Segment.context``
    and/or neighboring phone symbols); it is not inferred by this class.
    Inputs are copied, feature vectors are read-only, and context is immutable.
    Feature widths are checked against the memory's :class:`FeatureSpec` on
    insertion.  Use :meth:`from_static` to slice an HMS-encoded frame directly.
    """

    observation_id: str
    phoneme: str
    state: int
    mcep: np.ndarray
    aperiodicity: np.ndarray
    context: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _text(self.observation_id, "observation_id"))
        object.__setattr__(self, "phoneme", _text(self.phoneme, "phoneme"))
        object.__setattr__(self, "state", _state(self.state))
        for name in ("mcep", "aperiodicity"):
            vector = _vector(getattr(self, name), name)
            # A bytes-backed vector cannot be made writable again by a caller.
            object.__setattr__(self, name, np.frombuffer(vector.tobytes(), dtype="<f8"))
        object.__setattr__(self, "context", _context(self.context))

    @classmethod
    def from_static(cls, observation_id: str, phoneme: str, state: int,
                    static: np.ndarray, spec: FeatureSpec,
                    context: Optional[Mapping[str, str]] = None
                    ) -> "AcousticObservation":
        """Slice one unnormalized ``FeatureSpec.encode`` static frame.

        The complete frame must be finite and have exactly ``spec.static_dim``
        entries.  The F0 slot is validated but not retained; delta streams are
        not accepted.  No extraction, normalization or alignment is performed.
        """
        n_mcep, n_band = _dimensions(spec)
        frame = _vector(static, "static frame")
        if frame.shape != (1 + n_mcep + n_band,):
            raise ValueError(f"static frame must have shape ({1 + n_mcep + n_band},), "
                             f"got {frame.shape}")
        return cls(observation_id, phoneme, state, frame[1:1 + n_mcep],
                   frame[1 + n_mcep:], {} if context is None else context)


def _json_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def _bad_json_constant(value):
    raise ValueError(f"invalid JSON constant {value}")


def _parse_json(text: str):
    return json.loads(text, object_pairs_hook=_json_object,
                      parse_constant=_bad_json_constant)


def _unicode_array(array: np.ndarray, shape: tuple, name: str) -> None:
    if (not isinstance(array, np.ndarray) or array.shape != shape
            or array.dtype.kind != "U" or array.dtype != array.dtype.newbyteorder("<")):
        raise ValueError(f"{name} must be a little-endian Unicode array of shape {shape}")
    # A malformed NPY can contain invalid UTF-32 code points. NumPy otherwise
    # raises a low-level SystemError when such a value is converted to str.
    codepoints = array.reshape(-1).view("<u4")
    if np.any(codepoints > 0x10FFFF):
        raise ValueError(f"{name} contains invalid Unicode code points")


def _numeric_array(array: np.ndarray, shape: tuple, dtype: str, name: str) -> None:
    if (not isinstance(array, np.ndarray) or array.shape != shape
            or array.dtype != np.dtype(dtype)):
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")


class AcousticMemory:
    """Insertion-ordered, identifier-indexed observations for one FeatureSpec.

    This store does not perform nearest-neighbour search or participate in
    training or synthesis.  ``add_batch`` is atomic even if its iterable fails.
    A memory owns a snapshot of its spec and its observations are immutable.
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self._spec = _copy_spec(spec)
        self._observations: dict[str, AcousticObservation] = {}

    @property
    def spec(self) -> FeatureSpec:
        """A copy of the FeatureSpec that defines this memory's geometry."""
        return _copy_spec(self._spec)

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self) -> Iterator[AcousticObservation]:
        return iter(self._observations.values())

    def get(self, observation_id: str) -> AcousticObservation:
        """Return an observation by ID; raise ``KeyError`` if absent."""
        return self._observations[observation_id]

    def add(self, observation: AcousticObservation) -> None:
        """Add one observation, rejecting duplicate IDs or incompatible widths."""
        self.add_batch((observation,))

    def add_batch(self, observations: Iterable[AcousticObservation]) -> None:
        """Add all observations in order, or leave the memory unchanged on error."""
        pending: dict[str, AcousticObservation] = {}
        for observation in observations:
            if not isinstance(observation, AcousticObservation):
                raise ValueError("expected AcousticObservation instances")
            if observation.mcep.shape != (self._spec.n_mcep,):
                raise ValueError(f"mcep must have shape ({self._spec.n_mcep},), "
                                 f"got {observation.mcep.shape}")
            if observation.aperiodicity.shape != (self._spec.n_band,):
                raise ValueError(f"aperiodicity must have shape ({self._spec.n_band},), "
                                 f"got {observation.aperiodicity.shape}")
            key = observation.observation_id
            if key in self._observations or key in pending:
                raise ValueError(f"duplicate observation_id {key!r}")
            pending[key] = observation
        self._observations.update(pending)

    def filter(self, phoneme: Optional[str] = None,
               state: Optional[int] = None) -> list[AcousticObservation]:
        """Return insertion-ordered exact phoneme/state matches; None is a wildcard."""
        if phoneme is not None:
            phoneme = _text(phoneme, "phoneme")
        if state is not None:
            state = _state(state)
        return [obs for obs in self if (phoneme is None or obs.phoneme == phoneme)
                and (state is None or obs.state == state)]

    def save(self, path) -> Path:
        """Save a compressed, pickle-free version-1 NPZ; return its actual path.

        A missing ``.npz`` suffix is appended, like NumPy and HMS source PCA.
        """
        path = Path(path)
        if not str(path).endswith(".npz"):
            path = Path(str(path) + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        observations = list(self)
        count = len(observations)
        meta = json.dumps({"format": _FORMAT, "version": _VERSION, "count": count,
                           "feature_spec": self._spec.to_dict()},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
        mcep = (np.stack([o.mcep for o in observations]).astype("<f8", copy=False)
                if count else np.empty((0, self._spec.n_mcep), dtype="<f8"))
        aperiodicity = (np.stack([o.aperiodicity for o in observations])
                        .astype("<f8", copy=False) if count
                        else np.empty((0, self._spec.n_band), dtype="<f8"))
        contexts = [json.dumps(dict(o.context), sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False) for o in observations]
        # All members are numeric or Unicode: no object arrays / pickles.
        np.savez_compressed(
            path, meta=np.asarray(meta, dtype="<U"),
            ids=np.asarray([o.observation_id for o in observations], dtype="<U"),
            phonemes=np.asarray([o.phoneme for o in observations], dtype="<U"),
            states=np.asarray([o.state for o in observations], dtype="<i8"),
            mcep=mcep, aperiodicity=aperiodicity,
            contexts=np.asarray(contexts, dtype="<U"))
        return path

    @classmethod
    def load(cls, path, *, expected_spec: Optional[FeatureSpec] = None
             ) -> "AcousticMemory":
        """Load and validate an NPZ without pickle; reject incompatible files.

        If supplied, ``expected_spec`` must match the entire recorded spec.
        A nonexistent/unreadable path raises the usual ``OSError``; malformed
        or incompatible archives raise ``ValueError`` with the path and cause.
        """
        expected = _copy_spec(expected_spec) if expected_spec is not None else None
        path = Path(path)
        # Open outside the error wrapper so missing/permission errors stay OSError.
        with path.open("rb") as stream:
            try:
                archive = np.load(stream, allow_pickle=False)
                if not isinstance(archive, np.lib.npyio.NpzFile):
                    raise ValueError("expected an NPZ archive")
                with archive:
                    if len(archive.files) != len(_MEMBERS) or set(archive.files) != set(_MEMBERS):
                        raise ValueError(f"expected archive members {', '.join(_MEMBERS)}")
                    # Reject unsupported metadata before materializing the frame arrays.
                    meta_array = archive["meta"]
                    _unicode_array(meta_array, (), "meta")
                    meta = _parse_json(meta_array.item())
                    if (not isinstance(meta, dict)
                            or set(meta) != {"format", "version", "count", "feature_spec"}
                            or meta["format"] != _FORMAT
                            or type(meta["version"]) is not int or meta["version"] != _VERSION):
                        raise ValueError("unrecognized acoustic memory format or version")
                    count = meta["count"]
                    if type(count) is not int or count < 0:
                        raise ValueError("metadata count must be a non-negative integer")
                    spec = _spec_from_data(meta["feature_spec"])
                    if expected is not None and spec.to_dict() != expected.to_dict():
                        raise ValueError("feature_spec does not match expected_spec")
                    arrays = {key: archive[key] for key in _MEMBERS[1:]}

                for key in ("ids", "phonemes", "contexts"):
                    _unicode_array(arrays[key], (count,), key)
                _numeric_array(arrays["states"], (count,), "<i8", "states")
                _numeric_array(arrays["mcep"], (count, spec.n_mcep), "<f8", "mcep")
                _numeric_array(arrays["aperiodicity"], (count, spec.n_band),
                               "<f8", "aperiodicity")
                if (not np.isfinite(arrays["mcep"]).all()
                        or not np.isfinite(arrays["aperiodicity"]).all()):
                    raise ValueError("acoustic features must contain only finite values")
                result = cls(spec)
                result.add_batch(
                    AcousticObservation(str(arrays["ids"][i]), str(arrays["phonemes"][i]),
                                        arrays["states"][i], arrays["mcep"][i],
                                        arrays["aperiodicity"][i],
                                        _parse_json(str(arrays["contexts"][i])))
                    for i in range(count))
                return result
            except (ValueError, TypeError, KeyError, EOFError, OSError, OverflowError,
                    zipfile.BadZipFile, zlib.error, RuntimeError) as exc:
                raise ValueError(f"invalid HARE acoustic memory {path}: {exc}") from exc
