"""Optional HARE observation storage and its standalone NPZ format."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from hms.core.context import neighbor_contexts
from hms.core.features import FeatureSpec, add_dynamic_features
from hms.core.labels import Segment
from hms.hare import AcousticMemory, AcousticObservation


@pytest.fixture
def spec():
    return FeatureSpec(fs=22050, fft_size=1024, n_mcep=4, n_band=3,
                       frame_period=5.0, mcep_f_min=50.0, use_delta2=True)


def observation(key="utt:0", phone="a", state=0, *, mcep=None, ap=None,
                context=None):
    return AcousticObservation(
        observation_id=key, phoneme=phone, state=state,
        mcep=np.arange(4, dtype=float) if mcep is None else mcep,
        aperiodicity=[0.1, 0.2, 0.3] if ap is None else ap,
        context={} if context is None else context)


def assert_observation_equal(actual, expected):
    assert actual.observation_id == expected.observation_id
    assert actual.phoneme == expected.phoneme
    assert actual.state == expected.state
    assert dict(actual.context) == dict(expected.context)
    assert np.array_equal(actual.mcep, expected.mcep)
    assert np.array_equal(actual.aperiodicity, expected.aperiodicity)
    assert actual.mcep.dtype == np.dtype("<f8")
    assert actual.aperiodicity.dtype == np.dtype("<f8")


def archive_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def replace_meta(arrays, **changes):
    meta = json.loads(arrays["meta"].item())
    meta.update(changes)
    arrays["meta"] = np.asarray(json.dumps(meta))


def test_empty_memory_roundtrip_and_schema(spec, tmp_path):
    memory = AcousticMemory(spec)
    assert len(memory) == 0
    assert list(memory) == []
    assert memory.filter() == []
    assert memory.filter("a", 0) == []
    with pytest.raises(KeyError):
        memory.get("missing")

    path = memory.save(tmp_path / "nested" / "empty")
    assert path.name == "empty.npz" and path.exists()
    arrays = archive_arrays(path)
    assert set(arrays) == {"meta", "ids", "phonemes", "states", "mcep",
                           "aperiodicity", "contexts"}
    assert arrays["mcep"].shape == (0, spec.n_mcep)
    assert arrays["aperiodicity"].shape == (0, spec.n_band)
    assert arrays["states"].shape == arrays["ids"].shape == (0,)
    assert all(array.dtype.kind != "O" for array in arrays.values())
    meta = json.loads(arrays["meta"].item())
    assert meta["format"] == "hms.hare.acoustic_memory"
    assert meta["version"] == 1 and meta["count"] == 0
    assert meta["feature_spec"] == spec.to_dict()
    loaded = AcousticMemory.load(path, expected_spec=spec)
    assert len(loaded) == 0 and loaded.spec.to_dict() == spec.to_dict()
    assert loaded.save(tmp_path / "empty-again.npz").read_bytes() == path.read_bytes()


def test_observation_uses_existing_feature_spec_and_context(spec):
    """No alternate extraction or derived representation is needed."""
    f0 = np.array([220.0, 233.0])
    sp = np.full((2, spec.n_bins), 0.01)
    ap = np.full((2, spec.n_bins), 0.4)
    static = spec.encode(f0, sp, ap)
    pre, phone, post = list(neighbor_contexts(["a", "n"], "sil"))[0]
    segment = Segment(phone, 0.0, 0.1, context={"style": "legato"})
    context = {**segment.context, "pre_phone": pre, "future_phone": post}
    item = AcousticObservation.from_static("take:0", phone, np.int64(1),
                                           static[0], spec, context)
    memory = AcousticMemory(spec)
    memory.add(item)
    assert memory.get("take:0") is item
    assert np.array_equal(item.mcep, static[0, 1:1 + spec.n_mcep])
    assert np.array_equal(item.aperiodicity, static[0, 1 + spec.n_mcep:])
    assert item.context == {"style": "legato", "pre_phone": "sil", "future_phone": "n"}
    assert item.state == 1
    assert len(memory) == 1

    # The helper accepts exactly one raw static frame, not a dynamic stream.
    dynamics = add_dynamic_features(static, use_delta=True, use_delta2=True)
    with pytest.raises(ValueError, match="static frame.*shape"):
        AcousticObservation.from_static("x", phone, 0, dynamics[0], spec)
    with pytest.raises(ValueError, match="static frame.*1-D"):
        AcousticObservation.from_static("x", phone, 0, static, spec)


def test_batch_insertion_get_iteration_and_filters(spec):
    items = [observation("first", "a", 0), observation("二", "a", 2),
             observation("third", "n", 0), observation("last", "n", 2)]
    memory = AcousticMemory(spec)
    memory.add(items[0])
    memory.add_batch(item for item in items[1:])
    memory.add_batch([])
    assert len(memory) == 4
    assert list(memory) == items
    assert memory.get("二") is items[1]
    assert memory.filter() == items
    assert memory.filter(phoneme="a") == items[:2]
    assert memory.filter(state=np.int64(0)) == [items[0], items[2]]
    assert memory.filter(phoneme="n", state=2) == [items[3]]
    assert memory.filter(phoneme="missing", state=0) == []
    assert memory.filter(phoneme="A") == []  # exact/case-sensitive
    with pytest.raises(KeyError):
        memory.get("not present")
    with pytest.raises(ValueError, match="state"):
        memory.filter(state=-1)


def test_ids_are_unique_and_batches_are_atomic(spec):
    memory = AcousticMemory(spec)
    memory.add(observation("existing"))
    with pytest.raises(ValueError, match="duplicate observation_id"):
        memory.add(observation("existing", "n"))
    with pytest.raises(ValueError, match="duplicate observation_id"):
        memory.add_batch([observation("new"), observation("existing")])
    assert [o.observation_id for o in memory] == ["existing"]
    with pytest.raises(ValueError, match="duplicate observation_id"):
        memory.add_batch([observation("new"), observation("new")])
    assert len(memory) == 1
    with pytest.raises(ValueError, match="mcep.*shape"):
        memory.add_batch([observation("new"), observation("bad", mcep=[1, 2])])
    assert len(memory) == 1
    with pytest.raises(ValueError, match="AcousticObservation"):
        memory.add_batch([observation("new"), object()])
    assert len(memory) == 1

    def broken_generator():
        yield observation("new")
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        memory.add_batch(broken_generator())
    assert len(memory) == 1


def test_inputs_and_spec_are_snapshotted_and_immutable(spec):
    mcep = np.arange(4, dtype=np.float64)
    ap = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    context = {"style": "soft", "pre_phone": "n"}
    item = observation(mcep=mcep, ap=ap, context=context)
    memory = AcousticMemory(spec)
    memory.add(item)
    mcep[:] = 42
    ap[:] = 0
    context["style"] = "changed"
    spec.n_mcep = 20
    returned_spec = memory.spec
    returned_spec.n_band = 99
    assert memory.spec.n_mcep == 4 and memory.spec.n_band == 3
    np.testing.assert_array_equal(memory.get("utt:0").mcep, np.arange(4))
    np.testing.assert_array_equal(item.aperiodicity, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert item.context["style"] == "soft"
    with pytest.raises(ValueError):
        item.mcep[0] = 10
    with pytest.raises(ValueError):
        item.mcep.setflags(write=True)
    with pytest.raises(TypeError):
        item.context["style"] = "changed"
    with pytest.raises(FrozenInstanceError):
        item.phoneme = "changed"


@pytest.mark.parametrize("field,bad", [
    ("mcep", [1, 2, 3]), ("mcep", [[1, 2, 3, 4]]),
    ("mcep", []), ("mcep", 1.0),
    ("aperiodicity", [0.2, 0.3]), ("aperiodicity", [[0.1, 0.2, 0.3]]),
])
def test_invalid_feature_dimensions(spec, field, bad):
    kwargs = dict(observation_id="x", phoneme="a", state=0,
                  mcep=[1, 2, 3, 4], aperiodicity=[0.1, 0.2, 0.3])
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        AcousticMemory(spec).add(AcousticObservation(**kwargs))


@pytest.mark.parametrize("field", ["mcep", "aperiodicity"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_are_rejected(spec, field, bad):
    values = np.zeros(spec.n_mcep if field == "mcep" else spec.n_band)
    values[1] = bad
    with pytest.raises(ValueError, match="finite"):
        observation(**({"mcep": values} if field == "mcep" else {"ap": values}))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_f0_slot_in_static_frame_is_rejected(spec, bad):
    frame = np.zeros(spec.static_dim)
    frame[0] = bad
    with pytest.raises(ValueError, match="finite"):
        AcousticObservation.from_static("x", "a", 0, frame, spec)


@pytest.mark.parametrize("bad", [np.array([1, 2, 3, 4], dtype=bool),
                                 np.array([1, 2, 3, 4], dtype=complex),
                                 np.array(["1", "2", "3", "4"]),
                                 np.array([1, 2, 3, 4], dtype=object)])
def test_feature_inputs_must_be_real_numeric_not_strings_or_objects(bad):
    with pytest.raises(ValueError, match="real numeric"):
        observation(mcep=bad)


@pytest.mark.parametrize("kwargs", [
    {"key": ""}, {"key": "a\0"}, {"phone": ""}, {"phone": "x\0y"},
    {"state": -1}, {"state": True}, {"state": 1.0}, {"state": 2**63},
    {"state": np.uint64(2**63)},
    {"context": []}, {"context": {"": "value"}},
    {"context": {"key": 1}}, {"context": {1: "value"}},
])
def test_invalid_identifiers_state_and_context(kwargs):
    with pytest.raises(ValueError):
        observation(**kwargs)


def test_invalid_feature_specs_are_not_silently_coerced(spec):
    with pytest.raises(ValueError, match="FeatureSpec"):
        AcousticMemory(None)
    spec.n_mcep = 4.5
    with pytest.raises(ValueError, match="n_mcep"):
        AcousticMemory(spec)
    with pytest.raises(ValueError, match="n_mcep"):
        AcousticObservation.from_static("x", "a", 0, np.zeros(8), spec)
    spec.n_mcep = np.uint64(2**63)
    with pytest.raises(ValueError, match="n_mcep"):
        AcousticMemory(spec)
    with pytest.raises(ValueError, match="n_mcep"):
        AcousticObservation.from_static("x", "a", 0, np.zeros(8), spec)
    spec.n_mcep = 2**63 - 1
    with pytest.raises(ValueError, match="addressable"):
        AcousticMemory(spec)


def test_roundtrip_preserves_order_values_spec_and_context(spec, tmp_path):
    items = [observation("歌:0", "a", 0, context={"z": "λ", "a": ""}),
             observation("take:1", "ɲ", 2,
                         mcep=np.array([-0.0, -2.5, 1e-20, 7.0]),
                         ap=[0.0, 0.6, 1.0],
                         context={"pre_phone": "a", "note": "quoted \"\n\0"}),
             observation("take:2", "sil", 0)]
    memory = AcousticMemory(spec)
    memory.add_batch(items)
    path = memory.save(tmp_path / "memory.npz")
    loaded = AcousticMemory.load(path, expected_spec=spec)
    assert loaded.spec.to_dict() == spec.to_dict()
    assert len(loaded) == 3
    for actual, expected in zip(loaded, items):
        assert_observation_equal(actual, expected)
        # Round trip is bit-exact, including the sign bit of zero.
        assert actual.mcep.tobytes() == expected.mcep.tobytes()
    assert [o.observation_id for o in loaded.filter("ɲ", 2)] == ["take:1"]
    loaded.add(observation("take:3"))
    assert len(memory) == 3 and len(loaded) == 4


def test_int64_state_boundary_roundtrips(spec, tmp_path):
    maximum = np.uint64(2**63 - 1)
    memory = AcousticMemory(spec)
    memory.add(observation(state=maximum))
    loaded = AcousticMemory.load(memory.save(tmp_path / "boundary.npz"))
    assert loaded.get("utt:0").state == int(maximum)
    assert len(loaded.filter(state=maximum)) == 1


def test_expected_feature_spec_rejects_same_width_different_geometry(spec, tmp_path):
    path = AcousticMemory(spec).save(tmp_path / "memory.npz")
    other = FeatureSpec(**spec.to_dict())
    other.fs = 44100
    with pytest.raises(ValueError, match="expected_spec"):
        AcousticMemory.load(path, expected_spec=other)
    assert AcousticMemory.load(path).spec.fs == 22050


def test_deterministic_serialization_and_canonical_context(spec, tmp_path):
    left, right = AcousticMemory(spec), AcousticMemory(spec)
    left.add_batch([observation("one", context={"z": "last", "a": "first"}),
                    observation("two", "n", 2)])
    right.add_batch([observation("one", context={"a": "first", "z": "last"}),
                     observation("two", "n", 2)])
    first = left.save(tmp_path / "one.npz")
    second = left.save(tmp_path / "two.npz")
    third = right.save(tmp_path / "three.npz")
    fourth = AcousticMemory.load(first).save(tmp_path / "four.npz")
    assert first.read_bytes() == second.read_bytes() == third.read_bytes() == fourth.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    assert list(left.get("one").context) == ["a", "z"]


@pytest.fixture
def valid_arrays(spec, tmp_path):
    memory = AcousticMemory(spec)
    memory.add_batch([observation("first", context={"a": "x"}), observation("second")])
    return archive_arrays(memory.save(tmp_path / "valid.npz"))


@pytest.mark.parametrize("case", [
    "missing", "extra", "meta_shape", "meta_dtype", "bad_json", "duplicate_json_key",
    "nonfinite_json", "wrong_format", "future_version", "boolean_version", "bad_count",
    "bad_spec_fields", "bad_spec_type", "bad_geometry", "ids_shape", "ids_dtype",
    "bad_unicode", "empty_id", "duplicate_id", "empty_phoneme", "states_dtype", "states_shape",
    "negative_state", "mcep_shape", "mcep_dtype", "ap_shape", "ap_dtype", "nan_mcep",
    "infinite_ap", "contexts_shape", "bad_context_json", "context_not_mapping",
    "context_nonstring", "duplicate_context_key", "object_array",
])
def test_corrupt_or_incompatible_archives_are_rejected(valid_arrays, tmp_path, case):
    arrays = valid_arrays
    if case == "missing":
        del arrays["mcep"]
    elif case == "extra":
        arrays["unexpected"] = np.zeros(1)
    elif case == "meta_shape":
        arrays["meta"] = arrays["meta"].reshape(1)
    elif case == "meta_dtype":
        arrays["meta"] = np.array(1, dtype="<i8")
    elif case == "bad_json":
        arrays["meta"] = np.array("{this is not json")
    elif case == "duplicate_json_key":
        arrays["meta"] = np.array('{"version":1,"version":1}')
    elif case == "nonfinite_json":
        arrays["meta"] = np.array('{"version":NaN}')
    elif case == "wrong_format":
        replace_meta(arrays, format="another-format")
    elif case == "future_version":
        replace_meta(arrays, version=2)
    elif case == "boolean_version":
        replace_meta(arrays, version=True)
    elif case == "bad_count":
        replace_meta(arrays, count=True)
    elif case in {"bad_spec_fields", "bad_spec_type", "bad_geometry"}:
        meta = json.loads(arrays["meta"].item())
        if case == "bad_spec_fields":
            del meta["feature_spec"]["mcep_f_min"]
        elif case == "bad_spec_type":
            meta["feature_spec"]["n_mcep"] = 4.0
        else:
            meta["feature_spec"]["n_band"] = 1
        arrays["meta"] = np.asarray(json.dumps(meta))
    elif case == "ids_shape":
        arrays["ids"] = arrays["ids"].reshape(2, 1)
    elif case == "ids_dtype":
        arrays["ids"] = np.array([b"first", b"second"])
    elif case == "bad_unicode":
        arrays["ids"].view("<u4")[0] = 0x110000
    elif case == "empty_id":
        arrays["ids"] = np.array(["", "second"])
    elif case == "duplicate_id":
        arrays["ids"] = np.array(["same", "same"])
    elif case == "empty_phoneme":
        arrays["phonemes"] = np.array(["a", ""])
    elif case == "states_dtype":
        arrays["states"] = arrays["states"].astype("float64")
    elif case == "states_shape":
        arrays["states"] = np.array([0], dtype="<i8")
    elif case == "negative_state":
        arrays["states"][1] = -1
    elif case == "mcep_shape":
        arrays["mcep"] = arrays["mcep"][:, :-1]
    elif case == "mcep_dtype":
        arrays["mcep"] = arrays["mcep"].astype("float32")
    elif case == "ap_shape":
        arrays["aperiodicity"] = arrays["aperiodicity"].reshape(-1)
    elif case == "ap_dtype":
        arrays["aperiodicity"] = arrays["aperiodicity"].astype(">f8")
    elif case == "nan_mcep":
        arrays["mcep"][0, 0] = np.nan
    elif case == "infinite_ap":
        arrays["aperiodicity"][1, 0] = np.inf
    elif case == "contexts_shape":
        arrays["contexts"] = np.array(["{}"])
    elif case == "bad_context_json":
        arrays["contexts"] = np.array(["{", "{}"])
    elif case == "context_not_mapping":
        arrays["contexts"] = np.array(["[]", "{}"])
    elif case == "context_nonstring":
        arrays["contexts"] = np.array(['{"x":1}', "{}"])
    elif case == "duplicate_context_key":
        arrays["contexts"] = np.array(['{"x":"a","x":"b"}', "{}"])
    elif case == "object_array":
        arrays["ids"] = np.array(["first", "second"], dtype=object)
    path = tmp_path / f"{case}.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="invalid HARE acoustic memory"):
        AcousticMemory.load(path)


def test_corrupt_file_types_truncation_and_missing_file(spec, tmp_path):
    not_zip = tmp_path / "invalid.npz"
    not_zip.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="invalid HARE acoustic memory"):
        AcousticMemory.load(not_zip)
    npy = tmp_path / "array.npy"
    np.save(npy, np.array([1, 2]))
    with pytest.raises(ValueError, match="expected an NPZ"):
        AcousticMemory.load(npy)
    truncated = tmp_path / "truncated.npz"
    valid = AcousticMemory(spec).save(tmp_path / "valid.npz").read_bytes()
    truncated.write_bytes(valid[:len(valid) // 2])
    with pytest.raises(ValueError, match="invalid HARE acoustic memory"):
        AcousticMemory.load(truncated)
    with pytest.raises(FileNotFoundError):
        AcousticMemory.load(tmp_path / "missing.npz")


def test_duplicate_npz_members_are_rejected(spec, tmp_path):
    path = AcousticMemory(spec).save(tmp_path / "duplicate.npz")
    with zipfile.ZipFile(path, "a") as archive:
        original = archive.read("meta.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("meta.npy", original)
    with pytest.raises(ValueError, match="expected archive members"):
        AcousticMemory.load(path)


@pytest.mark.parametrize("kind", ["encryption", "compression"])
def test_unsupported_zip_features_are_reported_as_invalid(spec, tmp_path, kind):
    path = AcousticMemory(spec).save(tmp_path / "unsupported.npz")
    contents = bytearray(path.read_bytes())
    local = contents.find(b"PK\x03\x04")
    central = contents.find(b"PK\x01\x02")
    assert local >= 0 and central >= 0
    if kind == "encryption":
        contents[local + 6] |= 1
        contents[central + 8] |= 1
    else:
        contents[local + 8:local + 10] = (99).to_bytes(2, "little")
        contents[central + 10:central + 12] = (99).to_bytes(2, "little")
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="invalid HARE acoustic memory"):
        AcousticMemory.load(path)


@pytest.mark.parametrize("member", ["meta.npy", "ids.npy", "states.npy", "mcep.npy"])
def test_non_npy_zip_members_are_rejected(spec, tmp_path, member):
    valid = AcousticMemory(spec).save(tmp_path / "valid.npz")
    corrupt = tmp_path / "corrupt.npz"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(corrupt, "w") as dest:
        for name in source.namelist():
            dest.writestr(name, b"not an NPY" if name == member else source.read(name))
    with pytest.raises(ValueError, match="invalid HARE acoustic memory"):
        AcousticMemory.load(corrupt)


def _write_on_unpickle(path):
    Path(path).write_text("unsafe", encoding="utf-8")
    return "executed"


class _PicklePayload:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return _write_on_unpickle, (self.path,)


def test_loading_object_arrays_never_executes_pickle(valid_arrays, tmp_path):
    marker = tmp_path / "should-not-exist"
    valid_arrays["ids"] = np.array([_PicklePayload(marker), "second"], dtype=object)
    path = tmp_path / "unsafe.npz"
    np.savez_compressed(path, **valid_arrays)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        AcousticMemory.load(path)
    assert not marker.exists()


def test_mismatched_spec_is_rejected_before_reading_frame_arrays(valid_arrays, spec, tmp_path):
    # The object member would fail if read. Expected-spec validation comes first.
    valid_arrays["ids"] = np.array(["first", "second"], dtype=object)
    path = tmp_path / "wrong-spec.npz"
    np.savez_compressed(path, **valid_arrays)
    spec.fs = 44100
    with pytest.raises(ValueError, match="expected_spec"):
        AcousticMemory.load(path, expected_spec=spec)


def test_hare_is_not_imported_by_normal_hms_usage():
    code = ("import sys; import hms; import hms.core.synthesizer; "
            "assert 'hms.hare' not in sys.modules; "
            "assert 'hms.hare.memory' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hare_import_has_no_dependencies_beyond_numpy():
    code = ("import sys; import hms.hare; "
            "assert not ({'yaml', 'torch', 'scipy', 'librosa', 'faiss'} "
            "& sys.modules.keys())")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_benchmark_script_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools" / "bench_hare_memory.py"
    output = tmp_path / "bench.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--frames", "3", "--repeat", "1",
         "--warmup", "0", "--json", str(output)],
        check=True, capture_output=True, text=True)
    results = json.loads(completed.stdout)
    assert results == json.loads(output.read_text(encoding="utf-8"))
    assert results["frames"] == 3
    assert results["memory_npz_bytes"] > 0
    assert results["load_memory_median_ms"] >= 0
