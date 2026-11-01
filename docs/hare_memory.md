# HARE acoustic memory (milestone 1)

HARE (Historical Acoustic Retrieval Enhancement, also called Hybrid Acoustic
Retrieval Engine) is an **optional, standalone** store of individual acoustic
observations. This milestone does not retrieve nearest neighbours, blend
features, train embeddings, or change HMS training, synthesis, or voicebank
files. Nothing in the normal pipeline imports or uses `hms.hare`.

## What one observation means

An observation is one already-aligned frame in HMS's existing
`FeatureSpec` representation, **before feature normalization**:

* `mcep`: `n_mcep` static mel-cepstral coefficients (including c0), from
  columns `1:1 + n_mcep` of `FeatureSpec.encode(...)`;
* `aperiodicity`: `n_band` static band means, from the remaining columns;
* `phoneme`: the caller's phoneme symbol, typically canonicalized with the
  relevant HMS `PhonemeSet`;
* `state`: a non-negative, zero-based state index **within the HMM used for
  that phoneme occurrence** (which may be a phone, context, or backoff HMM);
* `observation_id`: a caller-supplied, unique, nonempty string;
* `context`: any available string-to-string metadata, including
  `Segment.context` and/or neighboring phone symbols. Keys must be nonempty;
  values may be empty. Nothing is inferred when metadata is absent.

The caller must supply the alignment and identifiers. HMS's
`LeftToRightHMM.segment` produces frame-to-state indices from a **normalized
static+dynamic** segment; it must be run with the appropriate HMM and the
same input as HMS, not on these stored slices alone. Record which HMM/utterance
was used in context metadata when relevant. `hms.core.context.neighbor_contexts`
provides the existing neighboring-phone convention, including the inventory's
silence symbol at utterance boundaries.

`from_static` accepts exactly a finite one-dimensional `FeatureSpec.encode`
static frame. It checks but discards the F0 slot; it neither extracts features
nor stores pitch, voicing, deltas or delta-deltas. HMS also produces
note-relative static pitch during analysis; that does not change the stored
mcep and band slices. Do not pass normalized HMM features as raw observations:
if using normalized static features, first apply the appropriate model's
`HMSModel.denormalize` to the **static block**. Width checks cannot detect
whether otherwise valid values were normalized or came from a different
`FeatureSpec` with the same widths, so the caller is responsible for their
provenance. The entire `FeatureSpec` is recorded in the archive, not just the
widths, and can be checked against the consuming model on load.

## Usage

```python
import numpy as np
from hms.core.context import neighbor_contexts
from hms.core.features import FeatureSpec
from hms.core.labels import Segment
from hms.hare import AcousticMemory, AcousticObservation

spec = FeatureSpec(fs=22050, fft_size=1024, n_mcep=4, n_band=3)
# In practice these are produced by the existing vocoder analysis pipeline.
f0 = np.array([220.0, 220.0])
sp = np.full((2, spec.n_bins), 0.01)
ap = np.full((2, spec.n_bins), 0.4)
static = spec.encode(f0, sp, ap)

segment = Segment("a", 0.0, 0.01, context={"style": "legato"})
pre, phone, post = list(neighbor_contexts(["sil", "a", "n"], "sil"))[1]
context = {**segment.context, "pre_phone": pre, "future_phone": post}
# Illustrative alignment only: use the actual trained HMM's state path.
states = [0, 1]
memory = AcousticMemory(spec)
memory.add_batch(
    AcousticObservation.from_static(f"take1:{i}", phone, states[i], frame,
                                    spec, context)
    for i, frame in enumerate(static)
)
assert len(memory) == 2
assert memory.get("take1:0").phoneme == "a"
assert len(memory.filter(phoneme="a", state=1)) == 1

path = memory.save("scratch/acoustic_memory.npz")  # separate from a voicebank
loaded = AcousticMemory.load(path, expected_spec=spec)
assert [o.observation_id for o in loaded] == ["take1:0", "take1:1"]
```

`AcousticObservation(...)` also accepts the two existing feature slices
separately. Real numeric inputs are converted to float64; complex, Boolean,
string and object arrays, wrong dimensions, and NaN/infinity are rejected.
The memory validates widths against its spec. IDs and phoneme symbols must be
nonempty Unicode strings without NUL (NumPy's fixed-width string padding could
otherwise lose characters); IDs are unique across the memory. State indices
must fit a signed int64; `bool` is not an index. Observations snapshot their
inputs and expose read-only feature arrays and immutable context; the memory
snapshots its spec, and `memory.spec` returns a copy.

`add` inserts one observation; `add_batch` accepts any iterable and is atomic
on validation or iteration failure. `get` raises `KeyError` for a missing ID.
Iteration and filtering preserve insertion order. `filter` does exact,
case-sensitive matching; either argument can be omitted, and `len(memory)` or
`len(memory.filter(...))` gives observation counts. ID lookup is dictionary
based; filtering is a linear scan, not a similarity search.

## Standalone NPZ schema, version 1

`save` writes a compressed NumPy `.npz`, creates parent directories, appends
`.npz` if needed, overwrites an existing file, and returns the actual `Path`.
It does not create or modify any HMS voicebank file. `N` is the observation
count; `M = feature_spec.n_mcep`, `B = feature_spec.n_band`.

| NPZ member | Shape / dtype | Meaning |
|---|---|---|
| `meta` | scalar Unicode (`<U...`) | Canonical JSON object containing `format: "hms.hare.acoustic_memory"`, integer `version: 1`, integer `count: N`, and the complete version-1 `FeatureSpec.to_dict()` as `feature_spec`. |
| `ids` | `(N,)`, Unicode (`<U...`) | Unique observation IDs, insertion order. |
| `phonemes` | `(N,)`, Unicode (`<U...`) | Corresponding exact phoneme symbols. |
| `states` | `(N,)`, little-endian signed int64 (`<i8`) | Zero-based non-negative HMM state indices. |
| `mcep` | `(N, M)`, little-endian float64 (`<f8`) | Raw static mel-cepstral coefficients. |
| `aperiodicity` | `(N, B)`, little-endian float64 (`<f8`) | Raw static aperiodicity bands. |
| `contexts` | `(N,)`, Unicode (`<U...`) | One canonical JSON object of string keys/values per observation; `{}` means unavailable. |

NumPy Unicode arrays are fixed-width UTF-32, padded to the longest string in
each array. JSON is ASCII-escaped with sorted keys. Numeric values, row order,
IDs, context contents and the complete feature spec survive a round trip;
float64 values are bit-preserved, including signed zero. Saving the same ordered
observations twice, or saving after loading, produces identical archive bytes
under the same Python/NumPy/compression environment. Compression output is not
promised byte-identical across library versions or platforms.

Loading uses `np.load(..., allow_pickle=False)` and rejects missing, extra or
duplicate members, malformed JSON or duplicate JSON keys, unknown format/version,
invalid FeatureSpec values, inconsistent counts/shapes/dtypes, non-finite
features, invalid Unicode, bad state indices and duplicate IDs. Malformed or
incompatible files raise `ValueError` with the path and cause; file-open errors
remain `OSError` (e.g. `FileNotFoundError`). Supplying `expected_spec` requires
an exact match of **all** recorded FeatureSpec fields. Omitting it reads the
stored spec without comparing it to another model.

This simple in-memory format is not a streaming store: save/load materialize
arrays and fixed-width strings may be expensive if a single ID or context is
much longer than the rest. Disabling pickle prevents code execution through
object arrays, but does not limit archive size or establish authenticity; bound
file sizes and use trusted sources when resource exhaustion is a concern.

## Reproducible overhead benchmark

`tools/bench_hare_memory.py` uses seeded synthetic *already-encoded* frames and
measures medians after one warmup, with allocation peaks measured separately by
`tracemalloc`. It does not time analysis, HMM alignment, retrieval, synthesis,
or real-corpus quality. The dict baseline indexes the same prebuilt objects
without width/uniqueness validation. The numeric-only NPZ baseline contains
only the two acoustic slices together, without IDs, phones, states, context,
or feature spec; it is **not** functionally equivalent to the memory.

Example run on a 2-vCPU Intel Xeon 2.60 GHz Linux sandbox, Python 3.11.2,
NumPy 2.4.6, one BLAS thread, seed 0, 10,000 frames (`n_mcep=30`, `n_band=5`),
five measured repetitions:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python tools/bench_hare_memory.py --frames 10000 --repeat 5
```

| Operation | Median |
|---|---:|
| Dictionary index of prebuilt observations (baseline) | 0.55 ms |
| HARE insertion of prebuilt observations, batch | 2.71 ms |
| HARE insertion of prebuilt observations, one at a time | 4.98 ms |
| Construct/validate observations from static frames and batch-insert | 210.21 ms (21.0 µs/frame) |
| Save numeric-only compressed NPZ (baseline) | 106.79 ms |
| Save HARE compressed NPZ, including all metadata | 165.09 ms |
| Load numeric-only NPZ (baseline) | 14.76 ms |
| Load/validate HARE NPZ and rebuild observations | 246.37 ms |

The numeric-only file was 2,689,185 bytes; the HARE file was 2,716,389 bytes
(about 1.0% larger for this repetitive synthetic metadata). Raw acoustic
slices occupy 2,800,000 bytes. Separately measured Python+NumPy allocation
peaks were 10,215,890 bytes to construct and batch-insert and 16,498,955 bytes
to save; these are not process RSS. Results are machine- and data-dependent;
the benchmark can be rerun with different frame counts and `--json`.
