# Preprocessing cache verification

The production change uses a shared CPU LRU cache, default 256 MiB of retained
tensor payload. It does not enable GPU residency, pinned memory, larger batches,
AMP, TF32, compilation or additional loader workers.

## Full-trainer CPU measurement

Executed on the Mac with Python 3.10, PyTorch 2.13.0, one CPU thread, seed 7,
batch 32, three epochs. The fixture contains three independently generated
synthetic session groups, 128 frames each: 256 training and 128 validation samples
per epoch. These are software fixtures, not claimed driving demonstrations.

A separate one-epoch library warmup preceded two paired repetitions, in the order
uncached/cached then cached/uncached. Each timed `train` call includes fresh data
validation, splitting, cache population, training, held-out validation and both
checkpoint writes. Fixture generation and Python import are outside the timer.

| Repetition | Uncached seconds | 256 MiB cache seconds |
|---|---:|---:|
| 1 | 4.6081 | 4.2362 |
| 2 | 4.5670 | 4.1878 |
| Mean | 4.5875 | 4.2120 |

The cache reduced total time by approximately 8.2% (1.09x throughput) in this small,
CPU-compute-heavy fixture. Complete metric histories and final weights were
bit-identical across all four runs. This is a full-trainer measurement; it is not
an RTX 5080 result and does not establish production-data speedup.

Reproduce in an environment with the project importable:

```sh
python scripts/benchmark_training_cache.py --output /tmp/cache-comparison-new.json
```

The script defaults to CPU and requires a new output path. Temporary fixtures and
checkpoints are removed at completion. Device `cuda` is available only for an
explicitly authorized GPU comparison; no GPU execution was performed for this
production patch.

## Correctness coverage

Tests compare every cached/uncached tensor, shuffled batches, predictions and
losses. Complete CPU training histories, model and optimizer states, RNG state,
and resumed runs are compared exactly. Legacy checkpoints without the new budget
field resume successfully. Existing best-checkpoint interruption/repair and
recording exclusion/split tests continue to run.

Focused tests also cover validation reuse, per-session identity changes,
preprocessing/alignment changes, accepted-row/label changes, LRU eviction,
undersized-budget fallback, allocation failure cleanup and disabling cache when
loader workers are requested. No real recordings, source image bytes or split
policies are changed.

## Relation to earlier GPU diagnostic

The earlier Windows experiment used a different isolated batch utility with 512
real samples and only six measured steps per configuration. It found large gains
from avoiding repeated JPEG preprocessing, including 3.6x for CPU caching at
unchanged batch 32. Those pipeline-only rates exclude full-trainer overhead and
must not be presented as the speedup of this production implementation. This
implementation additionally uses bounded lazy storage and returns protective
clones; full production GPU timing remains a separate measurement.

Final software verification: **535 tests and 197 subtests passed** in 25.06 s on
the Mac. An earlier broad-suite attempt exhausted local disk space; after removing
only this task's identified temporary fixtures, the complete rerun passed. No
product fix was required for that environment failure.
