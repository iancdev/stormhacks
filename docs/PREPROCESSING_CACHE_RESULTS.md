# Preprocessing cache verification

The current production default is 512 MiB of retained CPU tensor payload, with
fit-only admission across all accepted training and validation samples. The
original measurements below used the earlier 256 MiB LRU policy. It does not enable GPU residency, pinned memory, larger batches,
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
python scripts/benchmark_training_cache.py --cache-mib 256 --output /tmp/cache-comparison-new.json
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

Compatibility follow-up: the supported PyTorch 2.2 release exposes the OOM class
as `torch.cuda.OutOfMemoryError` rather than the newer top-level alias. The cache
now uses version-safe lookup. Six regressions removing the top-level alias first
reproduced the failure, then passed for both get/put with Python MemoryError,
PyTorch OOM and CPU allocator RuntimeError. All 45 cache/pipeline tests passed;
unrelated runtime exceptions continue to propagate. This simulates the missing
alias on the installed PyTorch version, not a full PyTorch 2.2 environment run.

Capacity caution: each image/speed/label entry occupies 158,408 tensor bytes, so
256 MiB retains at most 1,694 samples. A 1,988-sample working set needs about
300.33 MiB. If both training and validation passes exceed capacity, they can evict
one another entirely; cache-copy overhead then brings no cross-epoch reuse.
The small-fixture speedup above is not a larger-dataset guarantee. Subsequent full-trainer GPU measurements confirmed this regression, as recorded
below; they motivated fit-only admission.


## Fit-only admission and 512 MiB default

The original Windows full-trainer comparison at 6b02ac1 used unchanged batch32,
FP32, seed7 and independent synthetic groups, with two reversed-order repeats.
When the 58.01 MiB working set fit, mean full-call time fell from 1.27853 s to
0.82523 s. With a 348.07 MiB working set and the old 256 MiB budget, caching
instead increased mean time from 5.25569 s to 5.58995 s. Epoch-two training had
22.72% hits and validation zero hits, with 1,955 evictions. These bounded synthetic
results do not establish a universal real-data speedup.

Admission now estimates the full accepted training-plus-validation tensor working
set, counting exactly the same unique namespace/index keys used by the dataset.
No cache is constructed if that set exceeds the requested budget or its size
cannot be determined safely. Stderr reports the reason, required bytes and budget.
This removes partial-LRU churn at the trainer level; explicit smaller budgets
also fall back rather than forcing a partial cache. Zero remains disabled.

At the user's request, new training runs default to **512 MiB of system RAM**.
The 1,988-sample working set is 314,915,104 bytes (300.33 MiB), so it fits. A
4,000-sample set exceeds the default and is tested to fall back. Existing resumed
runs retain their saved budget and get the same fit-only check; batch size,
precision, shuffle, splits and objective remain unchanged.

CPU full-trainer regression checks used three independent synthetic groups of
128 samples, three epochs, batch32, FP32, seed7, one thread, two reversed-order
repetitions. With an intentionally undersized 32 MiB budget, all calls reported
`working_set_exceeds_budget` (60,828,672 bytes required), constructed no cache and
preserved exact histories and final weights. Timings are noisy short CPU runs;
this fallback is the unchanged uncached path plus one admission estimate, not a
claimed speed optimization.

- Over-budget 32 MiB: baseline mean 5.4952 s; requested-cache mean 5.0674 s. Exact histories/weights match in every call.

- Fitting explicit 256 MiB: baseline mean 4.9949 s; requested-cache mean 4.4159 s. Exact histories/weights match in every call.

The fitting case retains the benefit. Final 512 MiB default admission/CLI behavior
is regression-tested; its full-trainer GPU timing has not yet been remeasured.
The benchmark utility accepts `--cache-mib` and up to 768 frames per independent
synthetic group so both fit and over-budget cases can be measured explicitly.

Final admission/default verification: **549 tests and 197 subtests passed** in
26.60 s. The focused training suite passed 136 tests, including the 512 MiB
size boundary and CLI override checks.
