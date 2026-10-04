# Training pipeline

This pipeline learns physical wheel angle from RGB road crops and speed. Synthetic
fixtures verify software execution only. There is no trained driving policy or
real-world driving-quality result yet.

## Installation

The primary training host is the connected Windows desktop `DESKTOP-0HR4O88`;
it also serves live inference over LAN to the separate game/wheel PC. See
[TWO_PC_SETUP.md](TWO_PC_SETUP.md). Colab remains a fallback.

Use Python 3.10+ on the desktop, Colab, or another training machine. Install a CUDA-compatible
PyTorch build appropriate for that host, then run from the repository root:

```bash
python -m pip install -e '.[training,test]'
```

The base package needs NumPy and Pillow. The `training` extra adds PyTorch and
`test` adds pytest. The `hardware` extra is Windows-only; `requirements.txt` is
the original Windows diagnostic environment and is not a Colab requirements file.
No Windows library is imported by training or inference.

## Completed data

Follow [CONTRACTS.md](CONTRACTS.md). Put one or more session directories immediately
under a dataset directory. This trainer additionally **requires `completed: true`**
in each `metadata.json`; set it only after all recording/copying has finished.
Use distinct session IDs and archive/copy completed sessions onto the training
machine's local disk. Do not train directly against a recorder's output directory.

Input images are already road crops. The default preprocessing applies **no second
crop**, resizes to 200×66 with Pillow bilinear interpolation, converts RGB uint8
pixels to float32 CHW using `pixel / 127.5 - 1`, and divides speed by 50 m/s.
The model's tanh output is multiplied by 450 degrees; right is positive.
All these settings are stored in checkpoints and exported metadata.

Structural errors (missing/corrupt files, escaping image paths, duplicate IDs,
non-increasing timestamps, invalid units/ranges, non-finite numbers, unfinished
sessions) fail validation. Timing-ineligible frames are counted by reason. A
session with zero eligible frames fails CLI validation and training.

The defaults are zero label offset, maximum wheel sample gap 50 ms, and maximum
telemetry age/gap 100 ms. Speed comes from the latest sample at or before capture.
Wheel angle is exact or linearly interpolated at capture plus the configured
label offset. Every wheel sample spanning source capture and target must be in
the same expert mode (`manual` or `takeover`) with bounded gaps. Mode boundaries,
AI `assist` frames, missing coverage, stale telemetry, and race-off intervals are
excluded. Race state and telemetry freshness are checked over the full wheel
interpolation support bracket as well as the source-to-target interval; speed
input is still sampled causally at capture. Positive and negative offsets are supported; choose them based on
measured recording latency, not an assumed reaction time. A nonzero offset may
exclude frames at a session's edges.

V1 cannot detect an unmarked in-game rewind or pause while host telemetry still
reports race-on. End the session at rewinds or such pauses. Keep recording
metadata about car, route, camera, weather, and crop consistent; validation cannot
infer those conditions from pixels.

## CLI workflow

```bash
# Small CPU software smoke run (use real completed sessions for actual training).
forza-train synthetic /tmp/forza-fixture --sessions 3 --frames 24
forza-train validate /tmp/forza-fixture
forza-train train /tmp/forza-fixture /tmp/forza-run --epochs 1 --device cpu
forza-train resume /tmp/forza-run/last.pt /tmp/forza-fixture --epochs 2 --device cpu
forza-train evaluate /tmp/forza-run/best.pt /tmp/forza-fixture
forza-train export /tmp/forza-run/best.pt /tmp/forza-export
python -m pytest tests/training -q
```

`python -m forza_ai.training.cli` is equivalent to `forza-train`. For a GPU run,
use `--device cuda`, a suitable `--batch-size`, and a checkpoint output directory
on persistent storage. `auto` chooses CUDA when available and otherwise CPU.
Training options include `--learning-rate`, `--validation-fraction`, `--seed`,
`--workers`, `--label-offset-ms`, `--max-wheel-gap-ms`, and
`--max-telemetry-age-ms`. Use the same alignment settings when inspecting data.

At least two sessions are required. Splits use entire session IDs and a seeded
shuffle, never random neighboring frames. The training mean baseline uses only
training labels. Reports include model, zero-angle and training-mean MAE/RMSE in
degrees, aggregated by frame and separately by session. These offline metrics
cannot establish closed-loop driving quality. For related laps/sessions with
near-identical conditions, also evaluate a separately collected route/day.

`last.pt` contains model, Adam optimizer, completed epoch, random state, history,
configuration, split IDs, and content fingerprints including images. `best.pt`
is the checkpoint with lowest held-out RMSE. Writes use a temporary file followed
by replace. `last.pt` embeds a frozen best-checkpoint snapshot; resume repairs
`best.pt` from that snapshot before training, including after an interrupted pair
of checkpoint writes. This increases checkpoint storage to preserve recoverability. Persist the run directory outside Colab's ephemeral filesystem.
A stopped partial epoch is repeated from the last completed checkpoint.
`resume --epochs N` means a **total** of N epochs; saved configuration is reused.
Resume writes into the checkpoint directory. To relocate a run, copy the whole
run directory (including best.pt) and resume there.
Resume validates exact dataset contents, permitting a different filesystem root
but rejecting added, removed, or changed sessions. Deterministic CPU continuation
is tested; bitwise equality across devices/PyTorch versions is not promised.

`evaluate` selects the original validation sessions and verifies their contents.
Use `--unseen` to evaluate entirely new session IDs; this rejects all IDs used in
the original run. Neither path evaluates training sessions as held-out data.

## Portable prediction

```python
from forza_ai.policies.predictor import SteeringPredictor

predictor = SteeringPredictor('/path/to/forza-export')
angle_deg = predictor.predict(road_crop_rgb, speed_mps)
```

`road_crop_rgb` must be a uint8 NumPy array shaped H×W×3 in RGB order. The predictor
loads CPU weights, validates inputs, and returns a finite physical angle in
degrees. The exported directory contains `model.pt` and `metadata.json` describing
architecture version, input/output normalization, alignment, and evaluation.
Install this package plus the training extra on the inference machine. The
adapter owns freshness checks, takeover, torque limits and engagement; this
predictor is only the policy. Load artifacts from a trusted source.

## Architecture reference

The five convolution stages are inspired by Bojarski et al.,
[End to End Learning for Self-Driving Cars (2016), §4 and Figure 4](https://arxiv.org/html/1604.07316).
Their design uses three strided 5×5 convolutions followed by two 3×3 convolutions.
Our implementation uses 24/36/48/64/64 channels, ELU activations, a flattened
1152-element embedding, late fusion of normalized speed, and dense layers
100/50/10/1. It optimizes mean squared normalized wheel-angle error with Adam.

This is a baseline adaptation, not a replication: we use RGB instead of YUV,
add speed, and predict physical TMX angle instead of inverse turning radius.
There is no geometric augmentation or arbitrary shifted-image steering label.
Camera geometry and a justified label transform are needed before adding recovery
augmentation. Curve balancing and real-data tuning remain future work.

## Colab setup cells

`notebooks/train_colab.ipynb` defaults to an executable two-epoch CPU synthetic
smoke run. Download a ZIP of the training branch from signed-in private GitHub,
then select it in the repository upload cell. Alternatively select `existing`
and point at a securely cloned checkout. No access token belongs in the notebook.

For real data set `SYNTHETIC = False`, select a GPU runtime, and put completed
session ZIPs under the configured Drive `ARCHIVE_DIR`. Executable setup cells
extract into runtime-local `DATA` and validate every session. ZIPs may contain a
single session at the root or session folders directly below the root. Traversal,
symlinks/special files, duplicate paths, invalid layouts, duplicate session IDs,
and incomplete sessions are rejected. An existing destination is never replaced;
set `REUSE_DATA = True` only after a successful import, or choose a fresh path.
The stdlib helper is maintained in `scripts/colab_archives.py` and embedded in the
notebook for repository bootstrap; a test enforces equality.

Real checkpoints persist under `RUN` on Drive; synthetic defaults are ephemeral.
Set `RESUME = True` and a higher total `EPOCHS` to continue. Export folders include
the target epoch count and a unique suffix, including repeated evaluations of the
same epoch. Notebook cells are locally smoke-tested, but Google authentication,
Drive mounting, and GPU execution still require validation in Colab.

## Import recordings from `record.py`

The recorder added in `8e379da` writes `meta.json`, `labels.csv`, and JPEGs under
`frames/`. Import each **normally stopped, fully copied** recording explicitly:

```bash
forza-train import-recording /path/to/recording /path/to/dataset/drive-001 --expert-mode manual
forza-train import-recording /path/to/another-recording /path/to/dataset/drive-002 --expert-mode manual
forza-train validate /path/to/dataset
forza-train train /path/to/dataset /path/to/run --device cuda --epochs 10
```

On Windows the same commands accept quoted Windows paths. Only the normal
training dependencies are needed; the importer does not import the recorder,
OpenCV, wheel drivers, or other Windows hardware packages. `--expert-mode manual`
declares that the entire source recording contains human steering demonstrations.
Do not use it for AI-generated steering; this recorder does not log control mode.
Source files are never edited and an existing destination is never overwritten.

The importer retains original `labels.csv`, `meta.json`, and JPEG bytes. It adds
`metadata.json` identifying the **distinct `record_py_aligned_v1` format**. It does
not manufacture v1 wheel/telemetry streams or claim exact capture timestamps.
The existing training CLI recognizes this format alongside v1 sessions.

Completion is inferred conservatively from `meta.json` (written after the
recorder's normal writer closure), matching row/image/frame counts, segment
counts, ordered frame indexes, decodable JPEG dimensions, and valid values.
Missing/corrupt images, extra files in `frames/`, inconsistent counts, absent
metadata, invalid calibration, or `--no-telemetry` recordings fail import.
An arbitrary exception or a partial copy is not silently treated as completed.
There is no partial-recording recovery mode. Preserve the source and investigate
or recopy it rather than inventing completion metadata.

**Timing limitation:** `t` is elapsed `perf_counter` sampled after
`get_latest_frame()` returns, rounded to 0.0001 seconds. It is not capture time.
`video_mode=True` can return cached images, so image age has **no certified upper
bound**. Wheel/telemetry ages are rounded to 0.1 ms, wheel angle to 0.01 degrees,
and speed to 0.001 m/s. Validation summaries, checkpoints, and exports preserve
this provenance, capture configuration, saved dimensions, and unknown image-age
bound. Internal integer nanosecond representation of `t` remains quantized to
100,000 ns; it adds no timestamp precision.

Labels and speed are used only as already recorded on each row. **Only zero label
offset is supported**; no interpolation or label shifting occurs within or across
segments. Repeated wheel/telemetry observations are not reconstructed into fake
unique source samples from independently rounded times and ages. The maximum
wheel-gap setting is interpreted as maximum wheel **age** for this aligned format;
the telemetry-age setting keeps its age meaning. Eligibility uses the upper end
of each age's ±0.05 ms rounding interval. An age of 50.0 ms therefore exceeds a
50 ms limit. Zero/negative rounded ages cannot certify that the sampled value
existed at retrieval and are excluded, as are race-off rows. Consecutive identical
JPEGs or repeated rounded retrieval times are excluded and counted; this cannot
detect every stale/cached image or establish a true capture-age bound.

All segments of one parent recording remain a **single session and split group**.
At least two independent parent recordings are required for train/validation;
multiple segments in one recording are not independent held-out data. The
splitter also honors optional `split_group` in v1 metadata, keeping related
sessions together. Evaluation with `--unseen` rejects groups from the original
run even if their session IDs differ. Parent identity derives from the recorder's
`session` field; renaming directories does not create an independent recording.
Do not relabel/copy subsets as independent recordings to obtain a split.

The saved road images are already cropped, masked and resized using the recorder's
OpenCV `INTER_AREA` transform. Import copies them unchanged; model preprocessing
still applies the exported RGB/Pillow bilinear 200×66 transform. For live parity,
configure the runtime with that recording's exact capture configuration and saved
size before model preprocessing. No real recording or driving result is implied
by the synthetic import tests.

A normally closed recorder can report empty segments: it increments the segment
before enqueueing a frame, and `queue.Full` can drop every frame in that segment,
including a final resumed segment. For the original unbuffered schema, absent segment IDs need recorded queue
drops, subject to the narrow Ctrl+C exception below. Buffered producers use
their explicit discard counters as described below. Segment IDs must still be ordered and within the declared count. This
check admits that recorder behavior without inventing samples or accepting an
unexplained mismatch in completion metadata.

### Original Ctrl+C segment edge and diagnostic-only inspection

The original `8e379da` recorder increments `segment` before processing/enqueueing
the frame. A `KeyboardInterrupt` during that processing is caught as a normal
stop, but neither a frame nor a queue-full drop has been counted. Replaying the
actual historical loop with inert dependencies reproduced one saved segment,
`segments=2`, `dropped=0`, and normal cleanup. Import therefore permits **at most
one additional unaccounted trailing segment**, only for the original exact
metadata-key shape and original CSV columns. Other empty segments still require
recorded drops. Later/extended producer metadata does not receive this exception;
its segment semantics must be established from its source. Accepted legacy
imports report the Ctrl+C caveat in `empty_segment_evidence`.

### Buffered recorder variants (merged source 26f0970)

The producer lineage is now available: `2c4c81e` discards pending frames on stop,
`0bb45a1` adds yaw/game-clock fields, `82c2e46` adds gear/HUD patches, `4fdefa7`
adds game-takeover filtering, and `0117f25` adds car identity fields. Strict import
supports their exact ordered column families, including the sample's 17 columns
and the merged recorder's 20 columns. This establishes schema semantics, not the
exact binary/source revision executed for an archive without a recorded hash.

`frames` counts saved rows. Frame IDs count captured entries before pending-buffer
discards and can have gaps; filenames must exactly match the retained row IDs.
Those gaps need enough queue-drop or rewind/takeover discard evidence. Stop
discards cannot explain an earlier frame-ID gap. Opened segments can contain no
saved rows when their pending frames are discarded. Missing segments must be
accounted for by the same loss counters without spending a counter twice for
both earlier missing IDs and a later empty segment. `discarded_at_stop` is a
**frame count**; `drop_seconds` is a **buffer duration**, not that count.
Optional `accepted_frame_count` must reconcile with saved rows plus all losses;
older archives do not need this newer field. Explicit `completed: false`, missing
images, malformed data, contradictory counters and unknown schemas still fail.

Extra telemetry is preserved in the original CSV, checked for finite/type-correct
values, and excluded from model inputs. No distance rescaling or car-normalization
is inferred. Optional `hud/*.png` patches are auxiliary sync diagnostics: an
archive may omit them entirely, or contain a subset; present patches are validated,
copied, and fingerprinted. They are never used as road images. Optional
`capture_timing.csv` is preserved and must cover the saved row IDs. If source
metadata advertises that sidecar, it must exist. It is not silently promoted to
certified capture time: training continues to use zero-offset aligned labels and
conservative rounded-age bounds. New completed/jpeg_quality/measured_capture/
capture_provenance metadata remains intact in `source_metadata`.

The unchanged `smoke_20261003_152944.zip` (13,689,689 bytes, SHA-256
`7829b5458609202f8ddd970ed0d789800a1347e775cbb0360234a10f214fbaeb`)
now passes strict import with **1,988 accepted rows and 9 rounded-age exclusions**.
It declares two opened segments, one saved segment, no queue drops or rewinds,
and five discarded-at-stop frames; the source-backed buffering rules account for
that case. Original source bytes are preserved. This remains one independent
parent recording; normal training still requires at least two groups. No GPU
compute, held-out driving result, or model readiness follows from import success.

```bash
forza-train import-recording SOURCE DEST --expert-mode manual --exclude-sessions config/exclude_sessions.txt
forza-train validate DATASET
```

Exclusions are mandatory for normal import and production session loading
(including train/resume/evaluate and already-imported datasets). A checkout reads
its `config/exclude_sessions.txt`; an installed wheel uses the bundled policy
copy. Missing policy files fail closed. Current defaults exclude the parent
prefixes `20261003_150225` and `20261003_152123`; the supplied `20261003_152944`
sample is allowed. Checks use source IDs and split groups, including nested
original metadata, so renaming a folder or an imported artifact does not bypass
policy. Direct production splitting also checks the identities.

`--exclude-sessions PATH` adds prefixes to mandatory defaults; it never replaces
them. Checked prefixes are recorded in new import manifests, but production loads
recheck the current policy rather than trusting that historical list. Updating
repository exclusions requires updating the bundled `src/forza_ai/data/exclude_sessions.txt`
copy for wheel distributions; a regression test enforces equality. Diagnostic-only
read-only inspection remains available for excluded sources, without admitting
them to production splits. No original recordings are changed or deleted.

`inspect_recording_for_diagnostics(source, expert_mode="manual")` remains an
explicit read-only fallback for unresolved producer variants. Its results always
carry `diagnostic_only: true`, cannot enter production splitting, and must not be
used to manufacture held-out metrics or deployment artifacts. The now-supported
sample no longer needs that fallback for import; its bounded one-recording GPU
smoke remains separate from normal multi-group training.

New recordings may declare `producer_schema=record_py_buffered_20_v1` and
`producer_sha256`. If present, the schema must match the 20-column takeover-buffer
format and the hash must be 64 lowercase hexadecimal characters. Both fields
remain optional for old recordings. Imports preserve them as **declared** producer
identity; syntax validation alone does not prove the executing source's identity.

Archive paths are checked using the raw ZIP member name before host-specific
`ZipInfo` normalization. Colab extraction rejects literal backslashes by default.
For a known legacy Windows ZIP, the standalone helper can explicitly use
`extract_zip(..., allow_legacy_backslashes=True)`; normalization still rejects
traversal, absolute/drive paths, Windows reserved names, and mixed-separator
collisions. Import provenance and auxiliary fingerprint paths use `/` on every
host; newly generated import manifests use LF line endings. Original source
files are copied unchanged.

## Bounded preprocessing cache

Training now shares a run-local CPU cache between training and validation. The
cache stores the exact existing float32 image, speed and label tensors; batch
size (32), model, precision, loss, epoch-derived shuffle and session splits are
unchanged. `--cache-mib 256` is the default tensor-payload budget;
`--cache-mib 0` restores uncached loading. Least-recently-used entries are evicted
when the shared budget fills. If one sample exceeds the budget it is decoded
normally without being retained. Smaller-than-dataset caches may have limited
benefit due to eviction; choose a budget that fits your RAM and measurements.

The budget covers retained tensor bytes, not Python metadata, decoder working
memory, returned sample clones, batches or the PyTorch allocator. Cached tensors
are copied on return so callers cannot mutate future examples. An allocation
failure in cache storage/copying releases the cache and falls back to normal
loading. This cannot protect against OS process termination or a dataset/batch
that cannot fit in memory even without caching.

Caching is disabled with a warning when `--workers` is nonzero, preserving the
requested loader behavior without multiplying cache memory across Windows worker
processes. No GPU/pinned allocation, persistent workers or disk cache is enabled.

Cache identity includes validated content fingerprints, accepted sample paths,
timestamps, labels and control modes, alignment settings and exact preprocessing.
Validation and exclusion checks still run before caching. A new run or resumed
run rebuilds its cache; source recordings must remain immutable during a run.
The cache budget is saved in the existing training configuration; older format-1
checkpoints without the field remain loadable. Resume restores the saved budget
and leaves all model/optimizer/RNG and best-checkpoint recovery semantics intact.
`evaluate --cache-mib` also accepts an explicit budget; a one-pass standalone
validation usually has no cache hits, so it defaults to 0 to avoid retention.
Repeated in-training validation can reuse cached samples within the shared limit.

Use `scripts/benchmark_training_cache.py` for a bounded, full-trainer comparison
on three independent synthetic fixture groups. Its timings include validation,
checkpoint writes, initial data validation and cache filling. Synthetic timing
and numerical parity verify the software path; they do not establish real driving
accuracy or substitute for held-out human recordings.
