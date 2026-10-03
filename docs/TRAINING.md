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
