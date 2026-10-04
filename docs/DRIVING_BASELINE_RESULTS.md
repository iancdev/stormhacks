# First real steering-and-pedal baseline

The Windows GPU run completed 10 epochs in 246.303 seconds, using source
`b678c33baf7001bf570fda6f8aae2b72ee11a1ea`. The best held-out aggregate checkpoint
was epoch 2. The export is saved locally on the Windows training machine and has
not been activated. These are offline results on one held-out recording of the
same circuit, not evidence of successful closed-loop driving.

## Data and configuration

Source: `recordings_22min_20261003.zip`, 295,426,576 bytes, SHA-256
`fbb8f0583fcc509520488e802ca6f111f8e245a1cdd7b8359beac020ce38af98`.
All 42,615 original archive files were verified byte-identical after import and
training. Strict import accepted 42,208 frames and excluded 400:

| Recording | Accepted | Excluded | Role |
|---|---:|---|---|
| 20261003_152944 | 1,988 | 9 ambiguous ages | Training |
| 20261003_154540 | 34,982 | 262 ambiguous ages, 103 duplicate times/images | Training |
| 20261003_162649 | 5,238 | 26 ambiguous ages | Validation |

Seed 7 and validation fraction .25 choose one of three whole recording groups;
validation is **12.41% of rows**, not 25%. Segments from rewinds were not promoted
to independent groups. The two configured bad sessions are absent. The short
152944 recording had been used in earlier diagnostic work; it is training data,
not an untouched test set. Human demonstration provenance is based on the
user-supplied physical-input recordings and README, not independent video
attestation. Car identity is row-verified only in 162649; the source README
attributes all three to the same Audi R8 and circuit.

Configuration: v2 driving, CUDA FP32, batch 64, Adam learning rate .001, seed 7,
workers 0, 10 epochs, 900-second hard watchdog. The 512 MiB preprocessing cache
disabled itself because the full working set requires 6,686,422,528 bytes.
Images streamed through unchanged 200x66 preprocessing. No mirroring, additional
speed/gear/edge filtering, label shift, or steering clipping was added.

Training minimizes equal MSE on normalized angle (`angle/450`) and signed
longitudinal action (`throttle - brake`, brake priority on the one overlapping
training example). Best-checkpoint selection minimizes aggregate normalized
RMSE over public outputs `[angle/450, throttle, brake]`; this is not solely the
checkpoint with lowest steering error. Best normalized aggregate RMSE: .13490145.

## Held-out results at epoch 2

| Output / baseline | MAE | RMSE |
|---|---:|---:|
| Steering model, degrees | 14.7343 | 23.0301 |
| Always zero steering, degrees | 18.1033 | 27.8801 |
| Training-mean steering (6.6681 degrees) | 18.5150 | 26.7720 |
| Throttle model, fraction pressed | .126847 | .219626 |
| Brake model, fraction pressed | .012969 | .061158 |
| Always zero brake, fraction pressed | .006478 | — |

There are 97 validation frames with brake > .05. On these frames the model's
brake MAE is .154664 and RMSE .184053, versus zero-brake MAE .348823. All 97 have
predicted brake > .05; mean prediction is .43928 versus target .34882. However,
**224 of 5,138 exactly zero-brake frames also receive predicted brake > .05**.
Overall brake MAE is therefore worse than the zero-brake baseline. Frame-level
counts are not independent braking-event trials. The model is not established
as a reliable braking controller.

Steering predictions are not constant +25 degrees:

| Statistic | Prediction | Label |
|---|---:|---:|
| Mean, degrees | 3.0207 | 7.8752 |
| Standard deviation, degrees | 16.0168 | 26.7447 |
| Range, degrees | -73.4166 to 72.3541 | -95.3 to 176.96 |

Prediction/label correlation is .542762. Directional errors demonstrate
underestimated turns, especially to the right:

| Label bin | Frames | MAE degrees | Mean prediction | Mean label |
|---|---:|---:|---:|---:|
| Less than -5 degrees | 1,548 | 13.1992 | -8.162 | -16.031 |
| Between -5 and +5 degrees | 1,382 | 7.3281 | 1.063 | -.569 |
| Greater than +5 degrees | 2,308 | 20.1986 | 11.693 | 28.965 |

Later epochs lowered training loss while worsening held-out aggregate error.
More epochs alone are not supported as the immediate remedy.

## Export and evidence

CPU checkpoint and exported weights, plus predictions on three checked frames,
matched exactly. The initial cross-device tolerance of .001 degree failed at a
maximum CPU/GPU steering difference of .00108925 degree (pedals 5.32e-6).
This difference is retained in the evidence; exact cross-device parity is not
claimed. No server replacement, model activation, wheel or live vJoy action
was performed.

Windows root:
`C:\Users\user\Documents\git\stormhacks-driving-b678c33`

- Run: `runs\driving-v2-initial-20261003`
- Checkpoints: `checkpoints\best.pt` and `checkpoints\last.pt` under the run
- Export: `export-best-v2_NOT_ACTIVATED` under the run
- Metrics: `evaluation.json` and `heldout-predictions-labels.npz` under the run
- Readiness/evaluation/source-preservation report: separate local evidence repo
  `runs\dataset-readiness-22min`, commit `d01e798`

Weights and recordings were not pushed to GitHub. Source implementation tests:
567 tests plus 197 subtests passed after merging the RawInput change; later
profile/dashboard tests passed 155 and final pedal/provenance tests passed 14.
These are overlapping test selections, not additive totals.

## Assessment of the collaborator's critique

The critique's batch-32 steering-only description predates this run. The actual
split keeps the long 154540 recording in training, so the suggested accidental
long-recording holdout did not occur. The new model is demonstrably not constant
+25 degrees; the earlier 16-step diagnostic model is a separate artifact.

Mirroring and label/filter experiments are plausible improvements, not proven
causes of the observed errors. The current code has no image mirroring. Its
label-offset option applies to timestamped stream sessions; the imported rounded
`record.py` format intentionally rejects nonzero offsets. A shifted-label
experiment needs a defined within-segment alignment implementation. A quoted
66 ms screen delay and proposed 100-200 ms shift are not verified alignment for
this run.

Reducing the tanh output scale from 450 to 90 would also reduce representable
steering to +/-90 and lose genuine labels beyond that range. Do not silently
clip them. In this multitask model, increasing steering loss weight can test
the intended relative weighting while preserving the full output range. For
example, multiplying steering MSE by 25 has the squared-error scaling effect
of dividing steering error by 90 instead of 450, without imposing a 90-degree
output bound. This is an experiment to validate, not a selected setting.
The runtime's separately configurable target-angle limit is another concept.

Prioritized follow-ups, not performed in this baseline:

1. Use the saved predictions to inspect missed right turns, false braking and
   individual braking episodes. Keep the current split/checkpoint as the baseline.
2. Compare one steering-weight change and one training-only mirroring experiment
   separately, preserving pedal/speed labels and negating angle when mirroring.
   Judge each on directional steering errors and false-braking behavior, not
   only aggregate loss. Mirrored scene semantics still require validation.
3. Test braking imbalance treatment using training data only; preserve braking
   event metrics and avoid selecting a model just because most frames need no brake.
4. Evaluate documented speed/segment-edge filtering and measured timing alignment
   separately. Do not tune a collection of changes against a final test recording.
5. Obtain another untouched recording/route for final evaluation: all current
   recordings are from one circuit, and repeated tuning uses this validation set.

See [AUTOMATIC_PEDALS.md](AUTOMATIC_PEDALS.md) for the actual train/server/client
commands. `--auto-pedals --assist` selects control; shadow remains optional.
