# Delivery plan

Updated 2026-10-03. This file is the durable plan for both project chats.

## Objective

Train a vision policy offline from human driving recordings, then continuously steer a real Thrustmaster TMX in Forza Horizon 4 while the human controls the pedals. The game receives the measured physical angle through vJoy. Takeover interrupts assistance; it is not the normal driving mode once assistance is engaged.

## Priorities and ownership

1. **Training pipeline first:** chat `01a103b0-2b07-7c73-8934-0324136bc8f6` owns dataset validation/loading, training, evaluation, export, a portable inference wrapper, and Colab. It works in its own worktree and branch. It also owns project packaging and training dependencies.
2. **Adapter in parallel:** the originating chat owns hardware I/O, continuous steering control, telemetry, a placeholder policy, and runtime contracts. Windows-only packages must not be required to run training.
3. **Data collection:** the user will provide real driving data later. The verified hardware reference is `utils/test.py`. The user reports a successful short recording, but its recorder and session files are not in this repository yet. Preserve that result and integrate its format when available; do not assume the diagnostic script is a recorder.

The shared interface is documented in [CONTRACTS.md](CONTRACTS.md). Coordinate changes before diverging. Commit each implementation iteration. Do not commit recordings, credentials, or trained weights.

## Training deliverables

- Versioned session format, structural/timing validator, and a reproducible synthetic fixture.
- Session-level train/validation splits, timestamp alignment, invalid-state filtering, configurable label offset, and stale-data rejection.
- Small CNN consuming a road image and speed, predicting physical steering angle in degrees, right positive.
- CLI training, resume, evaluation against simple baselines, export, and checkpoint reload.
- Exported preprocessing, normalization, architecture/version, and steering scale with the weights.
- Thin Colab notebook calling the same Python code used on another GPU machine. Use runtime-local data for training; persist resumable checkpoints outside the ephemeral runtime.
- End-to-end synthetic smoke test. This verifies software execution only, not driving quality.

First acceptance milestone: synthetic data validates, trains, and reloads a checkpoint. Second milestone: a real session validates and starts training on a GPU device.

## Adapter deliverables

- Hardware abstraction: read TMX, forward measured steering and human pedals to vJoy, write bounded torque to the TMX.
- Independent local wheel-control loop with a PD controller, calibrated sign conversions, target limits, and finite-lived force effects.
- Explicit engagement/takeover, command expiry, wheel-loss handling, and cleanup. Re-engagement after timeout/takeover is explicit.
- FH4 telemetry decoding with host receive timestamps and age checks.
- Replaceable placeholder policy with marked test targets. It does not drive roads.
- Headless simulated hardware for meaningful software checks on this Mac, plus a Windows acceptance procedure.
- Road-force replay/blending is a separate acceptance stage after basic physical angle tracking. FFB packets require effect lifecycle interpretation, not simply retaining the last magnitude.

## Data and deployment

The gaming PC records frames, wheel inputs, telemetry, and control mode into completed sessions. GitHub stores code and tiny synthetic fixtures; session archives and checkpoints live on Drive or another user-chosen storage destination. GPU training consumes completed archives, validates them, and writes resumable checkpoints. Live inference initially runs on the gaming PC CPU, separate from the faster physical control loop.

## Integration and verification

- Preserve the tested `utils/test.py` as a diagnostic reference.
- Test pure conversions, command expiry, takeover/re-engagement, controller saturation, decoder validation, and exception cleanup without hardware.
- Verify on Windows: actual passthrough, force direction, bounded stationary angle tracking, takeover and stale-command release. Passing software tests cannot establish hardware stability.
- Run shadow inference before model-controlled driving; evaluate repeated route completion and interventions before lap time.
- Human takeover recordings are expert labels; AI-generated motion is not automatically expert data.

## Current status

- Private repository created and pushed: `iancdev/stormhacks`.
- Hardware diagnostic and pinned Windows dependencies received in commit `d18ec40`.
- Training work delivered by the designated chat and integrated through `02ba588`: session validation/alignment, image-plus-speed CNN, resumable CLI, metrics/baselines, CPU export/predictor, executable Colab ZIP setup, and synthetic notebook smoke. Real data and actual GPU/Colab execution remain outstanding.
- Adapter software now includes a PD controller, explicit engagement/takeover, command expiry, separate placeholder policy worker, simulated wheel, and a timestamped FH4 receiver. The first simulated three-second run reached a five-degree target while forwarding measured angles. Physical Windows acceptance is still outstanding; see [ADAPTER.md](ADAPTER.md).
- The supplied baseline reference is NVIDIA's [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). The training chat is incorporating the paper with explicit documentation of our speed input and physical-angle output differences.

## Next integration gates

1. Finish checkpoint interruption recovery review and merge the final training fix.
2. Obtain the existing recorder/sample schema and adapt it to version 1. Sessions require `completed: true` once fully written/transferred; preserve source timestamps and separate human corrections from AI motion.
3. User runs the notebook synthetic path on Colab, then supplies completed real sessions for GPU training. No real training is scheduled or started automatically.
4. User runs the Windows acceptance procedure for passthrough, +/-5-degree stationary tracking, takeover, expiry, and cleanup. Tune PD gains only against observed hardware behavior.
5. Connect real capture plus `SteeringPredictor` to the runtime in shadow mode, preserving the frame's capture time through inference. Road-force replay remains a later independent increment.
