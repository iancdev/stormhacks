# Delivery plan

Updated 2026-10-03. This file is the durable plan for both project chats.

## Objective

Train a vision policy offline from human driving recordings, then continuously steer a real Thrustmaster TMX in Forza Horizon 4 while the human controls the pedals. The game receives the measured physical angle through vJoy. Takeover interrupts assistance; it is not the normal driving mode once assistance is engaged.

## Priorities and ownership

1. **Training pipeline first:** chat `01a103b0-2b07-7c73-8934-0324136bc8f6` owns dataset validation/loading, training, evaluation, export, a portable inference wrapper, and the Colab fallback. It works in its own worktree and branch. Training setup now targets `DESKTOP-0HR4O88` first.
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

The wheel PC records frames, wheel inputs, telemetry, and control mode into completed sessions. The user confirmed on 2026-10-03 that the wheel is on a different PC from Codex's connected `DESKTOP-0HR4O88`, and selected that connected desktop for **both GPU training and live inference**. Colab remains a training fallback. GitHub stores code and tiny synthetic fixtures; session archives and checkpoints are transferred separately. GPU training consumes completed archives, validates them, and writes resumable checkpoints. The game PC sends road crops and causally matched speed over LAN; the desktop returns physical target angles. The fast physical controller, pedals, takeover, and stale-command shutdown remain on the wheel PC. No cross-machine clock comparison is used. Desktop GPU/PyTorch availability and real two-PC operation are not yet verified. See [TWO_PC_SETUP.md](TWO_PC_SETUP.md).

## Integration and verification

- Preserve the tested `utils/test.py` as a diagnostic reference.
- Test pure conversions, command expiry, takeover/re-engagement, controller saturation, decoder validation, and exception cleanup without hardware.
- Verify on Windows: actual passthrough, force direction, bounded stationary angle tracking, takeover and stale-command release. Passing software tests cannot establish hardware stability.
- Run shadow inference before model-controlled driving; evaluate repeated route completion and interventions before lap time.
- Human takeover recordings are expert labels; AI-generated motion is not automatically expert data.

## Current status

- Private repository created and pushed: `iancdev/stormhacks`.
- Hardware diagnostic and pinned Windows dependencies received in commit `d18ec40`.
- Training work delivered by the designated chat and integrated through `65d1917`: session validation/alignment, image-plus-speed CNN, resumable CLI, metrics/baselines, CPU export/predictor, executable Colab ZIP setup, synthetic notebook smoke, and interruption-safe best-checkpoint recovery. Real data and actual GPU/Colab execution remain outstanding.
- Adapter software includes a PD controller, explicit engagement/takeover, command expiry, fixed and sweep test policies, simulated wheel, timestamped FH4 receiver with causal lookup, fresh DXcam capture, foreground-process checks, and local/remote exported-model integration. The LAN client sends authenticated lossless crops and speed to a desktop server, retains local source times, and rejects expired or incorrectly correlated replies. Physical Windows and two-PC acceptance remain outstanding; see [TWO_PC_SETUP.md](TWO_PC_SETUP.md).
- The supplied baseline reference is NVIDIA's [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). [TRAINING.md](TRAINING.md) documents our RGB preprocessing, speed input, and physical-angle output differences.
- Latest combined validation on this Mac: **172 tests and 141 subtests passed**, including synthetic notebook execution, real TCP loopback image/angle transport, nonce/correlation checks, timeout/concurrent-close cases, remote predictions driving the simulated wheel, network-loss disengagement, and capture/foreground/telemetry failure paths. No real dataset, desktop GPU setup, actual two-PC network, or physical wheel validation was performed here.

## Next integration gates

1. Obtain the existing recorder/sample schema and adapt it to version 1. Sessions require `completed: true` once fully written/transferred; preserve source timestamps and separate human corrections from AI motion.
2. Set up the portable training CLI on `DESKTOP-0HR4O88`, verify CUDA availability, then transfer completed real sessions for training. Colab notebook is a fallback. No real training is scheduled or started automatically.
3. User runs the Windows acceptance procedure for passthrough, +/-5-degree stationary tracking, takeover, expiry, and cleanup. Tune PD gains only against observed hardware behavior.
4. Live capture, telemetry matching, local model wrapper, and LAN inference client/server are implemented. Verify the small fixed-target network path first, then load the exported real policy on the desktop and run shadow mode with source capture times preserved. Road-force replay remains a later independent increment.
