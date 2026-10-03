# StormHacks — Vision-Based Steering Assist

A driving AI for Forza Horizon 4 that learns from human demonstrations and physically steers a Thrustmaster TMX force-feedback wheel. The human controls throttle and brake and can take over steering.

## Initial scope

- One car, one repeatable route, consistent camera and weather.
- Racing line disabled; moderate human-controlled speed.
- A small PyTorch CNN predicts steering from a road image and vehicle speed.
- A local controller converts the target wheel angle into bounded motor torque.
- The measured physical wheel angle and human pedal inputs are forwarded to Forza through vJoy.
- An explicit takeover button disables AI steering torque and marks human correction data.

## Architecture

```text
Game/wheel PC: road crop + speed → LAN → Desktop: CNN inference
Game/wheel PC: physical controller ← LAN ← Desktop: target angle
Physical controller → TMX motor → measured angle → vJoy → Forza
Human pedals → vJoy → Forza
```

The physical wheel remains in the steering loop. Direct model-to-vJoy steering can serve as a separate software baseline.

## Planned stack

Windows gaming PC, Forza Horizon 4, Thrustmaster TMX and pedals, Python, PySDL2, vJoy with an FFB-capable Python wrapper, HidHide, DXcam, OpenCV, and PyTorch. Forza Data Out provides UDP telemetry. Exact dependency versions and compatibility will be verified during implementation.

## Milestones

1. Verify manual wheel/pedal passthrough and live telemetry values.
2. Implement physical angle tracking with calibrated signs, torque limits, takeover, and stale-command shutdown.
3. Record timestamped images, wheel inputs, and telemetry; validate synchronization on a short session.
4. Train a behavior-cloning baseline and evaluate on held-out laps or sessions.
5. Test closed-loop driving, collect marked human corrections, and retrain.
6. Prepare a demo showing completion rate, interventions per minute, and steering stability.

Overtaking, navigation, and reinforcement learning are future extensions.

## Status

The tested hardware diagnostic is in `utils/test.py`, and the incoming recorder is `record.py`. Offline training, recorder import, physical control, matching camera/telemetry input, and authenticated LAN inference are implemented. The desktop trains and serves predictions; the game/wheel PC captures input and controls the wheel locally. The 210-test suite covers synthetic training, recorder compatibility, loopback transport, simulated steering, and failure paths. Real sessions are pending, and actual Windows movement, desktop GPU, and two-PC operation still require acceptance.

- [Durable delivery plan and ownership](docs/PLAN.md)
- [Shared recording and model contracts](docs/CONTRACTS.md)
- [Adapter usage and Windows acceptance](docs/ADAPTER.md)
- [Training, dataset transfer, and inference](docs/TRAINING.md)
- [Colab notebook](notebooks/train_colab.ipynb)
- [Primary two-PC setup: desktop training/inference and game/wheel control](docs/TWO_PC_SETUP.md)

## Start training development

Use Python 3.10+ in a virtual environment. This small CPU smoke run needs no wheel, game, or real recording:

```sh
python -m pip install -e '.[training,test]'
forza-train synthetic /tmp/forza-fixture --sessions 3 --frames 24
forza-train validate /tmp/forza-fixture
forza-train train /tmp/forza-fixture /tmp/forza-run --epochs 1 --device cpu
forza-train resume /tmp/forza-run/last.pt /tmp/forza-fixture --epochs 2 --device cpu
forza-train evaluate /tmp/forza-run/best.pt /tmp/forza-fixture
forza-train export /tmp/forza-run/best.pt /tmp/forza-export
python -m pytest -q
```

Use new output paths for a new smoke run; use `resume` for an existing run. Synthetic success proves the pipeline executes, not that the model can drive. In Colab, the notebook can upload a private GitHub repository ZIP and run the synthetic path before real data exists. For real data, it extracts completed session archives from Drive to the runtime's local disk, while keeping checkpoints on Drive.

## Run the adapter without hardware

```sh
python -m forza_ai.runtime --backend sim --assist --duration 5
```

This is a fixed-angle stationary test policy. It has no road vision. See the adapter guide before switching to Windows hardware. The inspected Windows driver bindings are pinned in the optional `hardware` extra; the original `requirements.txt` belongs to the diagnostic environment, not Colab.
