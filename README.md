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
Forza image + speed → CNN → target wheel angle → torque controller → TMX motor
Forza ← vJoy ← measured TMX angle + human pedal inputs
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

Overtaking, navigation, reinforcement learning, and network-separated inference are future extensions.

## Status

Planning stage. Prior hardware experiments reportedly demonstrated wheel input, motor actuation, and Forza/vJoy communication separately. Those scripts are not yet in this repository; integrated operation and live telemetry decoding still require verification.
