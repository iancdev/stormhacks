# Forza AI Wheel: project context

Hobby physical-AI project: a neural network steers a car in **Forza Horizon 4 (Steam)** by physically turning a **Thrustmaster TMX** force-feedback wheel. The human keeps the pedals and can grab the wheel to override at any time.

## Architecture (decided)
- **Brain:** PilotNet-style CNN, screen frame → steering angle (degrees, **right = positive**). Speed from telemetry can be fused in after the conv layers (late fusion). Imitation learning first, then DAgger, optional RL fine-tune later.
- **Body:** PID loop turns the target angle into motor torque via force feedback. The game only ever sees the wheel angle.
- **vJoy passthrough:** Forza is bound only to the virtual vJoy wheel. Our Python script:
  1. reads the real TMX (PySDL2) and writes steering/pedals to vJoy (pyvjoyffb),
  2. receives Forza's FFB from vJoy and forwards it, plus the AI torque, to the TMX motor (capped, with a kill switch).
- **HidHide** hides the TMX from everything except python.exe, so Forza only sees vJoy.
- Inference runs on CPU during live driving so Forza keeps the GPU (GTX 1650, 4 GB). Training on GPU, possibly on the laptop (RTX 3050).

## Machine
- Project folder: `C:\Users\Administrator\ForzaTest`, venv `.venv` (Python 3.12).
- Activate: `Set-ExecutionPolicy -Scope Process Bypass; .\.venv\Scripts\Activate.ps1`
- Packages: pysdl2, pysdl2-dll, pyvjoyffb (imports as `pyvjoy`), dxcam, opencv-python, numpy, pyzmq.
- `check.py` has hardware tests: `wheel`, `ffb`, `vjoy [seconds]`, `bind steer|brake|gas`, `bindall [first] [gap]`, `telemetry`.

## TMX calibration (measured)
- Firmware 15 (staying on it), 900° rotation, pedals Separate, 2 pedals (no clutch).
- SDL axes (raw -32768..32767): **a0 steering** (left -, right +, ~73 units/deg), **a1 brake**, **a2 gas** (both +32767 released, -32768 floored), a3 unused.
- Pedal normalize: `(32767 - raw) / 65535` → 0 released, 1 floored.
- SDL haptics: 1 axis, CARTESIAN direction only (dir[0]=1), gain 100 supported, autocenter NOT settable via SDL (turn off in Thrustmaster panel).
- **Constant force +level pushes LEFT, -level pushes RIGHT** (opposite to the angle sign). PID torque: `level = -Kp * (target - angle)`.
- Forza grabs the motor exclusively while running (even paused), hence the passthrough.

## vJoy / Forza setup (working)
- vJoy BrunnerInnovation v2.2.2.0, device 1, ID `VID_1234&PID_BEAD`. Axes X Y Z Rx Ry Rz Slider Dial, 32 buttons, 1 POV, all FFB effects on.
- Installer hangs at 100% on the KMDF step but the driver installs fine; end the setup processes.
- `vjoy_as_wheel.reg` (OEMData `43 00 88 01 fe 00 00 00`) makes FH4 list vJoy as a wheel. Undo: `vjoy_as_wheel_UNDO.reg`.
- **Do not edit Forza's media zips**: it crashed the game. Originals are backed up as `*.ORIGINAL.zip`.
- FH4 custom WHEEL layout: Combine accel/brake OFF, combine steering ON. Steering = vJoy X (Axis 1), Brake = Y (Axis 2), Gas = Z. Every other row bound to vJoy buttons 1..N as placeholders (Forza won't save otherwise).
- **vJoy axis contract:** X steering 0x4000 centre (left low, right high); Y brake and Z gas **inverted**: 0x8000 released, 0x1 floored. Scripts must set pedals to 0x8000 on start and exit (vJoy defaults to mid-axis = half gas/half brake).
- Forza's FFB to vJoy: one **constant force** effect (type 1) updated ~60 Hz, magnitude about ±4000 of ±10000. Periodic effects arrive with gain 0 (ignore). No spring/damper.

## Telemetry
- Data Out ON, `127.0.0.1:9999`. Packets are 324 bytes (FH4 "dash" layout), ~60/s while driving.
- Offsets used in check.py: IsRaceOn int @0, CurrentEngineRpm float @16, Speed float m/s @256, Steer s8 @320. **Speed/steer offsets not yet verified while driving.**

## Data collection plan
- Solo circuit (Rivals/time attack, ghost off if possible), one mid-range B/A-class car, automatic gears, bonnet camera, HUD off, racing line off, lens effects off.
- One loop iteration = frame + wheel + latest telemetry + one timestamp. Crop sky/bonnet, resize ~200×66. Shift labels ~100–200 ms later (tune). Check frame-gap histogram before long recordings. Balance near-straight frames.
- 1–2 h base laps → train → DAgger rounds (AI drives, human corrects) → test on an unseen circuit.

## Status / next
- DONE: Python/VS Code, TMX calibration, FFB test, vJoy + registry flag, Forza wheel layout, Forza FFB reaches vJoy, telemetry packets arrive, HidHide installed with both python.exe paths allowed.
- NEXT: tick TMX in HidHide Devices + enable hiding + replug; verify telemetry values while driving; write the passthrough script (TMX → vJoy, Forza constant force → TMX motor, sign to calibrate); then the recorder.

## Working rules
- Test FFB with no game, and vJoy passthrough with no AI. Never debug both at once.
- Don't edit a file in VS Code while another tool is rewriting it (VS Code saved over check.py once).
