## What it does

A neural network drives **Forza Horizon 4** by watching the screen, and it steers by **physically turning a real Thrustmaster TMX force-feedback wheel**. The wheel rotates by itself in front of you, and you can grab it at any moment to take back control.

- **Vision-based driving:** a CNN reads the live game image plus the car's speed and outputs steering, throttle and brake, about 30 times a second.
- **The wheel moves on its own:** a motor control loop running at 100 Hz drives the TMX to the AI's steering angle.
- **A human stays in charge:** press a wheel button, touch a pedal or grab the wheel, and control is back with you within a fraction of a second. Press another button to hand it back to the AI.
- **Live dashboard:** the AI's view of the road, with its intended steering path drawn on top as a HUD, plus latency and control traces.
- **It learns from your corrections (DAgger):** when the AI makes a mistake, you rewind, take over and drive it properly. Those corrections become the next round of training data.

## How we built it

### 1. A CNN in one paragraph
A **convolutional neural network** slides small learned filters across an image. Early layers detect simple patterns like edges and colour changes; deeper layers combine them into lane lines, road edges and curves. A few fully connected layers at the end turn those features into a decision, here a steering angle.

### 2. PilotNet
We based the model on **NVIDIA's PilotNet** (*End to End Learning for Self-Driving Cars*, 2016), which learns to steer directly from camera images by imitating a human driver:

| Layer | PilotNet |
|---|---|
| Input | 66 × 200 image |
| Conv 1–3 | 24, 36, 48 filters, 5×5, stride 2 |
| Conv 4–5 | 64, 64 filters, 3×3 |
| Fully connected | 100 → 50 → 10 → 1 |
| Output | steering |

### 3. Our version: late fusion with game telemetry
One image doesn't tell you how fast you're going, and the same corner needs very different steering and braking at 80 km/h than at 200 km/h. So we **fused Forza's live telemetry** into the network:

```
road crop (320×66 → 200×66 RGB) ──► 5 conv layers (PilotNet) ──► 1152 image features ─┐
Forza speed (UDP telemetry) ─────────────────────────────────────────────────────────┴─► concat ──► FC 100 → 50 → 10
                                                                                                     └─► steering, throttle, brake
```

- **Late fusion:** the image is processed on its own, and speed joins just before the decision layers.
- **Three outputs:** steering angle (in degrees of real wheel rotation), throttle and brake.
- **Training data:** we recorded **over an hour of laps** at 30 fps, with every frame time-stamped together with the wheel, pedals and telemetry. We **measured the screen running 66 ms behind telemetry** by matching gear changes on screen against telemetry.
- **Cleaning:** crashes, rewinds and game pauses are detected automatically and cut from the data, including a buffer before each one.

### 4. Two PCs: inference over Ethernet

| Game PC | Laptop |
|---|---|
| Forza Horizon 4, screen capture (DXcam) | model inference (PyTorch) |
| TMX wheel, vJoy, 100 Hz control loop | ~5 ms per prediction |
| live dashboard | |

Frames and predictions travel over a **direct Ethernet cable**, authenticated with a shared key. **About 45 ms** from screen capture to the steering command coming back.

### 5. The wheel loop: TMX ↔ our runtime ↔ vJoy ↔ Forza

```
            ┌──────────── reads angle + pedals (RawInput) ────────────┐
 TMX wheel ─┤                                                         ├─► our runtime ──► vJoy (virtual wheel) ──► Forza
            └─◄─ motor torque (constant-force FFB, PD control) ◄──────┘        ▲
                                                                                 └── AI prediction (from the laptop)
```

- **HidHide** hides the real TMX from Forza, so the game only ever sees **vJoy**, a virtual wheel our runtime controls.
- **Manual driving:** the runtime forwards your wheel and pedals to vJoy.
- **AI driving:**
  - the AI's steering goes to Forza through vJoy,
  - a **PD controller with friction compensation** turns the physical TMX to match, using constant-force effects,
  - the pedals are AI-controlled, with throttle ramping and a corner speed limit.
- **Takeover:** a wheel button, a pedal, or pulling the wheel away from the AI's angle.
- **Safety:** if anything goes stale (frames, telemetry, network), the AI disengages. A torque cap keeps the wheel easy to overpower.

### Stack
| Area | Tools |
|---|---|
| ML | **Python**, **PyTorch**, NumPy |
| Capture & vision | **DXcam** (DXGI screen capture), **OpenCV** |
| Wheel & input | **PySDL2** (RawInput reading, DirectInput force feedback), **pyvjoyffb**, **vJoy**, **HidHide** |
| Game data | **Forza Data Out** UDP telemetry (speed, gear, race clock, yaw rate) |
| Networking | authenticated TCP over a direct Ethernet link |
| Tooling | data review and sync-check scripts, live web dashboard, DAgger correction recorder |

## Challenges we ran into

- **Reading the wheel killed its force feedback.** Our first recorder opened the TMX through DirectInput, which grabs force-feedback wheels exclusively, and Forza's force feedback went dead. We switched to **read-only RawInput**.
- **A 15.6 ms clock broke the control loop.** On Windows with Python 3.12, `time.monotonic()` only ticks every 15.6 ms, so our 100 Hz loop saw repeated timestamps and treated them as a frozen loop. We **switched to the high-resolution clock**.
- **The wheel motor fell seconds behind.** Resending motor commands every 10 ms made the TMX **queue** them: the delay grew from 80 ms to 260 ms, and the wheel swung up to 350°. **Throttling commands** to send only the latest force brought tracking error from **134° down to 9°**.
- **The wheel either wouldn't move or overshot.** The TMX needs about 20% force just to start moving, then spins freely. We added **static-friction compensation** and strong damping.
- **The AI learned to drive at a sign.** The model steered toward the big "HORIZON" signs, because our crash recordings had taught it "sign in view, steer toward it". We built **automatic crash, rewind and pause detection**, with adaptive buffers that cut the run-up to every reset.
- **Rewinds were invisible.** In some game modes Forza resets the car without turning the race clock back. We detect them from **distance travelled and pauses** instead.
- **False takeovers.** In hard corners the motor lags behind the AI's target, which at first looked like a human grabbing the wheel. We now detect a grab by **which way the wheel is moving**.
- **Network dropouts.** A single late prediction used to disengage the AI. We added **auto re-engagement after short network blips** and per-disengagement logging to find the root causes.
- **Understeer and over-aggressive driving.** We tuned it with a **steering gain**, **throttle ramping** and a **corner speed limit**, then improved the model itself with **DAgger** correction rounds.
