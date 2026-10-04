## What it does

An **embodied AI driver**, kind of: instead of sending keystrokes to a game, a neural network drives **Forza Horizon 4** through a **physical body**. It watches the screen like a player, decides how to steer, and then **physically turns a real Thrustmaster TMX force-feedback wheel**, the same wheel a human would hold. The world is simulated, but the hands are real: the wheel moves on its own, the AI also works the gas and brake, and you can grab the wheel, tap a pedal or press a button to take over at any moment. A live dashboard shows the AI's view of the road with its steering path drawn on top.

## How we built it

**Model:** a CNN based on **[NVIDIA's PilotNet](https://developer.nvidia.com/blog/researching-and-developing-an-autonomous-vehicle-lane-following-system/)**: 5 convolutional layers that learn road features (edges, lane lines, curves), then fully connected layers that turn them into driving outputs. We added **late fusion** of Forza's live telemetry: the car's speed joins the image features right before the decision layers, because the same corner needs different steering and braking at different speeds.

```
road crop (66×200) ──► 5 conv layers ──► image features ─┐
Forza speed (telemetry) ─────────────────────────────────┴─► FC 100 → 50 → 10 ──► steering, throttle, brake
```

**Wheel loop:** HidHide hides the real wheel from Forza, so the game only sees **vJoy**, a virtual wheel we control.

```
TMX wheel ──(angle, pedals)──► our runtime ──► vJoy ──► Forza
    ▲                              │
    └──── motor turns the wheel ◄──┘ ◄── AI prediction (laptop, over Ethernet)
```

**Stack:** Python, PyTorch, DXcam, OpenCV, PySDL2, vJoy + HidHide, Forza Data Out telemetry, two PCs over a direct Ethernet link.

## Challenges we ran into

- **Syncing frames with telemetry.** Every frame, wheel reading and telemetry packet is timestamped on one clock. We measured that the screen shows the game **66 ms after** telemetry reports it, by matching gear changes on screen against gear changes in the telemetry.
- **Hot-lap data doesn't teach recovery.** We mostly recorded fast laps, minimising lap time and chasing apexes. So the model never saw what to do once it drifted off the line, and it couldn't recover after going off the road.
- **Overfitting.** We tuned the driving (steering gain, throttle ramping, a corner speed limit) and added **DAgger**: when the AI makes a mistake, we rewind, take over and drive it properly, and those corrections become new training data.
- **Driving at signs instead of lanes.** The network learned a shortcut: steer toward the big "HORIZON" signs rather than follow the lane markings. Crash run-ups in our recordings had taught it that. We cut them from the data automatically.
