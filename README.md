# StormHacks: Forza AI Wheel

**We trained a neural net on our own racing laps, handed it a real steering wheel, and watched it rip.**

An end-to-end driving AI for **Forza Horizon 4**. A PilotNet-style CNN watches the game screen and steers by **physically turning a Thrustmaster TMX force-feedback wheel**. A versioned driving model can also work throttle and brake through vJoy; in steering-only mode the human keeps the pedals. A human can grab the wheel, press a button or touch a pedal to take back control at any moment.

## How it works

```text
              GAME / WHEEL PC                                   INFERENCE PC (desktop GPU)
 ┌────────────────────────────────────────────┐            ┌──────────────────────────────┐
 │ Forza Horizon 4 ──DXcam──► road crop        │──LAN/TCP──►│ PyTorch CNN (PilotNet +      │
 │ Forza Data Out (UDP :9999) ──► speed        │  HMAC-auth │ late speed fusion)           │
 │                                             │◄───────────│ steering (+ throttle/brake)  │
 │ 100 Hz controller ──PD torque──► TMX motor  │            └──────────────────────────────┘
 │ TMX angle + pedals (RawInput) ──► vJoy ──► Forza
 └────────────────────────────────────────────┘
```

- **Brain:** a PilotNet-style CNN (5 conv layers) turns a 200x66 road crop into a steering angle in degrees (right = positive). Forza's speed joins after the conv layers (late fusion). The `driving` task adds throttle and brake outputs.
- **Body:** a local 100 Hz PD controller with static-friction compensation drives the TMX motor toward the AI's target angle through constant-force effects. Torque is capped so a human can always overpower it.
- **vJoy passthrough:** Forza only sees a virtual vJoy wheel. **HidHide** hides the real TMX from everything except our Python process, which reads the TMX and writes steering and pedals to vJoy.
- **Safety:** stale frames, telemetry or network replies, Alt-Tabbing away from Forza, a takeover button, a pedal press or a pull against the motor all disengage the AI. Re-engaging always needs an explicit arm.
- **Learning loop:** behavior cloning on human laps, then **DAgger**. The AI drives, the human corrects, and the corrections become the next round of training data.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ (3.12 on the game PC) |
| ML / training | **PyTorch** (CUDA on the desktop, CPU inference), NumPy, Pillow; optional Google Colab notebook |
| Model | NVIDIA **PilotNet**-style CNN with late fusion of telemetry speed; VisualBackProp saliency for "what the model looks at" |
| Screen capture | **DXcam** (DXGI desktop duplication), **OpenCV** for crop, masks and resize |
| Wheel input & force feedback | **PySDL2** + `pysdl2-dll` (RawInput reading, haptic constant force), Thrustmaster TMX |
| Virtual controller | **vJoy** (BrunnerInnovation 2.2.2.0) via **pyvjoyffb**, **HidHide** to hide the physical wheel |
| Game data | **Forza Data Out** UDP telemetry (speed, gear, race clock, distance, yaw rate, steer) |
| Networking | Authenticated TCP (HMAC with a shared key) between the game PC and the inference PC |
| Operator tools | Live browser dashboard (AI view, intended path HUD, latency and control traces), saved JSON launch profiles, PowerShell launchers, run reports |
| Analytics (optional) | **Tiger Data** (Postgres + TimescaleDB) via `psycopg`, charts with matplotlib |
| Testing | pytest, a simulated wheel backend and loopback network tests (no hardware needed) |

## Repository layout

| Path | Contents |
|---|---|
| `record.py` | Standalone recorder: frames + wheel/pedals + telemetry at 30 FPS, with rewind/pause/crash discarding |
| `review.py`, `sync_check.py` | Review a recording before training, measure frame/label timing |
| `src/forza_ai/training/` | `forza-train`: import, validate, train, evaluate, export |
| `src/forza_ai/runtime.py` | `forza-run`: live capture, inference client, wheel control, takeover, recording |
| `src/forza_ai/inference_server.py` | LAN inference server for exported models |
| `src/forza_ai/control/`, `hardware/` | PD steering controller, SDL/vJoy/Windows device backends |
| `src/forza_ai/dashboard.py`, `launch.py` | Browser dashboard, profile launcher (`forza-launch`) |
| `src/forza_ai/analytics/` | Tiger Data analytics (`forza-analytics`) |
| `configs/`, `config/` | Launch profile examples, capture config (`config/capture.json`) |
| `scripts/` | `start-game.ps1`, `start-desktop.ps1`, wheel tests, Colab helpers |
| `utils/test.py` | Original hardware diagnostic (DirectInput: only run with Forza closed) |
| `notebooks/train_colab.ipynb` | Colab training fallback |

## Setup

### 0. Try it with no hardware

Any OS, Python 3.10+:

```sh
python -m venv .venv
python -m pip install -e ".[training,test]"
forza-train synthetic /tmp/forza-fixture --sessions 3 --frames 24
forza-train train /tmp/forza-fixture /tmp/forza-run --epochs 1 --device cpu
python -m forza_ai.runtime --backend sim --assist --duration 5
python -m pytest -q
```

This proves the pipeline runs. It doesn't prove the model can drive: the synthetic data has no roads, and the sim backend holds a fixed test angle.

### 1. Install each machine

Use a separate virtual environment on each PC (PowerShell, from the repo root):

```powershell
python -m venv .venv
# Inference / training PC: PyTorch, no wheel drivers
.\.venv\Scripts\python.exe -m pip install -e ".[training]"
# Game / wheel PC instead: capture and wheel packages, no PyTorch
.\.venv\Scripts\python.exe -m pip install -e ".[hardware]"
```

Optional extras: `analytics` (Tiger Data), `test` (pytest). On the desktop, check CUDA with `python -c "import torch; print(torch.cuda.is_available())"`; if it prints `False`, install a CUDA build from the [PyTorch selector](https://pytorch.org/get-started/locally/).

### 2. Configure the game PC (one time)

1. **Thrustmaster TMX:** 900° rotation, pedals set to Separate, autocenter turned off in the Thrustmaster control panel.
2. **vJoy:** install vJoy 2.2.2.0 and configure device 1 with axes X Y Z Rx Ry Rz Slider Dial, 32 buttons, 1 POV and all FFB effects. The installer may hang at 100% on the KMDF step; the driver is installed, so end the setup processes. Set vJoy's registry `OEMData` to `43 00 88 01 fe 00 00 00` so Forza lists it as a wheel (back up the original value first; don't edit Forza's media zips, which crashes the game).
3. **HidHide:** hide the TMX and add the `python.exe` that opens the wheel (your `.venv` one) to the allowlist.
4. **Forza Horizon 4:** custom wheel layout with steering = vJoy X, brake = Y, gas = Z, combine accel/brake OFF. Bind every other row to placeholder vJoy buttons, or Forza won't save the layout. Turn **Data Out** on at `127.0.0.1:9999`. Recommended: bonnet camera, HUD and racing line off.
5. **Check the wheel with Forza closed** (find your takeover button index with `python utils/test.py wheel` first):

   ```powershell
   python -m forza_ai.preflight --json
   .\scripts\windows-wheel-test.ps1 -TakeoverButton BUTTON_INDEX
   ```

### 3. Record laps

On the game PC, with Forza in front and driving:

```powershell
python record.py record            # writes data/recordings/<timestamp>/
python review.py                   # per-segment summary + review.png of suspect frames
python sync_check.py               # checks frame/label timing
```

Copy finished recordings to the inference PC and import them. Use at least two independent recordings so validation is held out:

```powershell
forza-train import-recording SOURCE_RECORDING data/completed/recording-001 --expert-mode manual
forza-train validate data/completed
```

### 4. Train and export

```powershell
forza-train train data/completed runs/driving-v2 --task driving --device cuda --epochs 10
forza-train evaluate runs/driving-v2/best.pt data/completed
forza-train export runs/driving-v2/best.pt models/driving-v2
```

Leave out `--task driving` for a steering-only model.

### 5. Drive over LAN

Set the same random key on both PCs, kept out of files and commands:

```powershell
$env:FORZA_LINK_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Shared LAN key' -AsSecureString)).Password
```

Inference PC (allow Python through the firewall on the private network; default port TCP 8765):

```powershell
python -m forza_ai.inference_server --bind INFERENCE_PC_IP --model models/driving-v2
```

Game PC: start in `--shadow` (predictions only, no AI torque), then switch to `--assist`:

```powershell
python -m forza_ai.runtime --backend windows --inference-host INFERENCE_PC_IP --capture-config config/capture.json --takeover-button SDL_INDEX --arm-button OTHER_SDL_INDEX --shadow --interactive --duration 0 --dashboard-port 8080
```

Add `--auto-pedals --assist` to let a driving model work the pedals. Add `--record-session runs/session-001` to save takeover corrections for DAgger. Instead of long commands, use the saved profiles and launchers in [configs/README.md](configs/README.md): `.\scripts\start-game.ps1 -Profile .\configs\game.local.json`.

## Status

Implemented: recording, import and review; training and export (steering-only and steering + pedals); live capture; LAN inference; the 100 Hz wheel controller with takeover and expiry; integrated correction recording; the dashboard, reports, profiles and launchers; and analytics. Software tests cover the simulated wheel, loopback networking and failure paths. Real-world performance depends on hardware tuning and the recordings available; see the results docs below.

## Docs

- [Two-PC setup](docs/TWO_PC_SETUP.md)
- [Automatic pedals: training and launch](docs/AUTOMATIC_PEDALS.md)
- [Training, dataset transfer and inference](docs/TRAINING.md)
- [First steering/throttle/brake training results](docs/DRIVING_BASELINE_RESULTS.md)
- [Operator controls, dashboard, correction recording and reports](docs/OPERATIONS.md)
- [Saved launch profiles](configs/README.md)
- [Recording and model contracts](docs/CONTRACTS.md)
- [Adapter usage and Windows acceptance](docs/ADAPTER.md)
- [Model-input and VisualBackProp saliency dashboard](SALIENCY.md)
- [Driving analytics on Tiger Data](docs/ANALYTICS.md)
- [Delivery plan](docs/PLAN.md)
- [Harness review and fixes](docs/HARNESS_REVIEW.md)
- [Colab notebook](notebooks/train_colab.ipynb)
