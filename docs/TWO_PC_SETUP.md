# Two-PC setup: desktop training and inference, separate game/wheel PC

The user-selected deployment is:

For daily use, [saved profiles](../configs/README.md) replace repeated long CLI
commands, and [OPERATIONS.md](OPERATIONS.md) covers wheel re-engagement, the
dashboard, integrated correction recordings, and run comparisons.

```text
Game + TMX PC                              DESKTOP-0HR4O88
road crop + causal speed  -- LAN request --> model inference
local capture timestamp  <-- same request's target angle --
       |
local 100 Hz controller -> TMX motor
measured TMX angle + human pedals -> vJoy -> Forza
```

The controller stays with the physical wheel. Network and model work run outside
that loop. Only one frame is in flight; no queue of old images or angle commands
accumulates. Predictions retain their original capture timestamp on the game PC.
No assumption is made that the two computers' monotonic clocks match.

## 1. Prepare each machine

Pull `main` in the project checkout. Use a separate virtual environment on each
machine. From the project root in PowerShell:

```powershell
python -m venv .venv
# Desktop: training and inference dependencies; no wheel driver packages.
.\.venv\Scripts\python.exe -m pip install -e ".[training]"

# Game/wheel PC, instead: capture and wheel dependencies; no Torch required.
.\.venv\Scripts\python.exe -m pip install -e ".[hardware]"
```

Use the appropriate command on each machine, not both by default. The wheel PC
also needs the already-tested Thrustmaster/vJoy/HidHide configuration. HidHide's
allowlist must include the Python executable that actually opens the wheel.
Changing environments may require updating that allowlist.

On the desktop, verify the training GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device detected')"
```

If CUDA is unavailable, install the appropriate build using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) and
verify again. The inference predictor currently runs on CPU on the desktop;
the small network does not require GPU inference. Use `--device cuda` for real
training once verified. Colab remains an optional fallback.

## 2. Verify physical motion before involving the network

Use the wheel PC's existing working Python environment, or activate the new one.
Identify the zero-based takeover button using `python utils/test.py wheel`.
Close Forza for this first stationary test, and keep the wheel clear when opening
its driver: SDL/Thrustmaster may enable native centering during initialization.
Start with the wheel near centre. Do not interrupt an active data-collection run;
run this check once the wheel is available.

```powershell
python -m forza_ai.preflight --json
.\scripts\windows-wheel-test.ps1 -TakeoverButton BUTTON_INDEX
```

Replace `BUTTON_INDEX` with the verified number. The script checks discovery,
then requests centre/right/centre/left/centre at +/-5 degrees. It records the
requested target, limited controller target, measured wheel angle, torque, mode,
and timing in a new `runs/wheel-test-.../` directory.

Preflight does not open the wheel, because SDL opening itself can change native
centering. It cannot prove pedal calibration, haptic behavior, or that the selected
button does what you expect. The runtime verifies the actual button range before
applying its effect. A successful process exit is not proof that the wheel moved.
Check direction, smooth tracking, and immediate removal of our AI torque on the
takeover button. Driver-native centering is distinct from the effect we command.

If script execution is disabled, run the underlying command directly:

```powershell
python -m forza_ai.runtime --backend windows --assist --sweep --duration 12 --takeover-button BUTTON_INDEX --status-csv runs/wheel-control.csv
```

Default gains are untuned: `--kp 0.008 --kd 0.001 --torque-limit 0.15`.
The CLI also exposes `--target-rate` and `--target-limit`. Check small targets
before changing gains or increasing authority. No game road-force replay is
included in this increment.

## 3. Connect the machines over LAN

Use the desktop's reachable LAN IPv4 address; the game PC connects to it on TCP
8765 by default. Permit the server's Python executable on the desktop's private
network if Windows Firewall prompts. No router port forwarding is needed.

Set the **same random shared key** as `FORZA_LINK_KEY` in both terminals. Generate
one with Python's `secrets.token_hex(32)` locally, then enter it on each machine:

```powershell
$env:FORZA_LINK_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Shared LAN key' -AsSecureString)).Password
```

Keys stay out of commands, repository files, and diagnostic logs. The protocol
authenticates bounded messages with HMAC; it does not encrypt the road images.
Use the trusted local network. Requests include session/request IDs and responses
must match. Socket deadlines, stale observations, disconnects and paused/missing
game input stop AI assistance; a fresh reply alone cannot re-engage it.

For a no-dataset network test, start this on the desktop:

```powershell
python -m forza_ai.inference_server --bind DESKTOP_LAN_IP --test-target 5
```

This is a fixed-angle test server, not a driving model. On the wheel PC, start
Forza, enter a parked active driving scene, enable Data Out at `127.0.0.1:9999`,
and keep the game in front. First use shadow mode:

Only one process should own the wheel and listen on Data Out port 9999. Stop a
separate recorder/old adapter before starting this runtime, or coordinate a
different configured Data Out port. Use the runtime's `--record-session` option
for integrated manual/correction recording without a second device listener;
see [OPERATIONS.md](OPERATIONS.md).

```powershell
python -m forza_ai.runtime --backend windows --inference-host DESKTOP_LAN_IP --capture-config config/capture.json --shadow --duration 30 --takeover-button BUTTON_INDEX --status-csv runs/network-shadow.csv
```

All uppercase placeholders must be replaced. With the incoming `record.py`, use
its `config/capture.json`: this loads the monitor, exact crop, HUD masks, and
OpenCV area resize to the recorder's saved dimensions. Capture is RGB rather than
the recorder's BGR working array; channel-independent masking/resizing preserves
the RGB image the training loader sees after decoding the recorder's JPEGs.
Live frames are not JPEG-recompressed. The desktop then applies the exported
model's own preprocessing (including the final 200x66 resize).

For another recorder that uses unmasked crops, `--crop LEFT TOP RIGHT BOTTOM`
and `--display` are available instead. These are absolute output-local pixel
coordinates, not the model's resized dimensions. Use identical camera/crop settings
for recording and inference. The game PC sends lossless RGB crops over LAN.

Live modes require `ForzaHorizon4.exe` to be the foreground process by default.
Use `--game-process` only if the actual game EXE differs. Alt-Tabbing disengages
assistance. After returning to the game, re-engage explicitly; issuing `arm` in
the terminal gives a five-second window to return to Forza and obtain valid
input. Prefer a configured `--arm-button` for re-engagement without leaving the
game. The check is based on the process EXE, not a matching browser/window title.

Then, while stationary, replace `--shadow` with `--assist` for a small physical
rightward target from the desktop. The policy waits for valid input during the
initial engagement window. `--interactive` allows `arm`, `manual`, and `quit`.
If the initial window expires, re-engage explicitly when the game is ready.
Stop the desktop server during an assisted test: AI torque must stop and remain
off when the server returns, until explicit engagement. The human pedals and
measured wheel still pass through locally while assistance is disengaged.

## 4. Train, then replace the test server with the exported policy

Transfer completed sessions to the desktop. Follow [TRAINING.md](TRAINING.md),
using local dataset/checkpoint paths and `--device cuda` after verifying CUDA.
Train/validation sessions remain separate. Export the selected checkpoint.

For the incoming `record.py`, copy each complete `data/recordings/TIMESTAMP/`
directory (JPEGs, `labels.csv`, and `meta.json`) to the desktop, then import it:

```powershell
forza-train import-recording SOURCE_RECORDING data/completed/recording-001 --expert-mode manual
forza-train import-recording SECOND_RECORDING data/completed/recording-002 --expert-mode manual
forza-train validate data/completed
forza-train train data/completed runs/baseline-001 --epochs 10 --device cuda
forza-train export runs/baseline-001/best.pt runs/baseline-001/export-best
```

Use at least two separate recordings. Pause segments from one recording stay in
one split group. The importer preserves the source JPEG/CSV bytes and recorded
per-frame labels; it cannot recover certified capture times from rounded elapsed
times and ages. These imports support only zero label offset. Their timing
limitations are carried into validation reports and model exports.

```powershell
python -m forza_ai.inference_server --bind DESKTOP_LAN_IP --model PATH_TO_EXPORTED_MODEL
```

The game PC command stays the same. Start in shadow mode to check latency and
predictions; use assistance only after real wheel acceptance and model evaluation.
The default request timeout is 200 ms (`--network-timeout`), and the controller
independently rejects old source observations. Increasing a network timeout does
not disable the controller's freshness bound.

The default desktop server is CPU inference, preserving the portable exported
model's behavior. GPU training and CPU inference can live on the same desktop;
benchmark live latency before training and inference compete for its resources.

## Verification status

Software tests exercise real loopback transport, capture/predictor boundaries,
simulated physical control, takeover, and failure paths. The actual Windows
devices, two-PC network, GPU, and road-driving performance require checks on
the user's machines. Dataset availability is not needed for the stationary and
fixed-target network tests above.
