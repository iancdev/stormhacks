# Saved launch profiles

Copy the appropriate example to `game.local.json`, `desktop-test.local.json`, or
`desktop-model.local.json` in this directory. Fill required `null` values using
the actual setup. The examples deliberately contain no guessed IP, capture
configuration, model, or takeover button. JSON comments and unknown fields are
rejected. Keep `FORZA_LINK_KEY` in the process environment, never in these files.

Every path in a profile resolves **relative to that profile's directory**. For
example, `"config_path": "../config/capture.json"` uses the recorder's saved crop,
monitor, masks, and resize. The capture JSON must already exist and validate.
`model_path` must be an exported directory containing `model.pt` and
`metadata.json`, not a training checkpoint file.

First check configuration without starting a process or opening hardware:

```powershell
.\scripts\start-desktop.ps1 -Profile .\configs\desktop-test.local.json -Check
.\scripts\start-game.ps1 -Profile .\configs\game.local.json -Check
```

`--check` prints structured argv with the key redacted and reports key presence
as a boolean. It writes no run/session files. A valid game profile can be
inspected on another OS; its `platform_ready` and exit code will indicate that
actual launching needs Windows. These checks do not verify connectivity or
physical behavior. Use the doctor for read-only package and CUDA checks:

```powershell
python -m forza_ai.launch doctor --role desktop
python -m forza_ai.launch doctor --role game
python -m forza_ai.launch doctor --role desktop --model .\checkpoints\export
```

Doctor never opens the wheel or binds/connects sockets. With `--model`, it loads
exported weights on CPU to validate the artifact. Desktop `training_ready` and
`gpu_training_ready` are reported separately from LAN/key readiness. Current
inference uses CPU; detected CUDA is available to the training pipeline.

After the checks, omit `-Check` to launch. Both scripts use the repo's
`.venv\Scripts\python.exe` when present, falling back to `python` on PATH. They do
not activate a virtual environment, change execution policy, or edit firewalls.
Ctrl+C and console commands are passed through to the runtime. All launches use
argument lists, never shell command strings.

On Ctrl+C, the launcher allows up to 15 seconds for child cleanup and recording
drain before bounded termination/kill escalation. `exit.json` records whether
shutdown was interrupted or forced. Normal graceful shutdown is regression-tested
with a real child process; Windows console behavior still requires device validation.

## Game settings

- `inference.host`: the desktop's numeric LAN IPv4. The desktop's
  `inference.bind` uses its own numeric LAN IPv4; both ports must match.
- `buttons.takeover`: required verified zero-based SDL index. Optional `arm` and
  `route` are also verified indices. All three must differ. `virtual_map` is a
  list such as `[{"physical": 4, "virtual": 1}]`; use only verified mappings.
  Control buttons cannot also be mapped into Forza.
- `mode`: `shadow` is the default and blocks all AI torque/re-engagement;
  `manual` starts without AI torque and allows explicit arm; `assist` requests
  steering at startup once inputs are valid. `-Assist` explicitly overrides a
  profile to assist. Shadow does open the driver for passthrough; driver-native
  forces can change on acquisition, so it is not a passive hardware probe.
- `run.duration_s`: zero runs until stopped. `interactive` enables console
  commands. `dashboard_port` may be `null` to disable the dashboard.
  `dashboard_host` defaults to `0.0.0.0` (all IPv4 interfaces) when enabled.
  Set `127.0.0.1` for local-only access or a specific LAN IPv4 to listen there. Browse to the game PC's actual LAN IP,
  not `0.0.0.0`. LAN binding exposes status and existing control buttons on that
  network; Host/Origin/CSRF checks remain. It does not engage assistance.
- `recording.enabled`: save sessions for later training; false by default.
  `include_manual` is an explicit assertion that ordinary manual driving is
  an expert demonstration. Leave it false for AI evaluation and correction
  collection. Human takeover segments are handled by the recorder separately.
- `control`: conservative starting gains/limits; physical tuning is still
  required. Profiles do not verify calibration or permit bypassing takeover.

Each real launch reserves a fresh directory below `run.output_dir` (default
`../runs`) containing `launch.json` and `exit.json`; the game also writes
`report.json`, plus `control.csv` for finite runs up to 100,000 control ticks.
Enabled recording writes a new `session` inside that run. To store sessions
elsewhere, set `recording.output_dir`; a unique session subdirectory is created
there. Existing runs/sessions are never selected for overwrite.

The fixed desktop example commands a constant five-degree test target. It is
only for stationary physical integration testing and cannot drive a road.
Switch to the model profile once a real exported model is available.
