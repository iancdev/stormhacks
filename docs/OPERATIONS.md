# Operator workflow

The runtime now provides wheel re-engagement, integrated human-correction
recording, a local dashboard, run reports, and saved launch profiles. These can
be exercised with synthetic inputs before a real dataset/model is ready.

## Setup and launch

Follow [TWO_PC_SETUP.md](TWO_PC_SETUP.md) for the machine roles, shared LAN key,
device checks, and initial stationary acceptance. Training coordination stays
in the existing Codex chats; there is no new training orchestration service.

Copy the examples described in [configs/README.md](../configs/README.md) to
`*.local.json`. Those local files are ignored by Git. Fill the actual desktop
IPv4, capture configuration, model export or fixed-test mode, and verified
wheel button indices. Profile paths are relative to the profile directory.

```powershell
python -m forza_ai.launch doctor --role desktop
.\scripts\start-desktop.ps1 -Profile .\configs\desktop-test.local.json -Check
.\scripts\start-game.ps1 -Profile .\configs\game.local.json -Check
```

Omit `-Check` to launch. The game example starts in shadow mode, so AI torque
and arm requests are disabled. Use profile mode `manual` to allow explicit
engagement, or explicitly choose `assist` after hardware acceptance. Each launch
creates a unique run directory with a redacted configuration manifest and report.
Passwords and the LAN key belong only in the process environment, not profiles.

## Wheel and operator controls

- **Takeover button:** removes AI torque immediately. A fresh explicit takeover
  declares a human correction; merely losing a model/network input does not.
- **Arm button:** a rising edge requests engagement. Holding it cannot re-arm
  after a fault, and a button already held at startup is ignored. Takeover wins
  if arm and takeover are pressed together.
- **Route button:** a rising edge starts a route attempt; the next press marks
  completion. These are operator annotations, not automatically inferred laps.
- **Console/dashboard:** `arm`, `manual`, `route_start`, `route_complete`, and
  `route_abort` request the same actions. `quit` or Ctrl+C ends the runtime.

Configure `--takeover-button`, `--arm-button`, and `--route-button` with distinct
zero-based SDL indices. Reserved control buttons must not also be mapped to game
actions. Re-engagement waits up to five seconds for a valid command. An arm
request while already assisted does not survive a later fault.

## Live dashboard

`--dashboard-port 8766` serves a dashboard on localhost by default at
`http://127.0.0.1:8766`. It shows wheel/target/policy angles, source observation
age, inference duration, control state, intervention count, route markers, and
recording/transport status. Stale snapshots visibly disable controls. Requests
are queued for the runtime; HTTP threads never write to the motor.

The redesigned view includes timestamp-scaled steering, source-age, and torque
traces with toggles/tooltips; observed capture/prediction/control rates; latency
percentiles; control limits; and readable recording health. Rates are measured
over a rolling two-second window and remain unknown during warm-up or when a
source is absent. Remote prediction time includes the network round trip.
Simulation, shadow, fault, stale, and disconnected states are explicitly distinct.
The sticky header keeps Disengage AI visible while scrolling. A queued request
is shown as confirmed only after a subsequent fresh runtime snapshot supports it.
An independent browser timer marks snapshots stale after 500 ms, even while an
HTTP request is stalled. Status and control requests have a one-second deadline,
including response decoding; polling resumes after failures. A timed-out control
request is reported as delivery unconfirmed because the runtime may have queued it.

Keep the game foreground during assisted driving. Inspect the dashboard on a
second screen without taking focus, or use the physical buttons. Clicking the
browser causes the foreground guard to disengage AI; an arm request gives you
time to return to the game. When enabled, the client dashboard listens on all IPv4 interfaces by default.
Use `--dashboard-host 127.0.0.1` to restrict it to the client PC.

### Access from another computer

Add `--dashboard-port 8766` to the existing runtime
command **on the PC running Forza and the wheel client**. This listens on all
IPv4 interfaces. Alternatively bind only that PC's specific LAN IPv4 address.
Open `http://<RACING-PC-LAN-IP>:8766` from the other computer; `0.0.0.0` is a
listen address, not a browser destination, and the other computer's localhost
would point at itself. This does not create a dashboard on the inference laptop.

Saved game profiles support `run.dashboard_host` (default `0.0.0.0`) alongside
`run.dashboard_port`. The existing launcher passes both to the runtime. Adding
the bind option does not change `manual`, `shadow`, or `assist` engagement mode.

LAN binding makes dashboard status and its existing controls reachable by other
computers on that network. There is no new login system. Exact Host/Origin and
CSRF-token checks remain: for wildcard binding, accepted Host/Origin must match
the actual local destination IP and port of the connection, never an arbitrary
header value. DNS aliases and wildcard Host/Origin values are not accepted.
No broad CORS or firewall changes are made by this option.

Local simulation preview (no wheel or dataset):

```sh
python -m forza_ai.runtime --backend sim --assist --sweep --duration 30 --dashboard-port 8766 --run-report /tmp/demo-report.json
```

This screenshot is from simulated hardware, not a physical driving result:

![Simulated operator dashboard](assets/dashboard-preview.jpg)

[Narrow-screen preview](assets/dashboard-mobile-preview.jpg). The screenshots
use synthetic signals, including a deliberately visible latched fault state;
they are not evidence of physical driving or hardware acceptance. The review
findings, implemented fixes and verification limits are in [HARNESS_REVIEW.md](HARNESS_REVIEW.md).

## Integrated recordings and human corrections

Enable `--record-session NEW_DIRECTORY` (or profile `recording.enabled`) to
record frames, wheel samples, telemetry, and control provenance in the same
process as assistance. The recorder is asynchronous and bounded. It does not
open a second wheel or telemetry listener.

`--record-manual` explicitly labels initial manual driving as expert examples.
Use it for human demonstrations, not to relabel AI-generated motion. For a
correction run, leave it off: only an explicit human takeover becomes expert
data. The first 100 ms after zero AI torque is written are excluded by default
(`--takeover-settle-ms`). Eligibility uses the wheel sample timestamp, and cached
frames from before the human-control boundary are excluded.

Automatic timeout/fault disengagements remain nonexpert. Assistance samples may
be retained for diagnostics but the training loader rejects them. A recording
with no human corrections can therefore be complete but have zero trainable
samples. Validate before adding it to a dataset.

The output is stream-v1 (`metadata.json`, `frames.csv`, `wheel.csv`,
`telemetry.csv`, PNG images, and an events sidecar). It loads directly; do not
run the legacy importer on these sessions. Queue drops, writer failures, and
incomplete shutdown keep `completed: false`. Ctrl+C is a normal recording stop
when the writer drains successfully, and hardware cleanup precedes that drain.

Manual collection without a trained model:

```powershell
python -m forza_ai.runtime --backend windows --shadow --capture-config config/capture.json --takeover-button BUTTON_INDEX --record-session data/sessions/manual-001 --record-manual --duration 0 --dashboard-port 8766 --run-report runs/manual-001.json
```

Capture defaults to 30 fresh frames/sec; inference remains independently paced.
Actual achieved rates depend on the game, drivers, and storage. Sessions should
span separate recording runs for independent train/validation groups.

## Run reports and comparisons

`--run-report PATH` stores bounded aggregates even for unlimited runtime
duration: explicit human interventions per assisted minute, mode durations,
assisted tracking RMSE, source-age/inference/control-gap percentiles, and route
markers. Percentiles use a fixed histogram and are labelled approximate.
Inference samples are deduplicated by prediction ID; observation age is sampled
per control tick. Reports do not turn manually marked completion into proof of
autonomous route success.

```powershell
python -m forza_ai.metrics report runs/run-001/report.json runs/run-002/report.json
python -m forza_ai.metrics report runs/run-001/report.json --json
```

`--status-csv` remains an optional detailed finite-run log (at most 100,000 ticks).
Saved profiles select it only when the run is suitably bounded. Cleanup faults
and recorder completion status are included in the report.

## Legacy recorder: fresh 30 FPS default

`python record.py record` defaults to `--fps 30` per the latest user preference;
explicit `--fps` overrides (including 60) remain supported. It uses paced one-shot
`new_frame_only` capture and skips `None` results rather than reusing a cached
image. `meta.json` reports measured fresh/saved FPS, elapsed time, gaps, inactive
frames, and queue drops. Requested FPS is not a guarantee of achieved FPS.

Legacy `labels.csv` columns and rounded post-retrieval timestamps remain
compatible. New `capture_timing.csv` retains integer host capture-start,
retrieval, wheel-poll, and telemetry-receive timestamps. These are not GPU render
or USB hardware timestamps. Keep the full original recording directory: the
legacy importer continues to use its conservative aligned labels, not the new
timing sidecar. The integrated recorder above supplies the stream-v1 route for
new correction datasets.

### Latest-frame inference scheduling

Camera policies infer once per successfully processed `(frame_id, capture_timestamp)`.
The policy worker waits for new frames, caps starts at `--policy-hz`, and retains only
one latest pending observation while a prediction runs. Slower inference therefore
reduces the actual inference cadence without accumulating old frames. Capture and
the 100 Hz local control loop remain independent. Recoverable failures retry at the
configured cap; successful predictions never refresh an old frame's timestamp or
command deadline. Invalidation still discards in-flight results.

Run reports and dashboard state include `policy_worker`: unique input frame rate,
model-call rate (including retries), duplicate publication and superseded-frame
counts. These rates cover the worker lifetime, whereas `rates` remains a rolling
window of capture, successful prediction, and control events. This distinguishes
new visual information from repeated model calls. No stale-input budget is extended.

The inference server has one fixed model worker and one transport worker. PNG
decode/model execution cannot trap the accept loop after a client disconnects.
While a model is still busy, another authenticated request is rejected rather than
queued or run concurrently. Python model execution cannot be forcibly cancelled;
a hung worker requires an operator restart, and shutdown reports it after closing
sockets. Client deadlines and re-arm requirements still apply.

`inference_server --cpu-threads 4` is the default for new model-server starts;
override it after measuring on the deployment machine. This sets PyTorch intra-op
threads in that server process only. The existing running server is not retuned.
A short batch-one FP32 trial on the RTX 5080 laptop (64 held-out crops, 8 warm-ups,
v2 epoch-2 weights) measured CPU threads 1/2/4/default-24 medians of
1.502/1.183/0.968/1.041 ms including preprocessing, versus CUDA 1.143 ms including
H2D/model/D2H. Respective p95 values were 1.834/1.395/1.390/1.500/1.622 ms.
Therefore CPU remains the inference device. This is a short model-stage benchmark,
not measured game capture or Ethernet latency. A separate 32-request authenticated
loopback trial using CPU1 measured 2.980 ms median / 3.483 ms p95 including PNG and
protocol work. Preprocessing, model weights, FP32 precision and control limits are
unchanged. Raw Windows evidence is `runs/batch1-latency-20261003` in the isolated
`stormhacks-driving-b678c33` checkout (local evidence commit `8dc6457`).

### Prediction visual

The dashboard includes a **forward-view schematic**, rendered in the browser from
existing bounded status snapshots (runtime publication capped at 10 Hz). It is
not a camera stream or overlay. Steering sign bends the relative curve left or
right (right positive); its length has no distance or time scale. There is no
calibrated camera projection, road-wheel ratio, or wheelbase, so it must not be
used as a planned route or track-boundary estimate. Speed remains a separate
telemetry readout; no heading or speed-based physical trajectory is invented.

Model steering/throttle/brake are separate from measured physical wheel/pedals
and virtual pedal output. Missing steering-only model pedals show dashes rather
than fabricated zeroes. The curve and prediction values disappear when command
lifetime or source-age limits expire, input is unavailable, or the dashboard is
stale/offline. Browser elapsed time advances these checks between responses. Polling runs at
most 10 Hz with one request in flight; full HTTP round-trip time is conservatively
added to snapshot age. No command lifetime is extended to keep the curve visible.
There is no extra capture, image encoding, hardware owner, or control-loop I/O.
# Inference connection diagnostics

The inference server CLI writes best-effort JSON lines alongside its startup
message. Events identify a process-local connection number and peer, UTC time,
first authenticated/protocol-valid request, first prediction sent, five-second
aggregate summaries while requests complete, and connection closure. A sent
response is not proof the game applied it. Rates are connection-lifetime means.

Closure records distinguish authentication/protocol rejection, idle receive
timeout (no new bytes), partial-request timeout, peer EOF/reset, busy model,
decode failure, model failure and server shutdown. `phase` identifies the
operation at failure; raw exception arguments, keys, HMACs, images, session
nonces and predictions are never logged. A client that rejects the server's
hello may simply appear as EOF; the server cannot know that client's reason.

`completed_model_calls` and model mean/max cover observed completed calls;
`model_elapsed_ms` also reports elapsed work at a model timeout/disconnect,
not the eventual duration of a still-running call. Request timing includes
waiting for bytes. Existing protocol, model outputs and timeouts are unchanged.

Formatting and output run on a daemon with a bounded 256-event queue. Overflow
or output failure drops diagnostics (`log_dropped`) rather than blocking model
or transport work. Shutdown waits at most 250 ms for logging; a blocked sink
can lose final records. These logs cannot reconstruct failures before deployment.
