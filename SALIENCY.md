# Model input and activation diagnostic

Optional dashboard diagnostic for the CPU PilotNet inference server and remote
runtime. It is disabled by default. Implementation and synthetic validation used
no production model weights or hardware; deployment must be coordinated separately.

## What the panel shows

The left image is the actual **200×66 normalized tensor fed into the model**,
converted back to RGB for display. It is captured by a model pre-hook, after any
predictor wrapper's contrast/saturation adjustments and the model's own
crop/resize/normalization. The right image is the same input with an amber
activation overlay. A single lossless 400×66 PNG keeps both images aligned.
The caption identifies the original frame/request, its age, and that frame's
prediction. This is separate from the existing faster road-crop HUD.

The implementation follows [VisualBackProp, section 3](https://arxiv.org/html/1611.05418#S3):
channel averages at each of the five convolutional activation layers, all-one
transposed convolutions using the original kernel/stride, multiplication with
shallower maps, and final within-frame normalization. NumPy scatter additions
implement transposed convolution, including correct asymmetric output padding;
a Torch reference test verifies the numeric result. Positive scalar rescaling
between stages prevents numeric overflow without changing the normalized mask.

**Adaptation:** this repository uses ELU, whereas the paper uses ReLU. We retain
only positive post-ELU activations before channel averaging. Negative evidence is
omitted. The model itself is never altered. This is not a faithful ReLU-network
reproduction, a steering-specific gradient attribution, a road segmentation, or
proof of what caused an output. It visualizes shared encoder activations; speed
fusion and output heads are not explained. Zero signal produces no colored
activation, not an invented heatmap. See also the [NVIDIA explanation](https://developer.nvidia.com/blog/explaining-deep-learning-self-driving-car/).

## Bounded operation and compatibility

- Explicit `--saliency` on the inference server and remote runtime. Defaults off.
  CPU PilotNet only. Unsupported architecture/device fails configuration.
- No hooks or rendering worker exist on a default server. Old `predict` requests
  retain the original v1/v2 schema, empty response payload and numeric outputs.
- Opted-in client uses `predict_preview` / `prediction_preview` variants with the
  existing HMAC/session/nonce/request-ID envelope. No new listener or credentials.
  These variants require the updated server. An old server rejects the new kind;
  do **not** enable the client flag until server compatibility is established.
  An updated server with diagnostics off returns normal predictions plus an
  empty preview; dashboard says waiting. A non-opted client gets no preview.
- Capture defaults to 10 Hz from existing forward passes. `--saliency-hz 0` removes
  the time cap, sampling each new source frame at most once per session. A positive
  value selects a bounded rate (for example `--saliency-hz 5`). No
  extra model forward/backward, no GPU transfer. Only the input and five reduced
  maps are copied (173,936 bytes for this architecture).
- A daemon renders independently. One in-progress render and one replaceable
  pending snapshot; no accumulating queue. The model worker never waits for a
  render. Snapshot copying does add bounded synchronous CPU work. A completed
  preview adds transport bytes to a prediction response at most once per sample;
  this is **not a claim of zero latency impact**. No deadline is extended.
- PNG capped at 100,000 bytes with fixed 400×66 dimensions. Original frame IDs
  are correlated against a bounded 256-entry local request history. Sessions
  cannot reuse each other's previews. No remote monotonic-clock subtraction.
- Rendering older than 500 ms is omitted; the client checks original local
  capture age and elapsed receipt age. Dashboard hides expired/unavailable images,
  including a browser expiry timer. Disconnect clears client preview state.
- Dashboard polls only the latest read-only `/api/saliency` endpoint at up to 20 Hz,
  with one fetch/decode in flight and no duplicate-frame decoding. HTTP and decode
  time count toward the 500 ms capture-age expiry; stale responses never replace
  a fresh image. This presentation ceiling is separate from uncapped server capture.
  Existing Host checks, CSP, Origin/token control guards and command routes are
  unchanged. No screenshot/camera owner, model control or actuation is added.

## Validation

New tests cover the five-map numeric reconstruction, zero/nonfinite maps,
bit-identical predictions, exact model tensor visualization, a transformed-input
wrapper, blocked renderer/latest replacement, rate limiting, session isolation,
legacy and opted-in authenticated transport, payload limits, stale/disconnect
handling, server restart cleanup and dashboard Host/method guards.

Initial focused run: **124 passed + 26 subtests**, with two failures also
reproduced on an untouched f68faca archive:

- `test_dashboard_frame.py::test_frame_endpoint_serves_latest_jpeg_and_rate_limits`
  assumes its intervening HTTP request completes within 80 ms.
- `test_dashboard_visual.py::test_prediction_curve_sign_and_controls_are_distinct`
  expects an older curve endpoint (`440 25` versus current `665 40`).

The feature-focused rerun passed **126 tests + 26 subtests**, with these two
known baseline failures explicitly deselected, not silently fixed or counted as
passes. Local validation reports are not checked into the repository. No native Windows/PowerShell, trained-model semantic quality or real
LAN performance claim is made.

Synthetic Mac arm64/Torch 2.13 CPU1 microbenchmark (64 forwards per mode; fixed
order, random seed 7, untrained model, synthetic image): disabled median 1.333 ms;
enabled unsampled 1.271 ms; sampled forward 1.480 ms. Independent render median
1.034 ms, p95 1.474 ms, cold maximum 14.563 ms. Preview PNG 17,514 bytes. All 128
enabled predictions exactly equal the disabled prediction. Sampling was forced
for forward-overhead measurement; these are not sustained real-LAN measurements.
The fixture script generates full values in `artifacts/saliency/benchmark.json`.

Browser QA uses a synthetic fixture, seeded **untrained** model and intercepted
requests only. The QA scripts generate desktop/mobile ready/stale/disabled
screenshots and a report in the ignored `artifacts/saliency/` directory. Production CSP was enforced; no JavaScript errors or
mobile horizontal overflow. The image/heatmap really comes from the synthetic
model's activations, but demonstrates no learned road understanding.

Reproduce (local CPU, no real export required):

```sh
PYTHONPATH=src python3 -m pytest tests/test_visualbackprop.py -q
PYTHONPATH=src python3 scripts/saliency_fixture.py
# With Playwright available and a local Chrome binary:
node scripts/saliency_browser_qa.cjs
# Optional PLAYWRIGHT_MODULE / CHROME_PATH point to existing installations.
```

## Integration on latest main

The feature was integrated on `a195373` with no conflicts, preserving the newer
force-feedback options, takeover-trigger logging, and Devpost documents.
The broader non-analytics run passed 643 tests and 197 subtests, with the two
known dashboard baseline failures deselected. One real-time FFB smoothing test
failed its ripple bound during that run; it passed on an isolated run against
unchanged `a195373`. The FFB production algorithm is untouched. Its test now uses a deterministic
100 Hz clock, retaining its original ripple/mean assertions; all 32 targeted
FFB, saliency, and runtime checks passed after this test-only repair.
Full analytics collection requires the optional `psycopg` dependency, absent on
the validation Mac; analytics tests were not run. These limitations mean this
was not an entirely green full-suite run.

## Enable on a compatible server and client

1. Update the actual server and racing-client source to a compatible revision.
   **Preserve Windows-local idle-timeout and input-adjustment patches**; this Mac
   base does not include those additions. Do not replace that server with an
   unmerged Mac checkout. Model artifacts and preprocessing stay unchanged.
2. Run focused tests on that merged revision. Check batch-one latency with the
   actual model/device before a driving session; previews add CPU/transport work.
3. At a separately coordinated restart, append `--saliency --saliency-hz 0` (uncapped) or `--saliency` (10 Hz default) to the existing
   server command, preserving its model path, same-key handling, CPU threads,
   bind/port, `--timeout 2`, `--idle-timeout 15` and contrast/saturation settings.
   No command here authorizes a restart or supplies a key.
4. On the racing PC, append `--saliency --saliency-hz 0` (uncapped) or `--saliency` (10 Hz default) to the existing remote-runtime command,
   or use the new optional `-Saliency` switch in `scripts/run-ai.ps1`. Preserve
   all existing capture, calibration, takeover and pedal settings. No flags that
   engage assistance are added by this feature. Profile launchers that don't
   pass this option need their normal runtime invocation updated explicitly.
5. Open the existing dashboard. Verify original-frame input/overlay pairing and
   expiry in shadow mode. Turning the option off on both next starts restores
   the ordinary transport and removes activation hooks entirely.

This feature does not automatically activate or engage a driving policy.
Optional output-specific occlusion validation remains future work;
this implementation intentionally makes no causal explanation claim.


## Freshness fix (October 4)

The former server capture gate was 1 Hz and the browser waited 500 ms after each
request. Those deliberate limits caused the visibly stepped preview. The new
configurable capture gate and 50 ms browser cadence remove those bottlenecks.
Input, activation overlay and prediction always come from the same inference;
there is no second camera capture or heatmap pasted onto a newer frame.
The renderer still has one active render and one replaceable pending snapshot.
Expired snapshots are discarded both before rendering and before publication.
Uncapped means new inference frames as available, not a busy loop or a FIFO.

The transport remains the same authenticated optional payload on a prediction
response, capped at 100 KB. Rendering never blocks model inference, but transferring
PNG bytes still consumes response time; this is not a separate priority channel.
No prediction deadline is extended. Old non-preview clients retain their schema.
Both server and racing-client source must be updated for the full freshness fix.

Windows owner measurements on the actual original weight25 epoch-2 model, CPU4,
three recorded frames, paced 30 Hz for four seconds per mode (fixed order, no
repeats): disabled inference p50/p95 1.580/1.931 ms; 10 Hz 1.775/2.221 ms;
uncapped 2.078/3.080 ms. Uncapped produced 119 previews for 120 frames, with one
pending at run end, versus 40/40 at 10 Hz. Prediction values were exactly unchanged.
Preview PNG averaged 57,679 bytes (53,687–61,175), about 4.33 Mbps at 10 Hz and
13.83 Mbps uncapped. Independent encode/render p50/p95 was 1.566/1.844 ms.
These pre-fix rate simulations establish cost, not actual LAN/browser latency.

Focused Mac validation: 120 tests and 26 subtests passed; the two documented
baseline dashboard failures were deselected. Chrome fixture QA displayed 20
updates in one second, max one in-flight request, no console errors or mobile
overflow, with repeated-frame dedup and delayed-response expiry checks. These
are synthetic browser tests, not measured racing-PC performance.
