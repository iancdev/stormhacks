# Shared contracts, version 1

## Units and time

- Physical steering angle: degrees, right positive, nominal TMX range -450 to +450 at the reported 900-degree calibration.
- Speed: metres per second. Pedals: 0 released, 1 fully pressed.
- Local timestamps: integer monotonic nanoseconds from the recording/runtime process clock. Timestamp streams separately. Never subtract monotonic times from separate computers.
- State includes timestamps and freshness; a repeated sample does not become fresh merely because it was read again.

## Session format

Each completed session is a directory with `metadata.json`, `frames.csv`, `wheel.csv`, `telemetry.csv`, and an `images/` directory. Sessions can be transferred as archives. Do not train from directories still being recorded or copied. The first version uses per-frame images; a video container can be supported later behind the same frame index and timing contract.

Required CSV columns:

| File | Columns |
| --- | --- |
| frames.csv | `frame_id,image_path,capture_time_ns` |
| wheel.csv | `timestamp_ns,angle_deg,throttle,brake,control_mode` |
| telemetry.csv | `timestamp_ns,speed_mps,is_race_on` |

Image paths are relative to the session and must stay within it. Encode images as ordinary RGB-compatible PNG/JPEG; loaders decode to RGB. These images are the chosen road crop, before model resizing. Preserve source frames where practical. `control_mode` is `manual`, `assist`, or `takeover`; only manual and takeover samples are expert labels. `is_race_on` is 0 or 1. Optional telemetry columns may include the game's timestamp, RPM, steering input, and yaw rate, with units documented.

Required metadata fields: `schema_version` (1), `session_id` (unique string), `clock` (`monotonic_ns`), `wheel_rotation_deg` (900 for the current calibration), and `image_stage` (`road_crop`). Optional metadata should record camera view, crop rectangle, source dimensions, car, route, weather, racing-line setting, recording tool/version, and completion status. Validation and all splits operate on whole sessions. A trainer can require additional metadata when needed and must document it.

Host receive time is not simulation time. Associate frames with the latest sufficiently fresh telemetry sample at or before the frame timestamp. A steering label can be sampled/interpolated at `capture_time_ns + label_offset_ns`, but only within a continuous expert-controlled segment with bounded sample gaps. Never interpolate across takeover boundaries, pauses, rewinds, missing-data gaps, or sessions. Offset and age/gap limits are training configuration, not an assumed human reaction time. Future labels are a supervised training choice; live model inputs must be causal.

## Initial model and export

Input: RGB road crop resized to width 200, height 66, plus speed in m/s. The training owner decides and exports exact pixel/speed normalization, interpolation method, architecture version, and steering output scaling. Public inference output is degrees, right positive. Initial telemetry input is speed only; physical and in-game steering are logged, not input as their own same-time label.

An exported artifact must contain model weights and enough metadata to reconstruct preprocessing and inference. The model owner exposes a portable predictor accepting a road-crop image plus speed and returning a physical angle; the adapter provides an integration wrapper if necessary. No Windows dependencies belong in this predictor. The originating chat owns `src/forza_ai/contracts.py`; training should not depend on unpublished Python dataclass definitions when it can use this stable file/unit contract.

## Runtime data boundary

`WheelState`: timestamp, physical angle, normalized pedals, buttons, connection status.

`VehicleState`: host receive timestamp, game timestamp where present, speed, race state, diagnostic telemetry.

`SteeringCommand`: target angle, generated timestamp, source observation timestamp, expiry timestamp. Reject non-finite angles, expired commands, future host timestamps, and excessively old source observations. Once assistance disengages from takeover or expiry, fresh commands alone cannot re-engage it.

The wheel loop forwards measured physical inputs in every active control mode. It never forwards the policy target directly as the game's steering input. Only the local controller writes motor torque. A logical positive torque means physical right; the TMX adapter translates that to SDL's measured negative force level. vJoy X steering is low-left/high-right; Y brake and Z throttle are inverted (32768 released, 1 pressed).

Assistance is explicit and continuous once engaged. A test placeholder emits deterministic targets only and is clearly identified as not a driving model. `manual`/`takeover` suppress AI torque. Effects should expire without host refresh, providing a bound when the process stalls; cleanup alone cannot handle a hard process failure.
