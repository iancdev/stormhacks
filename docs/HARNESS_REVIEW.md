# Harness review — 2026-10-03

Reviewed baseline: `7a9f425`. The four defects below were reproduced and then
fixed after explicit user authorization. The descriptions preserve the original
failure evidence. The original suite passed 365 tests despite these gaps.
Reproductions and fix validation used mock hardware, temporary files, simulated
control, and loopback networking; no physical motors were operated.

## Fix status

| Finding | Implemented correction | Evidence |
| --- | --- | --- |
| Native actuation expiry | Absolute deadline passed to the adapter; checks after preparation, bounded effect duration, late-return stop and latched fault | `53d76e2`, runtime integration `406fbe9`; delayed native-call and all-deadline-limit regressions |
| Transient inference failure | Persistent failure generation consumed before control and again before output; fresh retries cannot erase it or re-arm | `3557c73`, integration `406fbe9`; deterministic races and real loopback failure/reconnect |
| Recorder detach | Attachment checks around polling, invalidation of readiness/cached samples, error propagation and neutralization | `bef8f09`; detach-after-warmup and detach-during-capture regressions |
| Launcher interrupt | Explicit child lifecycle, 15-second cleanup grace, bounded escalation, durable exit outcome | `406fbe9`; real 700 ms child cleanup and repeated-interrupt regressions |

Final integrated verification: **445 tests and 192 subtests passed** after the
fixes and the separately owned bounded data-compatibility update. Follow-up
reviews found no blocking gap in deadline propagation or failure-generation
consumption. Desktop/narrow dashboard controls were tested with synthetic data.

## P1: expired commands can start new native motor effects

Location: `src/forza_ai/hardware/windows.py`, `WindowsAdapter.set_torque`, near
`SDL_HapticUpdateEffect` / `SDL_HapticRunEffect`.

The runtime checks wheel/command ages before entering the adapter. The adapter
then performs native attachment, stop, and update calls before running the force
effect, but it only receives a torque value and cannot revalidate deadlines.

Using the actual runtime and adapter with the existing mocked SDL Rig, delaying
`SDL_HapticUpdateEffect` by 300 ms still resulted in a new native force run: the
wheel sample was 300 ms old and the command expired 150 ms earlier. This is a
host-delay bug, separate from the documented lack of USB hardware timestamps.

Pass an absolute actuation deadline into the adapter, recheck after native
preparation and immediately before starting an effect, and constrain effect
duration to remaining validity. Add a delayed-native-call regression. Physical
driver behavior still needs hardware validation even after software correction.

## P1: a fast reconnect can erase an unobserved inference failure

Location: `src/forza_ai/runtime.py`, `PolicyWorker._run`, the
`ObservationUnavailable` handler and successful command publication.

A failed request sets the current command to `None`, but success immediately
overwrites it. If the failed request exceeded the policy period, retry starts
without a delay. The 100 Hz control loop can miss the failure entirely and remain
assisted, contrary to the documented explicit-rearm rule after disconnects.

With a real loopback server/client and simulated wheel, the third server
prediction waited 40 ms then failed; later predictions succeeded. Five trials
ended in ASSIST, with zero takeover time and only the initial arm event.

Keep a failure generation/event pending until the control loop consumes it and
disengages. A successful retry must not erase it. Cover failure followed by an
immediate reconnect, in addition to permanent-server-loss tests.

## P2: the standalone recorder does not detect wheel detachment

Location: `record.py`, `WheelReader._run` polling loop.

The recorder never checks `SDL_JoystickGetAttached`, refreshes sample timestamps
after every poll, and retains its initial `live=True` state. Cached/invalid axis
values after unplugging the wheel can therefore be saved as fresh expert labels
and optionally forwarded to vJoy.

Mocked disconnection after the second poll still yielded `live=True`, no error,
zero attachment checks, and a newly timestamped sample on the fifth poll.

Check attachment around polling and invalidate/stop recording on detach. The
separate integrated `WindowsAdapter` already performs attachment checks.

## P2: launcher interrupt can kill a child before cleanup finishes

Location: `src/forza_ai/launch.py`, `launch`, the `subprocess.run` call.

Ctrl+C reaches both launcher and child. Under the tested Python 3.10 runtime,
`subprocess.run` interrupts its wait and kills the child before a longer graceful
shutdown can finish. Recording drains and diagnostic output can be lost, and
the launcher's `exit.json` is not written.

A child needing 700 ms for normal interrupt cleanup completed when started
directly (about 720 ms), but was terminated through the launcher after about
287 ms; its completion marker remained missing after one second.

Manage child lifecycle explicitly: wait for a bounded graceful shutdown on
interrupt, escalate only after that deadline, and persist the exit manifest in
the cleanup path. Verify actual Windows console-signal behavior as well.

## Scope and next action

No additional confirmed defect was found in expert-mode filtering, causal
telemetry lookup, dataset grouping, incomplete-writer closure, PNG integrity,
or response correlation during this review. Passing tests do not prove these
paths or native drivers are defect-free.

All four confirmed code findings are addressed. Native SDL calls remain
synchronous; no Python wrapper can guarantee hard real-time behavior inside a
blocked driver call. Actual driver expiry behavior, physical tracking, Windows
console shutdown and real two-PC operation still need on-device acceptance.
Dataset/model performance and production data-schema compatibility remain
separate from the harness fixes.
