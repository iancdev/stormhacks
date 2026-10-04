"""
Open-loop motor step test: how fast and how strongly does the TMX respond to a constant force?

    python scripts\\wheel_step_test.py [--torque 0.15] [--takeover-button 11]

Forza and the Thrustmaster panel must be closed. Keep hands OFF the wheel; it will be pushed
right, released, pushed left, released, at --torque, three times, each driving the motor a
different way (restart every tick / update a running effect / one long effect per push).
Safety: force is cut and the test stops
if the wheel passes +-90 degrees, if the takeover button is pressed, or on Ctrl+C.
Writes runs/wheel-step-<time>.csv and prints the delay and response for each push.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from forza_ai.hardware import WindowsAdapter  # noqa: E402

LIMIT_DEG = 90.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torque", type=float, default=0.15)
    p.add_argument("--takeover-button", type=int, default=11)
    args = p.parse_args()
    if not 0 < args.torque <= 0.3:
        sys.exit("--torque must be in (0, 0.3]")

    # (label, method, [(duration s, torque)...]); positive torque = push right (runtime convention)
    #   restart: adapter.set_torque every 10 ms = stop + update + run (what the runtime does now)
    #   update:  start once, then only update the level every 10 ms; re-run without stopping every 100 ms
    #   long:    one stop + run per push with the push's whole duration (like utils/test.py ffb)
    push = [(0.4, +args.torque), (1.6, 0.0), (0.4, -args.torque), (1.6, 0.0)]
    phases = [("restart", "restart", [(1.0, 0.0)] + push), ("update", "update", push), ("long", "long", push)]

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
    os.makedirs(root, exist_ok=True)
    out = os.path.join(root, time.strftime("wheel-step-%Y%m%d-%H%M%S.csv"))
    adapter = WindowsAdapter(torque_limit=0.3, use_vjoy=False)
    rows, stopped = [], None
    t0 = time.monotonic()
    try:
        print("Hands off the wheel. Starting in 2 s...")
        time.sleep(2)
        sdl, haptic, eid, effect = adapter._sdl, adapter._haptic, adapter._effect_id, adapter._effect
        for label, method, steps in phases:
            for duration, torque in steps:
                end, last_run, first = time.monotonic() + duration, -1.0, True
                level = -int(torque * 32767)          # same sign convention as conversions.sdl_force_level
                while time.monotonic() < end:
                    now = time.monotonic()
                    w = adapter.read_state(time.monotonic_ns())
                    if args.takeover_button in w.buttons:
                        stopped = "takeover button"
                    elif abs(w.angle_deg) > LIMIT_DEG:
                        stopped = f"wheel past {LIMIT_DEG:g} deg"
                    if stopped:
                        raise StopIteration
                    if torque == 0:
                        if first:
                            sdl.SDL_HapticStopEffect(haptic, eid)
                    elif method == "restart":
                        adapter.set_torque(torque)
                    elif method == "update":
                        effect.constant.level, effect.constant.length = level, 200
                        sdl.SDL_HapticUpdateEffect(haptic, eid, effect)
                        if now - last_run >= 0.1:
                            sdl.SDL_HapticRunEffect(haptic, eid, 1)
                            last_run = now
                    elif method == "long" and first:
                        sdl.SDL_HapticStopEffect(haptic, eid)
                        effect.constant.level, effect.constant.length = level, int(duration * 1000)
                        sdl.SDL_HapticUpdateEffect(haptic, eid, effect)
                        sdl.SDL_HapticRunEffect(haptic, eid, 1)
                    first = False
                    rows.append((label, round(now - t0, 4), torque, round(w.angle_deg, 2)))
                    time.sleep(max(0.0, 0.01 - (time.monotonic() - now)))
    except (StopIteration, KeyboardInterrupt):
        stopped = stopped or "Ctrl+C"
    finally:
        adapter.set_torque(0.0)
        adapter.close()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "t", "torque", "angle_deg"])
        w.writerows(rows)
    print(f"{'STOPPED: ' + stopped if stopped else 'Done'}. {len(rows)} samples -> {out}")
    summarize(rows)


def summarize(rows):
    """For each push: delay until the wheel moved 1 deg, angle reached, peak speed, drift after release."""
    i = 0
    while i < len(rows):
        label, t, torque, angle = rows[i]
        if torque != 0 and (i == 0 or rows[i - 1][2] != torque):
            start_angle, start_t = angle, t
            j, moved_t = i, None
            while j < len(rows) and rows[j][2] == torque and rows[j][0] == label:
                if moved_t is None and abs(rows[j][3] - start_angle) > 1.0:
                    moved_t = rows[j][1]
                j += 1
            end_angle = rows[j - 1][3]
            k = j
            while k < len(rows) and rows[k][2] == 0 and rows[k][1] - rows[j - 1][1] < 1.6:
                k += 1
            settle = rows[k - 1][3] if k > j else end_angle
            speeds = [abs(rows[m + 1][3] - rows[m][3]) / max(rows[m + 1][1] - rows[m][1], 1e-3)
                      for m in range(i, min(k, len(rows)) - 1)]
            delay = f"{1000 * (moved_t - start_t):.0f} ms" if moved_t else "never moved"
            print(f"  [{label}] torque {torque:+.2f}: moved after {delay}, {start_angle:+.1f} -> {end_angle:+.1f} deg "
                  f"during push, coasted to {settle:+.1f}, peak {max(speeds, default=0):.0f} deg/s")
            i = j
        else:
            i += 1


if __name__ == "__main__":
    main()
