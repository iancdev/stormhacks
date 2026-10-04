"""
Does the TMX fall behind when the motor command is resent every 10 ms for several seconds?

    python scripts\\wheel_lag_test.py [--torque 0.25] [--takeover-button 11]

Forza and the Thrustmaster panel closed, hands OFF. The wheel is pushed right/left, switching at
irregular 0.1-0.3 s intervals (fixed seed; pushes back toward centre past 60 deg), for 6 s, twice:
  every10ms  stop + update + run every 10 ms (what the live controller does)
  throttled  only when the force changes, plus a renewal every 50 ms
It estimates the delay between command and wheel acceleration for the first, middle and last 2 s.
If 'every10ms' gets slower and slower while 'throttled' stays flat, the wheel is queueing commands.
Safety: force cut past +-150 degrees, on the takeover button, or Ctrl+C.
"""
import argparse
import random
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from forza_ai.hardware import WindowsAdapter  # noqa: E402

LIMIT_DEG, DURATION, RECENTER_DEG = 150.0, 6.0, 60.0


def schedule(seed=7):
    """Irregular switch times so delay estimates can't alias onto a fixed rhythm."""
    rng, t, out = random.Random(seed), 0.0, []
    while t < DURATION:
        t += rng.uniform(0.1, 0.3)
        out.append(t)
    return out


def next_sign(sign, angle):
    """Alternate at each switch, but push back toward centre once the wheel drifts too far."""
    if abs(angle) > RECENTER_DEG:
        return -1 if angle > 0 else 1
    return -sign


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torque", type=float, default=0.25)
    p.add_argument("--takeover-button", type=int, default=11)
    args = p.parse_args()
    if not 0 < args.torque <= 0.3:
        sys.exit("--torque must be in (0, 0.3]")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
    os.makedirs(root, exist_ok=True)
    out = os.path.join(root, time.strftime("wheel-lag-%Y%m%d-%H%M%S.csv"))
    adapter = WindowsAdapter(torque_limit=0.3, use_vjoy=False)
    rows, stopped = [], None
    t0 = time.monotonic()
    try:
        print("Hands off the wheel. Starting in 2 s...")
        time.sleep(2)
        switches = schedule()
        for label in ("every10ms", "throttled"):
            start, last_sent, last_torque = time.monotonic(), -1.0, None
            sign, done = 1, 0
            while time.monotonic() - start < DURATION:
                now = time.monotonic()
                w = adapter.read_state(time.monotonic_ns())
                while done < len(switches) and switches[done] <= now - start:
                    sign, done = next_sign(sign, w.angle_deg), done + 1
                torque = sign * args.torque
                if args.takeover_button in w.buttons:
                    stopped = "takeover button"
                elif abs(w.angle_deg) > LIMIT_DEG:
                    stopped = f"wheel past {LIMIT_DEG:g} deg"
                if stopped:
                    raise StopIteration
                if label == "every10ms" or torque != last_torque or now - last_sent >= 0.05:
                    adapter.set_torque(torque)
                    last_sent, last_torque = now, torque
                rows.append((label, round(now - t0, 4), torque, round(w.angle_deg, 2)))
                time.sleep(max(0.0, 0.01 - (time.monotonic() - now)))
            adapter.set_torque(0.0)
            time.sleep(1.5)   # let it coast to rest between the two methods
    except (StopIteration, KeyboardInterrupt):
        stopped = stopped or "Ctrl+C"
    finally:
        adapter.set_torque(0.0)
        adapter.close()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "t", "torque", "angle_deg"])
        w.writerows(rows)
    print(f"{'STOPPED: ' + stopped if stopped else 'Done'}. {len(rows)} samples -> {out}")
    summarize(rows)


def lag_ms(t, torque, angle):
    """Delay (ms) that best lines up the commanded torque with the wheel's acceleration."""
    import numpy as np
    t, torque, angle = map(np.asarray, (t, torque, angle))
    vel = np.gradient(angle, t)
    acc = np.gradient(np.convolve(vel, np.ones(3) / 3, mode="same"), t)
    best = None
    for k in range(0, 61):                       # 0..600 ms at 10 ms ticks
        a, b = torque[:len(torque) - k], acc[k:]
        if len(a) < 30 or a.std() == 0 or b.std() == 0:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if best is None or c > best[1]:
            best = (k * 10, c)
    return best


def summarize(rows):
    for label in ("every10ms", "throttled"):
        r = [x for x in rows if x[0] == label]
        if not r:
            continue
        t0 = r[0][1]
        parts = []
        for lo, hi in ((0, 2), (2, 4), (4, 6)):
            w = [x for x in r if lo <= x[1] - t0 < hi]
            est = lag_ms([x[1] for x in w], [x[2] for x in w], [x[3] for x in w]) if len(w) > 40 else None
            parts.append(f"{lo}-{hi}s: " + (f"{est[0]} ms (match {est[1]:.2f})" if est else "n/a"))
        print(f"  [{label}] command->wheel delay  " + " | ".join(parts))


if __name__ == "__main__":
    main()
