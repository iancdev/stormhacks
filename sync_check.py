"""
Check that a recording's frames and labels line up in time.

    python sync_check.py                    # newest session in data/recordings
    python sync_check.py data/recordings/20261003_150225

Measures, by cross-correlation (positive lag = the second signal happens later):
  wheel steer -> telemetry steer   input path: how long Forza takes to see the wheel
  yaw rate    -> image motion      capture path: how late the screen shows what telemetry reports
                                   (needs the yaw_rate column; the clean sync test)
  wheel steer -> image motion      everything: input + car response + capture, for the label shift
  telemetry gear -> HUD gear       capture path from sharp events: when telemetry reports a shift vs
                                   when the gear digit (hud/ patches) changes on screen. Most direct.
Image motion = sideways shift of the distant scenery (top of the frame) between frames.
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

SKIP_END_S = 5.0   # recordings made before Ctrl+C discarding may end in a crash


def load(session):
    rows = list(csv.DictReader(open(os.path.join(session, "labels.csv"))))
    if rows and rows[-1]["t"]:
        end = float(rows[-1]["t"]) - SKIP_END_S
        rows = [r for r in rows if float(r["t"]) <= end]
    return rows


def col(rows, name):
    if not rows or name not in rows[0] or rows[0][name] == "":
        return None
    return np.array([float(r[name]) for r in rows])


def image_motion(session, rows, band=0.45):
    """Horizontal shift (px) of the top band between consecutive frames; NaN across segment breaks."""
    dx, prev, prev_seg = [], None, None
    for r in rows:
        img = cv2.imread(os.path.join(session, "frames", f"{int(r['frame']):06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        g = img[: int(img.shape[0] * band)].astype(np.float32)
        same = prev is not None and r["segment"] == prev_seg
        dx.append(cv2.phaseCorrelate(prev, g)[0][0] if same else np.nan)
        prev, prev_seg = g, r["segment"]
    return np.array(dx)


def gear_sync(session, rows, frame_ms, window=10):
    """Frames between telemetry reporting a gear change and the HUD gear digit changing on screen.

    Positive = the screen shows the shift after telemetry reports it."""
    hud_dir = os.path.join(session, "hud")
    gear = col(rows, "gear")
    if gear is None or not os.path.isdir(hud_dir):
        print("  gear shifts: no hud/ patches or gear column (recorded before they were added)")
        return None
    masks = []
    for r in rows:
        patch = cv2.imread(os.path.join(hud_dir, f"{int(r['frame']):06d}.png"))
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        masks.append(cv2.inRange(hsv, (40, 150, 150), (95, 255, 255)) > 0)   # the green gear digit
    seg = [r["segment"] for r in rows]
    change = np.array([np.count_nonzero(masks[k] ^ masks[k - 1]) if k and seg[k] == seg[k - 1] else 0
                       for k in range(len(rows))])
    thr = max(20, 0.3 * np.median([np.count_nonzero(m) for m in masks]))
    hud_events = [k for k in range(1, len(change) - 1)
                  if change[k] > thr and change[k] >= change[k - 1] and change[k] >= change[k + 1]]
    tele_events = [k for k in range(1, len(gear)) if gear[k] != gear[k - 1] and seg[k] == seg[k - 1]]
    offsets = []
    for i in tele_events:
        near = [j - i for j in hud_events if abs(j - i) <= window]
        if near:
            offsets.append(min(near, key=abs))
    print(f"  gear shifts: {len(tele_events)} in telemetry, {len(hud_events)} seen on screen, {len(offsets)} matched")
    if not offsets:
        return None
    o = np.array(offsets, float)
    print(f"  {'telemetry gear -> on-screen gear':32s} {np.mean(o):+5.1f} frames = {np.mean(o) * frame_ms:+6.0f} ms"
          f"   (median {np.median(o):+.0f}, spread {np.std(o):.1f} frames)")
    return float(np.mean(o))


def best_lag(a, b, max_lag=20):
    """Lag (frames, sub-frame via parabola) that best lines b up after a, and the correlation there."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a = np.where(ok, (a - np.nanmean(a)) / np.nanstd(a), 0)
    b = np.where(ok, (b - np.nanmean(b)) / np.nanstd(b), 0)
    lags = np.arange(-max_lag, max_lag + 1)
    c = []
    for L in lags:
        x, y = (a[:len(a) - L], b[L:]) if L >= 0 else (a[-L:], b[:len(b) + L])
        c.append(np.sum(x * y) / max(np.count_nonzero(x * y), 1))
    c = np.array(c)
    i = int(np.argmax(np.abs(c)))
    frac = 0.0
    if 0 < i < len(c) - 1:
        y0, y1, y2 = np.abs(c[i - 1:i + 2])
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom else 0.0
    return lags[i] + frac, c[i]


def main():
    if len(sys.argv) > 1:
        session = sys.argv[1]
    else:
        sessions = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "data", "recordings", "*")))
        if not sessions:
            sys.exit("No recordings in data/recordings")
        session = sessions[-1]
    rows = load(session)
    if len(rows) < 100:
        sys.exit(f"{session}: only {len(rows)} usable frames, record longer")

    t = col(rows, "t")
    frame_ms = float(np.median(np.diff(t))) * 1000
    print(f"{session}\n{len(rows)} frames, {frame_ms:.1f} ms per frame (last {SKIP_END_S:g}s skipped)\n")

    wa, ta = col(rows, "wheel_age_ms"), col(rows, "tele_age_ms")
    print(f"Freshness when each frame arrived: wheel {np.median(wa):.1f} ms old (max {wa.max():.1f}), "
          f"telemetry {np.median(ta):.1f} ms old (max {ta.max():.1f})")
    game = col(rows, "game_ms")
    if game is not None:
        drift = (game - game[0]) - (t - t[0]) * 1000
        print(f"Game clock vs PC clock drift over the session: {drift[-1] - drift[0]:+.1f} ms "
              f"(jitter {np.std(np.diff(drift)):.1f} ms)")
    print()

    steer, tsteer, yaw = col(rows, "steer_deg"), col(rows, "tele_steer"), col(rows, "yaw_rate")
    print("Measuring image motion...")
    dx = image_motion(session, rows)

    def report(name, a, b, note):
        lag, c = best_lag(a, b)
        print(f"  {name:32s} {lag:+5.1f} frames = {lag * frame_ms:+6.0f} ms   corr {c:+.2f}   {note}")
        return lag, c

    print()
    report("wheel steer -> telemetry steer", steer, tsteer, "(input path)")
    lag = None
    if yaw is not None:
        lag, c = report("yaw rate -> image motion", yaw, dx, "(capture path)")
        if abs(c) < 0.3:
            lag = None
    report("wheel steer -> image motion", steer, dx, "(input + car response + capture)")
    shift = gear_sync(session, rows, frame_ms)

    # Gear shifts are sharp events, so they're the most direct measure; yaw rate is the backup.
    best = shift if shift is not None else lag
    if best is None:
        print("\nNo sync measurement: record a new session with corners and gear shifts.")
    elif abs(best) <= 1.5:
        print(f"\nOK: the screen is within {abs(best) * frame_ms:.0f} ms of telemetry (about one frame).")
    else:
        print(f"\nCHECK: the screen is {best * frame_ms:+.0f} ms off telemetry. Shift labels by this much.")


if __name__ == "__main__":
    main()
