"""
Review a recording before training on it.

    python review.py                        # newest session in data/recordings
    python review.py data/recordings/20261003_152944

Prints a per-segment summary and flags frames that probably aren't your driving:
  takeover   Forza's steering doesn't follow your wheel (game in control, e.g. after the finish line)
  reverse    gear 0
  slow       under 20 km/h (stuck, restarting)
  wild       wheel past 90 degrees. Shown for information only, KEPT for training: failed slides end
             in a rewind/crash that the recorder and seg_edge already remove, so what's left are
             slides you caught, which are the recovery examples the model needs.
  seg_edge   first 3 s and last 3 s of every segment (late-noticed mistakes before a rewind,
             a crash before Ctrl+C, lap banners after a restart)
Writes <session>/review.png: steering over time (flags in red, segment breaks in yellow), then
the first and last frames of every segment, then a sample of flagged frames. Opens it when done.
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

STEER_UNITS_PER_DEG = 73.0
TILE = (320, 66)


def excluded_reason(session):
    """Reason from config/exclude_sessions.txt if this session is left out of training, else None."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "exclude_sessions.txt")
    if not os.path.exists(path):
        return None
    name = os.path.basename(os.path.normpath(session))
    for line in open(path):
        entry, _, reason = line.partition("#")
        if entry.strip() and name.startswith(entry.strip()):
            return reason.strip() or "listed"
    return None


def flags_for(rows):
    f = lambda k: np.array([float(r[k]) if r.get(k, "") != "" else np.nan for r in rows])
    deg, ts, v, gear = f("steer_deg"), f("tele_steer"), f("speed_mps") * 3.6, f("gear")
    ok = (np.abs(deg) > 5) & (np.abs(deg) < 60) & (np.abs(ts) < 120)
    k = float(np.nanmedian(ts[ok] / deg[ok])) if ok.sum() > 50 else np.nan
    err = np.abs(ts - np.clip(k * deg, -127, 127)) if not np.isnan(k) else np.zeros(len(rows))
    flags = {
        "takeover": np.nan_to_num(err) > 15,
        "reverse": gear == 0,
        "slow": v < 20,
        "wild": np.abs(deg) > 90,
        "seg_edge": segment_edges(rows),
    }
    return flags, deg, ts, k, v


def segment_edges(rows, before_end_s=3.0, after_start_s=3.0):
    """Last/first seconds of every segment. Ends: mistakes often start before the rewind's 5 s
    window, and the session may end in a crash. Starts: lap banners and the car settling."""
    t = np.array([float(r["t"]) for r in rows])
    seg = np.array([r["segment"] for r in rows])
    out = np.zeros(len(rows), bool)
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        ts = t[idx]
        out[idx] = (ts > ts[-1] - before_end_s) | (ts < ts[0] + after_start_s)
    return out


def timeline(rows, deg, ts, k, any_flag, width=1600, height=220):
    img = np.full((height, width, 3), 30, np.uint8)
    n = len(rows)
    x = (np.arange(n) * (width - 1) / max(n - 1, 1)).astype(int)
    lim = max(30.0, float(np.nanpercentile(np.abs(deg), 99.5)) * 1.1)
    y = lambda d: (height / 2 - np.clip(d, -lim, lim) / lim * (height / 2 - 10)).astype(int)
    for i in np.where(any_flag)[0]:
        cv2.line(img, (x[i], 0), (x[i], height), (0, 0, 140), 1)
    seg = [r["segment"] for r in rows]
    for i in range(1, n):
        if seg[i] != seg[i - 1]:
            cv2.line(img, (x[i], 0), (x[i], height), (0, 220, 255), 2)
    cv2.line(img, (0, height // 2), (width, height // 2), (80, 80, 80), 1)
    cv2.polylines(img, [np.stack([x, y(deg)], 1).astype(np.int32)], False, (255, 200, 80), 1)
    if not np.isnan(k):
        cv2.polylines(img, [np.stack([x, y(ts / k)], 1).astype(np.int32)], False, (0, 160, 255), 1)
    cv2.putText(img, f"steering: blue = your wheel, orange = Forza (scaled), +-{lim:.0f} deg. "
                "red = flagged, yellow = new segment", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def tile(session, row, text):
    im = cv2.imread(os.path.join(session, "frames", f"{int(row['frame']):06d}.jpg"))
    im = cv2.resize(im, TILE)
    cv2.putText(im, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
    cv2.putText(im, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return im


def tile_row(tiles, n=5):
    tiles = tiles[:n] + [np.zeros((TILE[1], TILE[0], 3), np.uint8)] * (n - len(tiles[:n]))
    return np.hstack(tiles)


def main():
    if len(sys.argv) > 1:
        session = sys.argv[1]
    else:
        sessions = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "data", "recordings", "*")))
        if not sessions:
            sys.exit("No recordings in data/recordings")
        session = sessions[-1]
    rows = list(csv.DictReader(open(os.path.join(session, "labels.csv"))))
    if not rows:
        sys.exit(f"{session}: no frames")
    flags, deg, ts, k, v = flags_for(rows)
    info = {"wild"}   # shown, not excluded
    any_flag = np.any(np.stack([m for n, m in flags.items() if n not in info]), axis=0)

    print(f"{session}\n{len(rows)} frames = {len(rows) / 30:.0f} s.  Forza steer per wheel degree: {k:.3f}\n")
    reason = excluded_reason(session)
    if reason:
        print(f"*** WHOLE SESSION EXCLUDED from training (config/exclude_sessions.txt): {reason}\n")
    print("Flagged frames (left out of training):")
    for name, m in flags.items():
        if name not in info:
            print(f"  {name:9s} {int(m.sum()):5d}")
    print(f"  {'total':9s} {int(any_flag.sum()):5d}  ({any_flag.mean() * 100:.1f}%)")
    kept_wild = flags["wild"] & ~any_flag
    print(f"Caught slides kept for training (wheel past 90 deg): {int(kept_wild.sum())} frames\n")

    sheet = [timeline(rows, deg, ts, k, any_flag)]
    segs = sorted({r["segment"] for r in rows}, key=int)
    print("Segments:")
    for s in segs:
        idx = [i for i, r in enumerate(rows) if r["segment"] == s]
        rt = [rows[i]["race_time"] for i in (idx[0], idx[-1])]
        print(f"  {s:>3s}: {len(idx) / 30:6.1f} s, race clock {rt[0]} -> {rt[1]}, "
              f"{int(any_flag[idx].sum())} flagged")
        first = [tile(session, rows[i], f"seg {s} start  {deg[i]:+.0f} deg") for i in idx[:1]]
        last = [tile(session, rows[i], f"seg {s} end-{(idx[-1] - i) / 30:.1f}s  {deg[i]:+.0f} deg")
                for i in np.linspace(max(idx[0], idx[-1] - 60), idx[-1], 4).astype(int)]
        sheet.append(tile_row(first + last))

    flagged = np.where(any_flag)[0]
    if len(flagged):
        pick = flagged[np.linspace(0, len(flagged) - 1, min(10, len(flagged))).astype(int)]
        tiles = [tile(session, rows[i], f"{'/'.join(n for n, m in flags.items() if m[i])}  t={rows[i]['t']}")
                 for i in pick]
        sheet += [tile_row(tiles[:5]), tile_row(tiles[5:])] if len(tiles) > 5 else [tile_row(tiles)]

    width = max(s.shape[1] for s in sheet)
    sheet = np.vstack([np.hstack([s, np.zeros((s.shape[0], width - s.shape[1], 3), np.uint8)]) for s in sheet])
    out = os.path.join(session, "review.png")
    cv2.imwrite(out, sheet)
    print(f"\nWrote {out}")
    try:
        os.startfile(out)
    except (AttributeError, OSError):
        pass


if __name__ == "__main__":
    main()
