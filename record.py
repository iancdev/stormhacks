"""
Forza AI wheel project - frame + wheel/pedal recorder.

Run from the activated venv, in this order:

    python record.py setup              # grab a screen while driving, drag the crop + mask boxes
    python record.py preview            # grab a fresh screen and show what will be saved
    python record.py record             # record frames + steering/gas/brake (Ctrl+C to stop)
    python record.py wheel              # check the TMX readout (RawInput, leaves Forza's FFB alone)

Options:
    setup/preview --image PATH          # use a saved screenshot instead of grabbing the screen
    setup/preview --delay N             # seconds to alt-tab into Forza before the grab (default 5)
    record --fps N                      # capture rate (default 30)
    record --vjoy                       # also pass TMX steering/pedals through to vJoy (no FFB)
    record --no-telemetry               # record even without Forza Data Out (no speed, no pause)
    record --rewind-button N            # pressing TMX button N (your Forza rewind) drops the last
                                        # --drop-seconds (default 5) of frames + the rewind itself
    (rewinds are also detected automatically when Forza's race clock jumps backwards;
     --no-auto-rewind turns that off)

Crop/mask settings live in config/capture.json. Recordings go to data/recordings/<timestamp>/:
frames/000000.jpg ..., labels.csv (one row per frame) and meta.json.
"""
import argparse
import csv
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config", "capture.json")
DATA_DIR = os.path.join(ROOT, "data")

STEER_UNITS_PER_DEG = 73.0   # TMX a0, 900 deg rotation
MODEL_SIZE = (200, 66)       # PilotNet input, shown in the preview only
VJOY_REST = {"X": 0x4000, "Y": 0x8000, "Z": 0x8000}  # centred, brake released, gas released

# hud_box: the gear number on the speedometer (screen coords, 1920x1080 FH4 HUD). Saved as a tiny
# patch per frame so sync_check.py can line up on-screen gear changes with telemetry gear changes.
DEFAULT_CONFIG = {"monitor": 0, "crop": None, "masks": [], "save_width": 320,
                  "hud_box": [1700, 852, 1756, 904]}


# ---------------------------------------------------------------- config + image helpers

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH) as f:
        return {**DEFAULT_CONFIG, **json.load(f)}


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved {CONFIG_PATH}")


def save_size(cfg):
    x0, y0, x1, y1 = cfg["crop"]
    w = cfg["save_width"]
    h = int(round(w * (y1 - y0) / (x1 - x0) / 2)) * 2
    return w, h


def masks_in_crop(cfg):
    """Mask boxes are stored in screen coordinates; shift them into crop coordinates."""
    x0, y0 = cfg["crop"][:2]
    return [(mx0 - x0, my0 - y0, mx1 - x0, my1 - y0) for mx0, my0, mx1, my1 in cfg["masks"]]


def process(crop_img, masks, size):
    """Black out mask boxes, then shrink to the saved size. Same code for preview and record."""
    out = crop_img.copy()
    for mx0, my0, mx1, my1 in masks:
        out[max(my0, 0):max(my1, 0), max(mx0, 0):max(mx1, 0)] = 0
    return cv2.resize(out, size, interpolation=cv2.INTER_AREA)


def grab_screen(monitor, delay):
    import dxcam
    for s in range(delay, 0, -1):
        print(f"\rSwitch to Forza and drive (HUD visible) - grabbing in {s}s ", end="", flush=True)
        time.sleep(1)
    print()
    cam = dxcam.create(output_idx=monitor, output_color="BGR")
    frame = None
    for _ in range(100):
        frame = cam.grab()
        if frame is not None:
            break
        time.sleep(0.02)
    cam.release()
    if frame is None:
        sys.exit("dxcam returned no frame. Is Forza in borderless/windowed mode?")
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "setup_screen.png")
    cv2.imwrite(path, frame)
    print(f"Saved full screenshot to {path}")
    return frame


def get_screen(args, cfg):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Could not read {args.image}")
        return img
    return grab_screen(cfg["monitor"], args.delay)


def front_window(name):
    """Open an OpenCV window on top, so it doesn't hide behind Forza after the screen grab."""
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    return name


def fit_scale(img, max_w=1600, max_h=850):
    h, w = img.shape[:2]
    return min(1.0, max_w / w, max_h / h)


# ---------------------------------------------------------------- setup + preview

def cmd_setup(args):
    cfg = load_config()
    screen = get_screen(args, cfg)
    s = fit_scale(screen)
    disp = cv2.resize(screen, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    print("\nStep 1: drag a box around the part of the screen to KEEP (road ahead).\n"
          "        Cut the sky at the top and the bonnet at the bottom. Enter/Space = accept, c = cancel.")
    x, y, w, h = cv2.selectROI(front_window("1) crop: keep this area"), disp, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        sys.exit("No crop selected, config unchanged.")
    crop = [int(x / s), int(y / s), int((x + w) / s), int((y + h) / s)]
    crop_img = screen[crop[1]:crop[3], crop[0]:crop[2]]

    cs = fit_scale(crop_img)
    crop_disp = cv2.resize(crop_img, None, fx=cs, fy=cs, interpolation=cv2.INTER_AREA)
    print("\nStep 2: drag a box over each HUD element still inside the crop (minimap, speedometer...).\n"
          "        Enter/Space after each box, Esc when done (Esc straight away = no masks).")
    rects = cv2.selectROIs(front_window("2) masks: black these out"), crop_disp, showCrosshair=False)
    cv2.destroyAllWindows()
    masks = []
    for mx, my, mw, mh in (rects if len(rects) else []):
        if mw and mh:
            masks.append([crop[0] + int(mx / cs), crop[1] + int(my / cs),
                          crop[0] + int((mx + mw) / cs), crop[1] + int((my + mh) / cs)])

    cfg.update(crop=crop, masks=masks)
    save_config(cfg)
    print(f"crop={crop} ({crop[2]-crop[0]}x{crop[3]-crop[1]}), {len(masks)} mask(s), "
          f"saved frames will be {save_size(cfg)[0]}x{save_size(cfg)[1]}")
    show_preview(screen, cfg)


def cmd_preview(args):
    cfg = load_config()
    if not cfg["crop"]:
        sys.exit("No crop yet. Run: python record.py setup")
    show_preview(get_screen(args, cfg), cfg)


def show_preview(screen, cfg):
    x0, y0, x1, y1 = cfg["crop"]
    size = save_size(cfg)

    # left: full screen with the crop (green) and masks (red)
    overlay = screen.copy()
    for mx0, my0, mx1, my1 in cfg["masks"]:
        cv2.rectangle(overlay, (mx0, my0), (mx1, my1), (0, 0, 255), -1)
    full = cv2.addWeighted(overlay, 0.45, screen, 0.55, 0)
    cv2.rectangle(full, (x0, y0), (x1, y1), (0, 255, 0), 4)
    if cfg.get("hud_box"):
        hx0, hy0, hx1, hy1 = cfg["hud_box"]
        cv2.rectangle(full, (hx0, hy0), (hx1, hy1), (255, 128, 0), 4)   # blue: sync patch, not training
    full = cv2.resize(full, (960, int(960 * screen.shape[0] / screen.shape[1])), interpolation=cv2.INTER_AREA)

    # right: exactly what gets saved, and the squashed model input, both enlarged
    saved = process(screen[y0:y1, x0:x1], masks_in_crop(cfg), size)
    model = cv2.resize(saved, MODEL_SIZE, interpolation=cv2.INTER_AREA)
    saved_big = cv2.resize(saved, (960, int(960 * size[1] / size[0])), interpolation=cv2.INTER_NEAREST)
    model_big = cv2.resize(model, (960, int(960 * MODEL_SIZE[1] / MODEL_SIZE[0])), interpolation=cv2.INTER_NEAREST)

    def label(img, text):
        img = img.copy()
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    right = np.vstack([label(saved_big, f"saved frame {size[0]}x{size[1]}"),
                       np.zeros((10, 960, 3), np.uint8),
                       label(model_big, f"model input {MODEL_SIZE[0]}x{MODEL_SIZE[1]} (squashed)")])
    left = label(full, "green = crop, red = masked")
    h = max(left.shape[0], right.shape[0])
    pad = lambda im: np.vstack([im, np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)])
    sheet = np.hstack([pad(left), np.zeros((h, 10, 3), np.uint8), pad(right)])

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "crop_preview.png")
    cv2.imwrite(path, sheet)
    print(f"Preview saved to {path}. Check no HUD is left in the saved frame. Any key closes it.")
    s = fit_scale(sheet)
    cv2.imshow(front_window("preview (any key closes)"), cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- recording threads

class WheelReader(threading.Thread):
    """Reads the TMX at ~250 Hz (and optionally mirrors it to vJoy). All SDL calls stay on this thread."""

    def __init__(self, vjoy):
        super().__init__(daemon=True)
        self.vjoy = vjoy
        self.latest = None          # (t, steer_raw, brake_raw, gas_raw)
        # SDL reads 0 on every axis until the wheel sends its first report, and raw 0 on a pedal
        # means half pressed. Only trust (and pass through) the axes once each one has moved.
        self.live = False
        self.buttons = frozenset()  # SDL button indices currently held (same numbers as check.py wheel)
        self.error = None
        self.ready = threading.Event()
        self.stop = threading.Event()

    def run(self):
        try:
            self._run()
        except Exception as e:  # report to the main thread instead of dying silently
            self.error = repr(e)
            self.ready.set()

    def _run(self):
        import sdl2
        sdl2.SDL_SetHint(sdl2.SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, b"1")
        # Read the wheel through RawInput, never DirectInput: SDL's DirectInput backend acquires
        # force-feedback wheels exclusively, which steals the TMX motor from Forza (FFB goes dead).
        sdl2.SDL_SetHint(sdl2.SDL_HINT_DIRECTINPUT_ENABLED, b"0")
        sdl2.SDL_SetHint(sdl2.SDL_HINT_JOYSTICK_RAWINPUT, b"1")
        if sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK) != 0:
            raise RuntimeError("SDL init failed: " + sdl2.SDL_GetError().decode())
        js = None
        deadline = time.perf_counter() + 3   # RawInput lists devices a moment after init
        while not js and time.perf_counter() < deadline:
            sdl2.SDL_JoystickUpdate()
            for i in range(sdl2.SDL_NumJoysticks()):
                name = (sdl2.SDL_JoystickNameForIndex(i) or b"?").decode(errors="replace").lower()
                if "vjoy" not in name and ("tmx" in name or "thrustmaster" in name):
                    js = sdl2.SDL_JoystickOpen(i)
                    break
            time.sleep(0.05)
        if not js:
            raise RuntimeError("No Thrustmaster/TMX wheel found (plugged in? HidHide allowing this python.exe?)")

        j = None
        if self.vjoy:
            import pyvjoy
            j = pyvjoy.VJoyDevice(1)
            axes = {"X": pyvjoy.HID_USAGE_X, "Y": pyvjoy.HID_USAGE_Y, "Z": pyvjoy.HID_USAGE_Z}
            for k, v in VJOY_REST.items():
                j.set_axis(axes[k], v)

        def to_vjoy(raw):  # -32768..32767 -> 0x1..0x8000 (pedals stay inverted, like the TMX)
            return 1 + (raw + 32768) * 0x7FFF // 65535

        n_buttons = sdl2.SDL_JoystickNumButtons(js)
        first, moved = None, [False, False, False]
        try:
            while not self.stop.is_set():
                sdl2.SDL_JoystickUpdate()
                steer = sdl2.SDL_JoystickGetAxis(js, 0)
                brake = sdl2.SDL_JoystickGetAxis(js, 1)
                gas = sdl2.SDL_JoystickGetAxis(js, 2)
                self.latest = (time.perf_counter(), steer, brake, gas)
                self.buttons = frozenset(i for i in range(n_buttons) if sdl2.SDL_JoystickGetButton(js, i))
                if not self.live:
                    first = first or (steer, brake, gas)
                    moved = [m or v != f for m, v, f in zip(moved, (steer, brake, gas), first)]
                    self.live = all(moved)
                if j and self.live:
                    j.set_axis(axes["X"], to_vjoy(steer))
                    j.set_axis(axes["Y"], to_vjoy(brake))
                    j.set_axis(axes["Z"], to_vjoy(gas))
                self.ready.set()
                time.sleep(0.004)
        finally:
            if j:
                for k, v in VJOY_REST.items():
                    j.set_axis(axes[k], v)
            sdl2.SDL_JoystickClose(js)
            sdl2.SDL_Quit()


class TelemetryReader(threading.Thread):
    """Keeps the latest Forza Data Out packet (FH4 324-byte dash layout)."""

    def __init__(self, port):
        super().__init__(daemon=True)
        # (t, race_on, speed_mps, tele_steer, race_time, distance, yaw_rate, game_ms)
        self.latest = None
        # Rewind watchdog: the race clock only runs forwards while driving, so any step back means
        # Forza rewound (seen either during the rewind, or on the first packet after it).
        self.last_backjump = -1e9
        self.stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)

    def run(self):
        prev_race_time = None
        while not self.stop.is_set():
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            if len(data) < 321:
                continue
            now = time.perf_counter()
            race_on = struct.unpack_from("<i", data, 0)[0]
            race_time = struct.unpack_from("<f", data, 308)[0]
            if race_on == 1:
                if prev_race_time is not None and race_time < prev_race_time - 0.05:
                    self.last_backjump = now   # set before latest, so the main loop never misses it
                prev_race_time = race_time
            self.latest = (now, race_on,
                           struct.unpack_from("<f", data, 256)[0],
                           struct.unpack_from("<b", data, 320)[0],
                           race_time,
                           struct.unpack_from("<f", data, 292)[0],
                           struct.unpack_from("<f", data, 48)[0],    # AngularVelocityY (yaw rate, rad/s)
                           struct.unpack_from("<I", data, 4)[0],     # TimestampMS (game clock)
                           data[319])                                # Gear
        self.sock.close()


COLUMNS = ["frame", "segment", "t", "steer_raw", "steer_deg", "brake", "gas",
           "wheel_age_ms", "speed_mps", "race_on", "tele_steer", "tele_age_ms", "race_time", "distance",
           "yaw_rate", "game_ms", "gear"]


class Writer(threading.Thread):
    """Encodes JPEGs and appends CSV rows off the capture thread, so disk hiccups don't cost frames."""

    def __init__(self, session_dir, quality):
        super().__init__(daemon=True)
        self.frames_dir = os.path.join(session_dir, "frames")
        self.hud_dir = os.path.join(session_dir, "hud")
        os.makedirs(self.frames_dir)
        os.makedirs(self.hud_dir)
        self.csv_file = open(os.path.join(session_dir, "labels.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(COLUMNS)
        self.params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        self.q = queue.Queue(maxsize=300)

    def run(self):
        n = 0
        while True:
            item = self.q.get()
            if item is None:
                break
            img, hud, row = item
            cv2.imwrite(os.path.join(self.frames_dir, f"{row[0]:06d}.jpg"), img, self.params)
            if hud is not None:
                cv2.imwrite(os.path.join(self.hud_dir, f"{row[0]:06d}.png"), hud)
            self.csv.writerow(row)
            n += 1
            if n % 100 == 0:
                self.csv_file.flush()
        self.csv_file.close()


# ---------------------------------------------------------------- record

def cmd_record(args):
    import dxcam
    cfg = load_config()
    if not cfg["crop"]:
        sys.exit("No crop yet. Run: python record.py setup")
    size = save_size(cfg)
    masks = masks_in_crop(cfg)

    wheel = WheelReader(args.vjoy)
    wheel.start()
    wheel.ready.wait(5)
    if wheel.error or wheel.latest is None:
        sys.exit(f"Wheel error: {wheel.error or 'no reading within 5 s'}")

    tele = None
    if not args.no_telemetry:
        try:
            tele = TelemetryReader(args.port)
            tele.start()
        except OSError as e:
            wheel.stop.set()
            sys.exit(f"Can't listen on 127.0.0.1:{args.port} ({e}). Close check.py telemetry, "
                     "or use --no-telemetry.")

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(DATA_DIR, "recordings", session)
    writer = Writer(session_dir, args.quality)
    writer.start()

    cam = dxcam.create(output_idx=cfg["monitor"], output_color="BGR")
    # One capture region covering the crop and the gear patch; both are sliced out of each frame.
    hud_box = cfg.get("hud_box")
    boxes = [cfg["crop"]] + ([hud_box] if hud_box else [])
    region = (min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes))
    cx0, cy0, cx1, cy1 = (v - o for v, o in zip(cfg["crop"], region[:2] * 2))
    hx0, hy0, hx1, hy1 = (v - o for v, o in zip(hud_box, region[:2] * 2)) if hud_box else (0, 0, 0, 0)
    cam.start(region=region, target_fps=args.fps, video_mode=True)

    print(f"Recording to {session_dir}\n"
          f"crop={cfg['crop']} masks={len(cfg['masks'])} saved={size[0]}x{size[1]} fps={args.fps} "
          f"vjoy={'on' if args.vjoy else 'off'} telemetry={'on' if tele else 'off'}\n"
          + ("Frames are only kept while Forza reports IsRaceOn=1 (driving, not paused/menus).\n" if tele else "")
          + (f"Rewind (button {args.rewind_button}) throws away the last {args.drop_seconds:g}s of frames.\n"
             if args.rewind_button is not None else "")
          + "Ctrl+C to stop.\n")

    frame_idx, segment, recording = 0, -1, False
    dropped = discarded = rewinds = 0
    was_rewinding = False
    times = []                       # (segment, t) of kept frames, for the gap report
    # Frames wait here for drop_seconds before going to disk, so a rewind can still take them back.
    pending = deque()

    def flush(block=False):
        nonlocal dropped
        pt, seg, im, hud, row = pending.popleft()
        try:
            writer.q.put((im, hud, row), block=block)
            times.append((seg, pt))
        except queue.Full:
            dropped += 1

    t_start, last_status = time.perf_counter(), 0.0
    try:
        while True:
            img = cam.get_latest_frame()
            t = time.perf_counter()
            if img is None:
                continue
            wt, steer, brake_raw, gas_raw = wheel.latest
            if wheel.error:
                raise RuntimeError("Wheel thread died: " + wheel.error)

            tl = tele.latest if tele else None
            tele_fresh = tl is not None and t - tl[0] < 0.5
            if not wheel.live:
                active, state = False, "turn wheel + press both pedals once"
            elif tele:
                active = tele_fresh and tl[1] == 1
                state = "REC" if active else ("paused (IsRaceOn=0)" if tele_fresh else "waiting for telemetry")
            else:
                active, state = True, "REC"

            # Rewind: the last few seconds were the mistake, and the rewind playback itself is
            # backwards footage. Drop both; driving resumes in a new segment afterwards.
            # Detected by the race clock going backwards (watchdog) or the rewind button.
            button = args.rewind_button is not None and args.rewind_button in wheel.buttons
            auto = tele is not None and not args.no_auto_rewind and t - tele.last_backjump < 0.3
            rewinding = button or auto
            if rewinding:
                if not was_rewinding:
                    rewinds += 1
                    discarded += len(pending)
                    pending.clear()
                active, state = False, f"rewind, dropped last {args.drop_seconds:g}s"
            was_rewinding = rewinding

            if active and not recording:
                segment += 1          # new segment after every pause: don't shift labels across gaps
            recording = active

            if active:
                brake = (32767 - brake_raw) / 65535
                gas = (32767 - gas_raw) / 65535
                row = [frame_idx, segment, f"{t - t_start:.4f}", steer, f"{steer / STEER_UNITS_PER_DEG:.2f}",
                       f"{brake:.4f}", f"{gas:.4f}", f"{(t - wt) * 1000:.1f}",
                       f"{tl[2]:.3f}" if tl else "", tl[1] if tl else "",
                       tl[3] if tl else "", f"{(t - tl[0]) * 1000:.1f}" if tl else "",
                       f"{tl[4]:.3f}" if tl else "", f"{tl[5]:.1f}" if tl else "",
                       f"{tl[6]:.4f}" if tl else "", tl[7] if tl else "", tl[8] if tl else ""]
                crop_img = img[cy0:cy1, cx0:cx1]
                hud = img[hy0:hy1, hx0:hx1].copy() if hud_box else None
                pending.append((t, segment, process(crop_img, masks, size), hud, row))
                frame_idx += 1
                # Only flush while driving: if telemetry stops during a rewind, the frames before it
                # must still be here when the clock jump shows up on the first packet after it.
                while pending and t - pending[0][0] > args.drop_seconds:
                    flush()

            if t - last_status > 0.5:
                last_status = t
                speed = f"{tl[2] * 3.6:5.0f} km/h" if tl else "   -- km/h"
                print(f"\r[{state:22s}] saved={len(times):6d} seg={max(segment, 0):3d} "
                      f"steer={steer / STEER_UNITS_PER_DEG:+7.1f}deg gas={(32767 - gas_raw) / 65535:4.2f} "
                      f"brake={(32767 - brake_raw) / 65535:4.2f} {speed} rewinds={rewinds} dropped={dropped}   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.stop()
        cam.release()
        wheel.stop.set()
        if tele:
            tele.stop.set()
        # People usually stop right after a mistake, so Ctrl+C throws away the last few seconds too.
        discarded_at_stop = len(pending)
        pending.clear()
        writer.q.put(None)
        writer.join()
        wheel.join(2)

    report = gap_report(times, args.fps)
    meta = {"session": session, "config": cfg, "saved_size": size, "fps_target": args.fps,
            "frames": len(times), "segments": segment + 1, "dropped": dropped,
            "rewinds": rewinds, "discarded_by_rewind": discarded, "discarded_at_stop": discarded_at_stop,
            "drop_seconds": args.drop_seconds,
            "rewind_button": args.rewind_button, "vjoy": args.vjoy,
            "telemetry": tele is not None, "steer_units_per_deg": STEER_UNITS_PER_DEG,
            "pedals": "0 = released, 1 = floored", "gaps": report}
    with open(os.path.join(session_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(times)} frames in {segment + 1} segment(s) to {session_dir}\n"
          f"Rewinds: {rewinds} ({discarded} frames thrown away). Last {args.drop_seconds:g}s before "
          f"Ctrl+C thrown away ({discarded_at_stop} frames). Dropped (disk too slow): {dropped}")
    if report:
        print(f"Frame gaps (ms): median {report['median_ms']}, p99 {report['p99_ms']}, "
              f"max {report['max_ms']}, {report['over_2x']} gap(s) > {2000 / args.fps:.0f} ms")


def cmd_wheel(args):
    """Show what the recorder reads from the TMX, and the range each axis covered."""
    wheel = WheelReader(False)
    wheel.start()
    wheel.ready.wait(5)
    if wheel.error or wheel.latest is None:
        sys.exit(f"Wheel error: {wheel.error or 'no reading within 5 s'}")
    print("Turn the wheel fully both ways, press each pedal fully, press any buttons.\n"
          "Works with Forza focused too (check the ranges afterwards). Ctrl+C to stop.\n")
    lo, hi, seen = [32767] * 3, [-32768] * 3, set()
    try:
        while True:
            _, *raw = wheel.latest
            lo = [min(a, b) for a, b in zip(lo, raw)]
            hi = [max(a, b) for a, b in zip(hi, raw)]
            seen |= wheel.buttons
            steer, brake, gas = raw
            print(f"\rsteer {steer:+6d} ({steer / STEER_UNITS_PER_DEG:+6.1f} deg)  "
                  f"brake {brake:+6d} ({(32767 - brake) / 65535:4.2f})  gas {gas:+6d} ({(32767 - gas) / 65535:4.2f})  "
                  f"buttons {sorted(wheel.buttons)}      ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    wheel.stop.set()
    wheel.join(2)
    print("\n\nRange seen (expect steer about -32768..32767, pedals +32767 released .. -32768 floored):")
    for name, a, b in zip(("steer", "brake", "gas"), lo, hi):
        print(f"  {name:5s} {a:+6d} .. {b:+6d}")
    print(f"  buttons pressed: {sorted(seen)}")


def gap_report(times, fps):
    gaps = [(b[1] - a[1]) * 1000 for a, b in zip(times, times[1:]) if a[0] == b[0]]
    if not gaps:
        return None
    g = np.array(gaps)
    return {"median_ms": round(float(np.median(g)), 1), "p99_ms": round(float(np.percentile(g, 99)), 1),
            "max_ms": round(float(g.max()), 1), "over_2x": int((g > 2000 / fps).sum())}


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "preview"):
        sp = sub.add_parser(name)
        sp.add_argument("--image", help="use a saved screenshot instead of grabbing the screen")
        sp.add_argument("--delay", type=int, default=5, help="seconds before the screen grab")
    rp = sub.add_parser("record")
    rp.add_argument("--fps", type=int, default=30)
    rp.add_argument("--quality", type=int, default=90, help="JPEG quality")
    rp.add_argument("--vjoy", action="store_true", help="pass TMX steering/pedals through to vJoy")
    rp.add_argument("--no-telemetry", action="store_true")
    rp.add_argument("--port", type=int, default=9999)
    rp.add_argument("--rewind-button", type=int, default=None,
                    help="TMX button bound to Rewind in Forza (number from: python utils\\test.py wheel)")
    rp.add_argument("--drop-seconds", type=float, default=5.0,
                    help="seconds of frames thrown away on a rewind")
    rp.add_argument("--no-auto-rewind", action="store_true",
                    help="turn off the race-clock rewind watchdog (button only)")
    sub.add_parser("wheel", help="live TMX readout as the recorder sees it (RawInput)")
    args = p.parse_args()
    {"setup": cmd_setup, "preview": cmd_preview, "record": cmd_record, "wheel": cmd_wheel}[args.cmd](args)


if __name__ == "__main__":
    main()
