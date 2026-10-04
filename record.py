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
    record --fps N                      # fresh capture target (default 30, measured rate reported)
    record --vjoy                       # also pass TMX steering/pedals through to vJoy (no FFB)
    record --no-telemetry               # record even without Forza Data Out (no speed, no pause)
    record --rewind-button N            # pressing TMX button N (your Forza rewind) drops the last
                                        # --drop-seconds (default 10) of frames + the rewind itself
    (rewinds are also detected automatically when Forza's race clock or distance travelled jumps
     backwards; --no-auto-rewind turns that off. Any pause also drops the held-back frames, because
     car resets don't move either; --keep-before-pause turns that off)

Crop/mask settings live in config/capture.json. Recordings go to data/recordings/<timestamp>/:
frames/000000.jpg ..., hud/*.png, labels.csv, capture_timing.csv, and meta.json.
labels.csv t remains rounded post-retrieval host time. The timing sidecar separately
records host capture-start/retrieval and input-poll timestamps, not exact game-render
or USB-report times. Rewinds, game takeovers and stopping intentionally discard pending
frames: saved frame IDs may have gaps. Metadata counts explain those discards.
"""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
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
        self.latest_ns = None       # same atomic snapshot with integer perf_counter_ns time
        # SDL reads 0 on every axis until the wheel sends its first report, and raw 0 on a pedal
        # means half pressed. Only trust (and pass through) the axes once each one has moved.
        self.live = False
        self.buttons = frozenset()
        self.error = None
        self.ready = threading.Event()
        self.stop = threading.Event()

    def run(self):
        try:
            self._run()
        except Exception as e:  # report to the main thread instead of dying silently
            self._fail(repr(e))

    def _fail(self, message):
        # Publish the failure before clearing samples so the capture thread can
        # report the actual cause even when detach races with its cached read.
        self.error = message
        self.live = False
        self.buttons = frozenset()
        self.latest = self.latest_ns = None
        self.ready.set()

    def _run(self):
        import sdl2
        sdl2.SDL_SetHint(sdl2.SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, b"1")
        # RawInput keeps the recorder from taking Forza's DirectInput FFB ownership.
        sdl2.SDL_SetHint(sdl2.SDL_HINT_DIRECTINPUT_ENABLED, b"0")
        sdl2.SDL_SetHint(sdl2.SDL_HINT_JOYSTICK_RAWINPUT, b"1")
        if sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK) != 0:
            raise RuntimeError("SDL init failed: " + sdl2.SDL_GetError().decode())
        js, j = None, None
        def to_vjoy(raw):  # -32768..32767 -> 0x1..0x8000 (pedals stay inverted, like the TMX)
            return 1 + (raw + 32768) * 0x7FFF // 65535

        def require_attached():
            if not sdl2.SDL_JoystickGetAttached(js):
                # Invalidate immediately, before potentially slow vJoy/SDL
                # cleanup. Detached handles may still expose cached axis values.
                self._fail("TMX disconnected")
                raise RuntimeError("TMX disconnected")

        first, moved = None, [False, False, False]
        try:
            deadline = time.perf_counter() + 3
            while not js and not self.stop.is_set() and time.perf_counter() < deadline:
                sdl2.SDL_JoystickUpdate()
                for i in range(sdl2.SDL_NumJoysticks()):
                    name = (sdl2.SDL_JoystickNameForIndex(i) or b"?").decode(errors="replace").lower()
                    if "vjoy" not in name and ("tmx" in name or "thrustmaster" in name):
                        js = sdl2.SDL_JoystickOpen(i)
                        break
                if not js:
                    self.stop.wait(.05)
            if not js:
                raise RuntimeError("No Thrustmaster/TMX wheel found (plugged in? HidHide allowing this python.exe?)")
            if self.vjoy:
                import pyvjoy
                j = pyvjoy.VJoyDevice(1)
                axes = {"X": pyvjoy.HID_USAGE_X, "Y": pyvjoy.HID_USAGE_Y, "Z": pyvjoy.HID_USAGE_Z}
                for k, v in VJOY_REST.items():
                    j.set_axis(axes[k], v)
            n_buttons = sdl2.SDL_JoystickNumButtons(js)
            while not self.stop.is_set():
                require_attached()
                sdl2.SDL_JoystickUpdate()
                steer = sdl2.SDL_JoystickGetAxis(js, 0)
                brake = sdl2.SDL_JoystickGetAxis(js, 1)
                gas = sdl2.SDL_JoystickGetAxis(js, 2)
                buttons = frozenset(i for i in range(n_buttons) if sdl2.SDL_JoystickGetButton(js, i))
                require_attached()
                sampled_ns = time.perf_counter_ns()
                self.buttons = buttons
                self.latest_ns = (sampled_ns, steer, brake, gas)
                self.latest = (sampled_ns / 1e9, steer, brake, gas)
                if not self.live:
                    first = first or (steer, brake, gas)
                    moved = [m or v != f for m, v, f in zip(moved, (steer, brake, gas), first)]
                    self.live = all(moved)
                if j and self.live:
                    j.set_axis(axes["X"], to_vjoy(steer))
                    j.set_axis(axes["Y"], to_vjoy(brake))
                    j.set_axis(axes["Z"], to_vjoy(gas))
                self.ready.set()
                self.stop.wait(0.004)
        finally:
            try:
                if j:
                    for k, v in VJOY_REST.items():
                        j.set_axis(axes[k], v)
            finally:
                try:
                    if js:
                        sdl2.SDL_JoystickClose(js)
                finally:
                    sdl2.SDL_Quit()


class TelemetryReader(threading.Thread):
    """Keeps the latest Forza Data Out packet (FH4 324-byte dash layout)."""

    def __init__(self, port):
        super().__init__(daemon=True)
        # (t, race_on, speed, steer, race_time, distance, yaw_rate, game_ms,
        #  gear, car_ordinal, car_class, car_pi)
        self.latest = None
        self.last_backjump = -1e9
        self.latest_ns = None       # same snapshot with precise host receive time
        self.error = None
        self.stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("127.0.0.1", port))
            self.sock.settimeout(0.2)
        except BaseException:
            self.sock.close()
            raise

    def run(self):
        prev_race_time = prev_distance = None
        try:
            while not self.stop.is_set():
                try:
                    data, _ = self.sock.recvfrom(1024)
                except socket.timeout:
                    continue
                received_ns = time.perf_counter_ns()
                if len(data) < 321:
                    continue
                values = (struct.unpack_from("<i", data, 0)[0],
                          struct.unpack_from("<f", data, 256)[0],
                          struct.unpack_from("<b", data, 320)[0],
                          struct.unpack_from("<f", data, 308)[0],
                          struct.unpack_from("<f", data, 292)[0],
                          struct.unpack_from("<f", data, 48)[0],
                          struct.unpack_from("<I", data, 4)[0], data[319],
                          struct.unpack_from("<i", data, 212)[0],
                          struct.unpack_from("<i", data, 216)[0],
                          struct.unpack_from("<i", data, 220)[0])
                if (values[0] not in (0, 1) or values[1] < 0
                        or not all(math.isfinite(values[i]) for i in (1, 3, 4, 5))):
                    continue
                if values[0] == 1:
                    # Some modes (long events/free roam) reset or rewind the car without
                    # turning the race clock back, but distance travelled does go back.
                    if ((prev_race_time is not None and values[3] < prev_race_time - .05)
                            or (prev_distance is not None and values[4] < prev_distance - 5.0)):
                        self.last_backjump = received_ns / 1e9
                    prev_race_time, prev_distance = values[3], values[4]
                self.latest_ns = (received_ns, *values)
                self.latest = (received_ns / 1e9, *values)
        except Exception as error:
            self.error = repr(error)
        finally:
            self.sock.close()


COLUMNS = ["frame", "segment", "t", "steer_raw", "steer_deg", "brake", "gas",
           "wheel_age_ms", "speed_mps", "race_on", "tele_steer", "tele_age_ms", "race_time", "distance",
           "yaw_rate", "game_ms", "gear", "car_ordinal", "car_class", "car_pi"]


TIMING_COLUMNS = ["frame", "segment", "capture_index", "capture_start_ns", "retrieved_ns",
                  "wheel_sample_ns", "telemetry_sample_ns"]
PRODUCER_SCHEMA = "record_py_buffered_20_v1"


def capture_fps(value):
    """Argparse and programmatic validation before opening any hardware."""
    try:
        fps = float(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("fps must be finite and within (0, 240]") from error
    if not math.isfinite(fps) or not 0 < fps <= 240:
        raise argparse.ArgumentTypeError("fps must be finite and within (0, 240]")
    return fps


class FreshCapture:
    """Paced one-shot capture: None never reuses or relabels a previous image.

    DXcam v0.3.0 source: https://github.com/ra1nty/DXcam/blob/v0.3.0/dxcam/dxcam.py
    grab(new_frame_only=True) applies only while its producer is stopped. Never
    start a ring buffer here or claim ownership of an already-running camera.
    """

    def __init__(self, camera, region, fps, *, clock_ns=None, sleep=None):
        self.camera, self.region = camera, tuple(region)
        self.period_ns = round(1e9 / capture_fps(fps))
        self.clock_ns = clock_ns or time.perf_counter_ns
        self.sleep = sleep or time.sleep
        self.next_start_ns = None
        self.attempts = 0
        self.no_frame_polls = 0
        self.times = []
        if camera.is_capturing:
            raise RuntimeError("DXcam output is already capturing; exclusive camera ownership required")

    def grab(self):
        if self.camera.is_capturing:
            raise RuntimeError("DXcam ring-buffer capture must remain stopped")
        now = self.clock_ns()
        if self.next_start_ns is not None and now < self.next_start_ns:
            self.sleep((self.next_start_ns - now) / 1e9)
        started = self.clock_ns()
        self.next_start_ns = started + self.period_ns  # No catch-up bursts after a slow frame.
        image = self.camera.grab(region=self.region, copy=True, new_frame_only=True)
        returned = self.clock_ns()
        if returned < started or (self.times and started <= self.times[-1][1]):
            raise RuntimeError("capture clock reversed or repeated")
        self.attempts += 1
        if image is None:
            self.no_frame_polls += 1
        else:
            expected = (self.region[3] - self.region[1], self.region[2] - self.region[0], 3)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != expected:
                raise ValueError("fresh capture must be uint8 BGR matching the full road/HUD region")
            self.times.append((0, started))
        return image, started, returned


class Writer(threading.Thread):
    """Encodes JPEGs and appends CSV rows off the capture thread, so disk hiccups don't cost frames."""

    def __init__(self, session_dir, quality):
        super().__init__(daemon=True)
        self.session_dir = session_dir
        self.frames_dir = os.path.join(session_dir, "frames")
        self.hud_dir = os.path.join(session_dir, "hud")
        os.makedirs(self.frames_dir)
        os.makedirs(self.hud_dir)
        self.params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        self.q = queue.Queue(maxsize=300)
        self.stop = threading.Event()
        self.error = None
        self.saved = 0
        self.last_frame_id = -1
        self.times = []
        self.capture_times = []

    def run(self):
        try:
            with ExitStack() as files:
                labels_file = files.enter_context(open(os.path.join(self.session_dir, "labels.csv"), "x", newline=""))
                timing_file = files.enter_context(open(os.path.join(self.session_dir, "capture_timing.csv"), "x", newline=""))
                labels, timing = csv.writer(labels_file), csv.writer(timing_file)
                labels.writerow(COLUMNS)
                timing.writerow(TIMING_COLUMNS)
                while not self.stop.is_set() or not self.q.empty():
                    try:
                        item = self.q.get(timeout=.05)
                    except queue.Empty:
                        continue
                    try:
                        img, hud, row, timestamps = item
                        if type(row[0]) is not int or row[0] <= self.last_frame_id:
                            raise ValueError("saved frame IDs must be strictly increasing")
                        if len(row) != len(COLUMNS) or timestamps[:2] != row[:2]:
                            raise ValueError("label/timing frame identity mismatch")
                        path = os.path.join(self.frames_dir, f"{row[0]:06d}.jpg")
                        if not cv2.imwrite(path, img, self.params):
                            raise OSError(f"JPEG write failed: {path}")
                        if hud is not None:
                            hud_path = os.path.join(self.hud_dir, f"{row[0]:06d}.png")
                            if not cv2.imwrite(hud_path, hud):
                                raise OSError(f"HUD patch write failed: {hud_path}")
                        labels.writerow(row)
                        timing.writerow(timestamps)
                        self.saved += 1
                        self.last_frame_id = row[0]
                        self.times.append((row[1], timestamps[4] / 1e9))
                        self.capture_times.append((row[1], timestamps[3] / 1e9))
                        if self.saved % 100 == 0:
                            labels_file.flush()
                            timing_file.flush()
                    finally:
                        self.q.task_done()
        except BaseException as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"recording writer failed: {self.error}") from self.error

    def finish(self, timeout=5):
        self.stop.set()
        self.join(timeout)
        if self.is_alive():
            raise RuntimeError("recording writer did not stop before shutdown deadline; no meta.json published")
        self.check()


# ---------------------------------------------------------------- record

def make_record_row(frame_idx, segment, retrieved_ns, origin_ns, wheel_sample, tele_sample):
    """Keep the original per-frame label precision and retrieval-time semantics."""
    wt_ns, steer, brake_raw, gas_raw = wheel_sample
    brake = (32767 - brake_raw) / 65535
    gas = (32767 - gas_raw) / 65535
    return [frame_idx, segment, f"{(retrieved_ns - origin_ns) / 1e9:.4f}",
            steer, f"{steer / STEER_UNITS_PER_DEG:.2f}", f"{brake:.4f}", f"{gas:.4f}",
            f"{(retrieved_ns - wt_ns) / 1e6:.1f}",
            f"{tele_sample[2]:.3f}" if tele_sample else "", tele_sample[1] if tele_sample else "",
            tele_sample[3] if tele_sample else "",
            f"{(retrieved_ns - tele_sample[0]) / 1e6:.1f}" if tele_sample else "",
            f"{tele_sample[4]:.3f}" if tele_sample else "",
            f"{tele_sample[5]:.1f}" if tele_sample else "",
            f"{tele_sample[6]:.4f}" if tele_sample else "",
            tele_sample[7] if tele_sample else "", tele_sample[8] if tele_sample else "",
            tele_sample[9] if tele_sample else "", tele_sample[10] if tele_sample else "",
            tele_sample[11] if tele_sample else ""]


def measured_capture_report(capture, writer, duration_ns, fps, dropped, inactive):
    """Observed throughput, counting only fresh grabs and successfully saved JPEGs."""
    duration = max(0, duration_ns) / 1e9
    fresh = len(capture.times)
    return {
        "duration_s": round(duration, 6), "capture_attempts": capture.attempts,
        "fresh_frames": fresh, "no_new_frame_polls": capture.no_frame_polls,
        "saved_frames": writer.saved, "dropped_queue_frames": dropped,
        "inactive_fresh_frames": inactive, "fresh_not_saved": fresh - writer.saved,
        "captured_fps": round(fresh / duration, 3) if duration else 0.0,
        "saved_fps": round(writer.saved / duration, 3) if duration else 0.0,
        "rate_denominator": "whole acquisition duration including inactive/paused periods; excludes writer drain",
        "fresh_capture_gaps": gap_report([(segment, ns / 1e9) for segment, ns in capture.times], fps),
        "saved_capture_gaps": gap_report(writer.capture_times, fps),
    }


def cmd_record(args):
    args.fps = capture_fps(args.fps)
    if not 1 <= args.quality <= 100:
        raise ValueError("JPEG quality must be within [1, 100]")
    if not math.isfinite(args.drop_seconds) or args.drop_seconds < 0:
        raise ValueError("drop_seconds must be finite and nonnegative")
    if args.rewind_button is not None and args.rewind_button < 0:
        raise ValueError("rewind_button must be a nonnegative SDL index")
    # Fingerprint the producing file once, before device acquisition; no Git checkout required.
    with open(__file__, "rb") as source_file:
        producer_sha256 = hashlib.sha256(source_file.read()).hexdigest()
    cfg = load_config()
    if not cfg["crop"]:
        sys.exit("No crop yet. Run: python record.py setup")
    size = save_size(cfg)
    masks = masks_in_crop(cfg)
    # Keep one fresh source image for both the road crop and the synchronization HUD patch.
    hud_box = cfg.get("hud_box")
    boxes = [cfg["crop"]] + ([hud_box] if hud_box else [])
    region = (min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes))
    cx0, cy0, cx1, cy1 = (v - o for v, o in zip(cfg["crop"], region[:2] * 2))
    hx0, hy0, hx1, hy1 = (v - o for v, o in zip(hud_box, region[:2] * 2)) if hud_box else (0, 0, 0, 0)
    import dxcam

    wheel = tele = writer = cam = capture = None
    session_dir = None
    frame_idx, segment, recording = 0, -1, False
    dropped, inactive = 0, 0
    discarded = discarded_at_stop = rewinds = takeovers = pause_discards = 0
    was_rewinding = in_takeover = False
    ratios = deque(maxlen=900)
    mismatch_run = match_run = 0
    pending = deque()

    def flush_pending():
        nonlocal dropped
        _, _, img, hud, row, timestamps = pending.popleft()
        try:
            writer.q.put_nowait((img, hud, row, timestamps))
        except queue.Full:
            dropped += 1

    t_start_ns = t_end_ns = None
    last_status = -math.inf
    failure = None
    cleanup_errors = []
    try:
        wheel = WheelReader(args.vjoy)
        wheel.start()
        wheel.ready.wait(5)
        if wheel.error or wheel.latest_ns is None:
            raise RuntimeError(f"Wheel error: {wheel.error or 'no reading within 5 s'}")
        if not args.no_telemetry:
            try:
                tele = TelemetryReader(args.port)
                tele.start()
            except OSError as error:
                raise RuntimeError(f"Can't listen on 127.0.0.1:{args.port} ({error}). Close other telemetry "
                                   "receivers, or use --no-telemetry.") from error
        session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_dir = os.path.join(DATA_DIR, "recordings", session)
        os.makedirs(session_dir, exist_ok=False)
        writer = Writer(session_dir, args.quality)
        writer.start()
        candidate = dxcam.create(output_idx=cfg["monitor"], output_color="BGR")
        if candidate.is_capturing:
            raise RuntimeError("DXcam output is already capturing; close the other capture owner")
        cam = candidate  # Only release a camera we acquired while stopped.
        capture = FreshCapture(cam, region, args.fps)
        print(f"Recording to {session_dir}\n"
              f"crop={cfg['crop']} masks={len(cfg['masks'])} saved={size[0]}x{size[1]} "
              f"fresh fps target={args.fps:g} vjoy={'on' if args.vjoy else 'off'} "
              f"telemetry={'on' if tele else 'off'}\n"
              + ("Frames are only kept while Forza reports IsRaceOn=1 (driving, not paused/menus).\n" if tele else "")
              + "Ctrl+C to stop.\n")
        t_start_ns = time.perf_counter_ns()
        while True:
            writer.check()
            if wheel.error:
                raise RuntimeError("Wheel thread died: " + wheel.error)
            if tele and tele.error:
                raise RuntimeError("Telemetry thread died: " + tele.error)
            img, capture_ns, retrieved_ns = capture.grab()
            t = retrieved_ns / 1e9  # Still post-retrieval time, never relabelled as capture start.
            ws = wheel.latest_ns
            if wheel.error or ws is None:
                raise RuntimeError("Wheel thread died: " + (wheel.error or "wheel sample unavailable"))
            wt_ns, steer, brake_raw, gas_raw = ws
            tl = tele.latest_ns if tele else None
            tele_fresh = tl is not None and 0 <= retrieved_ns - tl[0] < 500_000_000
            wheel_fresh = 0 <= retrieved_ns - wt_ns < 100_000_000
            if not wheel.live:
                active, state = False, "turn wheel + press both pedals once"
            elif not wheel_fresh:
                active, state = False, "waiting for fresh wheel"
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

            if img is not None and tele_fresh and wheel.live:
                deg, ts = steer / STEER_UNITS_PER_DEG, tl[3]
                if recording and not in_takeover and 5 < abs(deg) < 60 and abs(ts) < 120:
                    ratios.append(ts / deg)
                if len(ratios) >= 100:
                    k = float(np.median(ratios))
                    mismatch = abs(ts - max(-127.0, min(127.0, k * deg))) > 15
                    mismatch_run = mismatch_run + 1 if mismatch else 0
                    match_run = 0 if mismatch else match_run + 1
                    if not in_takeover and mismatch_run >= 6:
                        in_takeover = True
                        takeovers += 1
                        while pending and pending[-1][0] >= t - 0.5:   # the run-up to detection
                            pending.pop()
                            discarded += 1
                    elif in_takeover and match_run >= 30:
                        in_takeover = False
            if in_takeover:
                active, state = False, "game steering, not you"

            if recording and not active and pending and not in_takeover and not args.keep_before_pause:
                # Any pause may be a reset the watchdog can't see (e.g. the car put back on the road,
                # with neither race clock nor distance moving back): drop the held-back frames too.
                pause_discards += 1
                discarded += len(pending)
                pending.clear()
            if not active:
                recording = False
            if img is not None:
                if not active:
                    inactive += 1
                else:
                    next_segment = segment if recording else segment + 1
                    row = make_record_row(frame_idx, next_segment, retrieved_ns, t_start_ns, ws, tl)
                    timestamps = [frame_idx, next_segment, len(capture.times) - 1, capture_ns, retrieved_ns,
                                  wt_ns, tl[0] if tl else ""]
                    crop_img = img[cy0:cy1, cx0:cx1]
                    hud = img[hy0:hy1, hx0:hx1].copy() if hud_box else None
                    if hud is not None and not hud.size:
                        raise ValueError("HUD box is outside the captured image")
                    pending.append((t, next_segment, process(crop_img, masks, size), hud, row, timestamps))
                    segment, recording = next_segment, True
                    frame_idx += 1
                    # Retain producer rewind protection: only flush while driving.
                    # A zero holdback explicitly disables buffering; stopping still discards pending frames.
                    while pending and t - pending[0][0] >= args.drop_seconds:
                        flush_pending()
            if t - last_status > .5:
                last_status = t
                speed = f"{tl[2] * 3.6:5.0f} km/h" if tl else "   -- km/h"
                elapsed = max((retrieved_ns - t_start_ns) / 1e9, .001)
                print(f"\r[{state:22s}] frames={frame_idx:6d} seg={max(segment, 0):3d} "
                      f"steer={steer / STEER_UNITS_PER_DEG:+7.1f}deg gas={(32767 - gas_raw) / 65535:4.2f} "
                      f"brake={(32767 - brake_raw) / 65535:4.2f} {speed} dropped={dropped} "
                      f"fresh={len(capture.times) / elapsed:.1f}/s   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except BaseException as error:
        failure = error
    finally:
        t_end_ns = time.perf_counter_ns()
        # Deliberate producer behavior: stopping after a mistake discards the holdback window.
        discarded_at_stop = len(pending)
        pending.clear()
        for reader in (wheel, tele):
            if reader is not None:
                reader.stop.set()
        if cam is not None:
            try:
                cam.release()
            except BaseException as error:
                cleanup_errors.append(f"camera release: {error}")
        if writer is not None:
            try:
                writer.finish()
            except BaseException as error:
                cleanup_errors.append(str(error))
        for name, reader in (("wheel", wheel), ("telemetry", tele)):
            if reader is not None:
                try:
                    reader.join(2)
                    if reader.is_alive():
                        raise RuntimeError(f"{name} thread did not stop before shutdown deadline")
                    if reader.error:
                        raise RuntimeError(f"{name}: {reader.error}")
                except BaseException as error:
                    cleanup_errors.append(str(error))
    if failure is not None or cleanup_errors:
        detail = "; ".join(cleanup_errors)
        raise RuntimeError(f"Recording incomplete; no meta.json published. {failure or ''} {detail}") from failure
    if capture is None or t_start_ns is None:
        print("Stopped before acquisition began; no completed recording published.")
        return
    if writer.saved + dropped + discarded + discarded_at_stop != frame_idx:
        raise RuntimeError("Saved/discarded frame count mismatch; no meta.json published")
    report = gap_report(writer.times, args.fps)
    measured = measured_capture_report(capture, writer, t_end_ns - t_start_ns,
                                       args.fps, dropped, inactive)
    meta = {"session": session, "config": cfg, "saved_size": size, "fps_target": args.fps,
            "producer_schema": PRODUCER_SCHEMA, "producer_sha256": producer_sha256,
            "frames": writer.saved, "segments": segment + 1, "dropped": dropped, "vjoy": args.vjoy,
            "accepted_frame_count": frame_idx, "frame_ids_may_have_gaps": True,
            "rewinds": rewinds, "takeovers": takeovers, "pause_discards": pause_discards,
            "discard_on_pause": not args.keep_before_pause,
            "discarded_by_rewind_or_takeover": discarded, "discarded_at_stop": discarded_at_stop,
            "drop_seconds": args.drop_seconds, "rewind_button": args.rewind_button,
            "auto_rewind": not args.no_auto_rewind,
            "telemetry": tele is not None, "steer_units_per_deg": STEER_UNITS_PER_DEG,
            "pedals": "0 = released, 1 = floored", "gaps": report, "completed": True,
            "jpeg_quality": args.quality, "measured_capture": measured,
            "capture_provenance": {
                "method": "dxcam 0.3 one-shot grab(copy=True, new_frame_only=True); no ring buffer/video mode",
                "source": "https://github.com/ra1nty/DXcam/blob/v0.3.0/dxcam/dxcam.py",
                "clock": "perf_counter_ns", "origin_ns": t_start_ns,
                "timing_file": "capture_timing.csv", "timing_columns": TIMING_COLUMNS,
                "labels_t": "relative perf_counter after image retrieval, rounded to four decimals (legacy semantics)",
                "capture_start_ns": "host time immediately before grab; not a GPU presentation/game render timestamp",
                "retrieved_ns": "host time immediately after grab returned",
                "wheel_sample_ns": "host polling completion time; not a USB hardware report timestamp",
                "telemetry_sample_ns": "host UDP receive time; not game simulation time",
                "alignment": "latest input samples read after retrieval; legacy importer still uses rounded row labels",
                "cached_frames_reused": False, "actual_game_render_time_known": False,
            }}
    temporary = os.path.join(session_dir, "meta.json.tmp")
    with open(temporary, "x") as handle:
        json.dump(meta, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, os.path.join(session_dir, "meta.json"))
    print(f"Saved {writer.saved} frames in {segment + 1} segment(s) to {session_dir} (queue dropped {dropped})")
    print(f"Rewinds: {rewinds}; game takeovers: {takeovers}; pauses: {pause_discards}; "
          f"intentionally discarded {discarded} frames, "
          f"plus {discarded_at_stop} pending frames at stop.")
    print(f"Measured fresh capture {measured['captured_fps']:.1f} fps; saved {measured['saved_fps']:.1f} fps "
          f"over {measured['duration_s']:.1f}s (includes pauses, excludes writer drain).")
    if report:
        print(f"Frame gaps (ms): median {report['median_ms']}, p99 {report['p99_ms']}, "
              f"max {report['max_ms']}, {report['over_2x']} gap(s) > {2000 / args.fps:.0f} ms")


def cmd_wheel(args):
    """Show what the recorder reads from the TMX, and the range each axis covered."""
    wheel = WheelReader(False)
    wheel.start()
    lo, hi, seen = [32767] * 3, [-32768] * 3, set()
    try:
        wheel.ready.wait(5)
        if wheel.error or wheel.latest is None:
            raise RuntimeError(f"Wheel error: {wheel.error or 'no reading within 5 s'}")
        print("Turn the wheel fully both ways, press each pedal fully, press any buttons.\n"
              "Works with Forza focused too (check the ranges afterwards). Ctrl+C to stop.\n")
        while True:
            sample = wheel.latest
            if wheel.error or sample is None:
                raise RuntimeError("Wheel thread died: " + (wheel.error or "wheel sample unavailable"))
            _, *raw = sample
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
    finally:
        wheel.stop.set()
        wheel.join(2)
        if wheel.is_alive():
            raise RuntimeError("wheel thread did not stop before shutdown deadline")
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

def argument_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "preview"):
        sp = sub.add_parser(name)
        sp.add_argument("--image", help="use a saved screenshot instead of grabbing the screen")
        sp.add_argument("--delay", type=int, default=5, help="seconds before the screen grab")
    rp = sub.add_parser("record")
    rp.add_argument("--fps", type=capture_fps, default=30.0, help="fresh-frame target, 0 < fps <= 240 (default 30)")
    rp.add_argument("--quality", type=int, default=90, help="JPEG quality")
    rp.add_argument("--vjoy", action="store_true", help="pass TMX steering/pedals through to vJoy")
    rp.add_argument("--no-telemetry", action="store_true")
    rp.add_argument("--port", type=int, default=9999)
    rp.add_argument("--rewind-button", type=int, default=None,
                    help="TMX button bound to Rewind in Forza (number from: python utils\\test.py wheel)")
    rp.add_argument("--drop-seconds", type=float, default=10.0,
                    help="seconds of frames held back and thrown away on a rewind/reset/pause (default 10)")
    rp.add_argument("--keep-before-pause", action="store_true",
                    help="don't drop the held-back frames on a plain pause (only on detected rewinds)")
    rp.add_argument("--no-auto-rewind", action="store_true",
                    help="turn off the race-clock rewind watchdog (button only)")
    sub.add_parser("wheel", help="live TMX readout as the recorder sees it (RawInput)")
    return p


def main(argv=None):
    args = argument_parser().parse_args(argv)
    {"setup": cmd_setup, "preview": cmd_preview, "record": cmd_record, "wheel": cmd_wheel}[args.cmd](args)


if __name__ == "__main__":
    main()
