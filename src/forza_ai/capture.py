"""Fresh RGB road crops from DXcam, without capture work on the motor thread.

DXcam 0.3.0's one-shot ``grab(new_frame_only=True)`` returns None when no new
desktop frame is available. Its threaded/ring-buffer mode ignores that flag, so
this adapter owns a separate thread and never calls ``camera.start()``.
"""

from __future__ import annotations

import math
import sys
import threading
import time

import numpy as np

from .contracts import CapturedFrame


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _create_camera(region, output_idx):
    if sys.platform != "win32":
        raise RuntimeError("DXCamCapture requires Windows and the hardware dependencies")
    import dxcam

    return dxcam.create(region=region, output_idx=output_idx, output_color="RGB", backend="dxgi")


class DXCamCapture:
    """Capture an explicit road crop in output-local absolute pixel coordinates.

    ``start`` waits for camera creation and reports startup errors. ``latest``
    only reads the most recent owned frame; it never calls the driver or changes
    a timestamp. A frame's host timestamp precedes its grab, conservatively
    including driver/copy latency. It is not a measured game presentation time.

    This instance must exclusively own its DXcam output. Only the capture thread
    releases the camera, including after errors. A hung driver makes ``close``
    report a timeout instead of racing release against an in-flight grab.
    """

    def __init__(self, region, fps=30, output_idx=0):
        if not isinstance(region, (tuple, list)) or len(region) != 4:
            raise ValueError("region must be (left, top, right, bottom) pixel coordinates")
        for coordinate in region:
            _nonnegative_int(coordinate, "region coordinate")
        left, top, right, bottom = region
        if right <= left or bottom <= top:
            raise ValueError("region must have positive width and height")
        if isinstance(fps, bool) or not math.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("fps must be finite and in (0, 240]")
        _nonnegative_int(output_idx, "output_idx")
        self.region = tuple(region)
        self.fps = fps
        self.output_idx = output_idx
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._frame = None
        self._error = None
        self._cleanup_error = None
        self._running = False

    def start(self):
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("capture thread is still stopping; close it before restarting")
                with self._lock:
                    if self._error is not None:
                        raise RuntimeError("screen capture failed") from self._error
                return
            self._stop.clear()
            self._ready.clear()
            with self._lock:
                self._frame = None
                self._error = None
                self._cleanup_error = None
                self._running = False
            self._thread = threading.Thread(target=self._capture, name="road-capture", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                raise
            if not self._ready.wait(timeout=10.0):
                self._stop.set()
                raise RuntimeError("screen capture startup did not finish within ten seconds")
            with self._lock:
                if self._error is not None:
                    raise RuntimeError("screen capture startup failed") from self._error

    def latest(self, now_ns=None, max_age_ns=250_000_000):
        _nonnegative_int(max_age_ns, "max_age_ns")
        with self._lock:
            if self._error is not None:
                raise RuntimeError("screen capture failed") from self._error
            frame = self._frame if self._running else None
            if now_ns is None:
                now_ns = time.monotonic_ns()
        _nonnegative_int(now_ns, "now_ns")
        if frame is None or not 0 <= now_ns - frame.timestamp_ns <= max_age_ns:
            return None
        return frame

    def _capture(self):
        camera = None
        try:
            candidate = _create_camera(self.region, self.output_idx)
            if candidate is None:
                raise RuntimeError("DXcam did not return a camera")
            # DXcam's factory can return a camera owned by other code. Do not
            # stop/release such an active camera or reuse its cached frames.
            if candidate.is_capturing:
                raise RuntimeError("DXcam output is already capturing; exclusive ownership is required")
            camera = candidate
            with self._lock:
                self._running = not self._stop.is_set()
            self._ready.set()
            frame_id = 0
            expected_shape = (self.region[3] - self.region[1], self.region[2] - self.region[0], 3)
            while not self._stop.is_set():
                if camera.is_capturing:
                    raise RuntimeError("DXcam ring-buffer capture must remain stopped")
                started_ns = time.monotonic_ns()
                pixels = camera.grab(region=self.region, copy=False, new_frame_only=True)
                if pixels is not None:
                    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.shape != expected_shape:
                        raise ValueError("capture must return uint8 HWC RGB matching the requested road crop")
                    owned = np.array(pixels, copy=True, order="C")
                    owned.flags.writeable = False
                    frame = CapturedFrame(frame_id, started_ns, owned)
                    with self._lock:
                        if not self._stop.is_set():
                            self._frame = frame
                    frame_id += 1
                elapsed = (time.monotonic_ns() - started_ns) / 1e9
                self._stop.wait(max(0.0, 1.0 / self.fps - elapsed))
        except BaseException as error:
            with self._lock:
                self._error = error
        finally:
            # Keep the resource on its owning thread until grab has returned.
            if camera is not None:
                try:
                    camera.release()
                except BaseException as error:
                    with self._lock:
                        self._cleanup_error = error
                        if self._error is None:
                            self._error = error
            with self._lock:
                self._running = False
                self._frame = None
            self._ready.set()

    def close(self):
        with self._lifecycle_lock:
            self._stop.set()
            with self._lock:
                self._running = False
                self._frame = None
            if self._thread is None:
                return
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("capture thread did not stop; camera remains owned by that thread")
            self._thread = None
            with self._lock:
                if self._cleanup_error is not None:
                    raise RuntimeError("screen capture cleanup failed") from self._cleanup_error

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
