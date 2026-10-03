"""Best-effort console progress outside the motor loop."""

import threading
from collections import deque


class LiveRates:
    """Bounded two-second event rates for display, not a capture-rate guarantee."""

    def __init__(self, window_ns=2_000_000_000):
        if type(window_ns) is not int or window_ns <= 0:
            raise ValueError("window_ns must be a positive integer")
        self.window_ns = window_ns
        self._start_ns = None
        self._last_ns = None
        self._events = {name: deque(maxlen=4096) for name in ("capture", "policy", "control")}
        self._last_frame = None
        self._last_prediction = None
        self._has_camera = False

    def update(self, timestamp_ns, *, frame=None, prediction_id=None, has_camera=False):
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("timestamp_ns must be nonnegative integer nanoseconds")
        if self._last_ns is not None and timestamp_ns <= self._last_ns:
            return
        self._start_ns = timestamp_ns if self._start_ns is None else self._start_ns
        self._last_ns = timestamp_ns
        self._has_camera = has_camera
        self._events["control"].append(timestamp_ns)
        if has_camera and frame is not None:
            identity = (frame.frame_id, frame.timestamp_ns)
            if identity != self._last_frame and 0 <= frame.timestamp_ns <= timestamp_ns:
                self._events["capture"].append(frame.timestamp_ns)
                self._last_frame = identity
        if prediction_id is not None and prediction_id != self._last_prediction:
            if type(prediction_id) is int and 0 <= prediction_id <= timestamp_ns:
                self._events["policy"].append(prediction_id)
                self._last_prediction = prediction_id
        for samples in self._events.values():
            while samples and samples[0] <= timestamp_ns - self.window_ns:
                samples.popleft()

    def summary(self):
        elapsed = 0 if self._start_ns is None else min(self.window_ns, self._last_ns - self._start_ns)
        # A short warm-up avoids presenting an unhelpful single-tick spike.
        ready = elapsed >= min(250_000_000, self.window_ns)
        seconds = elapsed / 1e9
        values = {name: len(samples) / seconds if ready else None for name, samples in self._events.items()}
        return {"capture_fps": values["capture"] if self._has_camera else None,
                "policy_fps": values["policy"], "control_hz": values["control"],
                "window_seconds": seconds, "method": "observed unique events in a rolling window"}


class ProgressReporter:
    def __init__(self, period=0.5):
        self._period = period
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._value = None
        self._thread = threading.Thread(target=self._run, name="control-progress", daemon=True)

    def start(self):
        self._thread.start()

    def publish(self, value):
        with self._lock:
            self._value = dict(value)

    def _run(self):
        while not self._stop.wait(self._period):
            with self._lock:
                value = self._value
            if value is not None:
                try:
                    predicted = value.get("predicted_angle_deg")
                    predicted = "--" if predicted is None else f"{predicted:+.1f}"
                    print(f"{value['mode']} [{value['reason']}] "
                          f"prediction={predicted}deg target={value['target_angle_deg']:+.1f}deg "
                          f"wheel={value['actual_angle_deg']:+.1f}deg input={value['input_status']}",
                          flush=True)
                except (BrokenPipeError, OSError):
                    return

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=0.5)
