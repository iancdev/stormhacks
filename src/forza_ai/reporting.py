"""Best-effort console progress outside the motor loop."""

import threading


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
