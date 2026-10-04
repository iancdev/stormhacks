"""Bounded, best-effort inference diagnostics. Never serialize exception text or payloads."""

from datetime import datetime, timezone
import json
import math
import queue
import threading


_FIELDS = frozenset((
    'connection_id', 'peer_ip', 'peer_port', 'protocol_version', 'phase', 'cause',
    'authenticated', 'requests', 'responses', 'received_bytes', 'duration_ms',
    'response_hz', 'completed_model_calls', 'model_elapsed_ms', 'model_mean_ms', 'model_max_ms', 'request_ms',
    'timeout_ms', 'log_dropped',
))


class InferenceLog:
    """Only the daemon writer performs formatting/output; producers never wait for it.

    Overflow drops diagnostics, never inference work. A failed or blocked output
    sink cannot crash the server or delay its request thread. Close is bounded;
    the last records may be lost if the sink remains blocked.
    """

    def __init__(self, sink=None, capacity=256):
        self._sink = sink or (lambda line: print(line, flush=True))
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self.dropped = 0
        self._thread = threading.Thread(target=self._write, name='inference-log', daemon=True)
        self._thread.start()

    def emit(self, event, **fields):
        try:
            record = {'event': event, 'utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds')}
            for key, value in fields.items():
                if key in _FIELDS and (value is None or isinstance(value, (bool, int))
                        or isinstance(value, str) and len(value) <= 64
                        or isinstance(value, float) and math.isfinite(value)):
                    record[key] = value
            self._queue.put_nowait(record)
        except Exception:
            self.dropped += 1

    def _write(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                record = self._queue.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                record['log_dropped'] = self.dropped
                self._sink(json.dumps(record, separators=(',', ':'), allow_nan=False))
            except Exception:
                self.dropped += 1
            finally:
                self._queue.task_done()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=.25)
