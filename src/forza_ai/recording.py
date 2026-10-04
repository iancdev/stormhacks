"""Bounded asynchronous session recording; no hardware or model dependencies.

The control loop publishes immutable observations here. All image encoding and
CSV writes happen on the writer thread. A failed or overloaded recording remains
explicitly incomplete, so the training loader cannot silently use partial data.
"""

from contextlib import ExitStack
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import threading
import uuid

import numpy as np
from PIL import Image

from .contracts import CapturedFrame, ControlMode, VehicleState, WheelState


@dataclass(frozen=True)
class _Item:
    wheel: WheelState
    vehicle: VehicleState | None
    frame: CapturedFrame | None
    mode: ControlMode | str
    expert: bool
    reason: str
    # (generated_ns, observation_ns, predicted_deg, applied_target_deg, throttle, brake) or None
    prediction: tuple | None = None


class SessionRecorder:
    """Write a new stream-v1 session, with expert labels explicitly opted in.

    ``destination`` is the session directory itself and must not exist, even if
    empty. ``start`` and ``close`` belong outside the motor loop. ``submit`` uses
    ``put_nowait`` and returns False after overload/failure or during shutdown.
    The caller must not mutate a frame's RGB array after publication (the capture
    module publishes owned, read-only arrays). No image copy occurs on submit.

    Only explicitly expert manual/takeover states become training labels; fault,
    assistance, unmarked manual, and unmarked takeover rows are written as assist.
    The original mode, eligibility, and reason remain in events.csv. This permits
    the v1 loader to reject mode-boundary interpolation without fabricated rows.
    Completion means the archive closed successfully, not that expert examples
    exist: an all-assist archive can have zero trainable samples. Inspect it with
    the session validator before adding it to a training split.
    """

    CLOSE_TIMEOUT_S = 5.0
    _FIELDS = {
        "frames": ["frame_id", "image_path", "capture_time_ns"],
        "wheel": ["timestamp_ns", "angle_deg", "throttle", "brake", "control_mode"],
        "telemetry": ["timestamp_ns", "speed_mps", "is_race_on", "game_timestamp_ms",
                      "rpm", "steering_input"],
        "events": ["timestamp_ns", "control_mode", "training_mode", "expert", "reason"],
        # For DAgger analysis only (the training loader never reads it): every AI prediction,
        # also while the human drives, so corrections can be compared with what the AI wanted.
        "predictions": ["generated_time_ns", "observation_time_ns", "predicted_angle_deg",
                        "applied_target_deg", "predicted_throttle", "predicted_brake", "control_mode"],
    }

    def __init__(self, destination, session_id=None, metadata=None, queue_size=256):
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a nonempty string")
        self.destination = Path(destination).absolute()
        self.session_id = session_id or str(uuid.uuid4())
        # Copy nested caller metadata and fail before creating files on bad JSON.
        self._metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        if not isinstance(self._metadata, dict):
            raise ValueError("metadata must be an object")
        required = {
            "schema_version": 1, "session_id": self.session_id, "clock": "monotonic_ns",
            "wheel_rotation_deg": 900, "image_stage": "road_crop", "completed": False,
        }
        for name, value in required.items():
            if name in self._metadata and self._metadata[name] != value:
                raise ValueError(f"metadata {name} must be {value!r}")
        self._metadata.update(required)
        self._metadata["recording_tool"] = "forza_ai.recording.SessionRecorder"
        self._metadata["expert_basis"] = "explicit runtime manual demonstration or human takeover"
        self._queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._started = False
        self._accepting = False
        self._closed = False
        self._error = None
        self._counters = dict(submitted=0, wheel_samples=0, telemetry_samples=0,
                              frames_written=0, frames_skipped=0, dropped=0, events=0)
        self._completed = False
        self._last_wheel = None
        self._last_vehicle = None
        self._last_frame = None
        self._last_event = None
        self._boundary = None
        self._eligible_since = None

    @property
    def stats(self):
        with self._lock:
            return {**self._counters, "queue_depth": self._queue.qsize(),
                    "error": None if self._error is None else str(self._error),
                    "completed": self._completed,
                    "running": bool(self._thread and self._thread.is_alive())}

    def check(self):
        """Raise if data integrity was lost; checking never performs disk I/O."""
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"session recording incomplete: {error}") from error

    def _fail(self, error):
        with self._lock:
            if self._error is None:
                self._error = error
            self._accepting = False

    def _count(self, key):
        with self._lock:
            self._counters[key] += 1

    def _write_metadata(self, completed):
        metadata = {**self._metadata, "completed": bool(completed),
                    "recording_stats": self.stats}
        temporary = self.destination / f".metadata-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
            temporary.replace(self.destination / "metadata.json")
        finally:
            temporary.unlink(missing_ok=True)

    def start(self):
        if self._started or self._closed:
            raise RuntimeError("recorder can only be started once")
        if self.destination.exists() or self.destination.is_symlink():
            raise ValueError("recording destination must not exist")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.mkdir()  # Exclusive creation closes the existence-check race.
        self._started = True
        try:
            (self.destination / "images").mkdir()
            self._write_metadata(False)
            self._thread = threading.Thread(target=self._run, name="session-recorder", daemon=True)
            with self._lock:
                self._accepting = True
            self._thread.start()
        except BaseException as error:
            self._fail(error)
            raise
        return self

    def submit(self, wheel: WheelState, vehicle: VehicleState | None,
               frame: CapturedFrame | None, mode: ControlMode | str,
               *, expert: bool = False, reason: str = "", prediction: tuple | None = None) -> bool:
        """Enqueue independent source samples without changing their timestamps."""
        item = _Item(wheel, vehicle, frame, mode, expert, reason, prediction)
        with self._lock:
            if not self._accepting:
                return False
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._counters["dropped"] += 1
                self._error = RuntimeError("recording queue full; a control sample was dropped")
                self._accepting = False
                return False
            self._counters["submitted"] += 1
        return True

    @staticmethod
    def _timestamp(value, name):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be nonnegative integer nanoseconds")

    @staticmethod
    def _number(value, name, low, high):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"invalid {name}: {value}")

    def _write_item(self, item, writers):
        wheel = item.wheel
        self._timestamp(wheel.timestamp_ns, "wheel timestamp")
        self._number(wheel.angle_deg, "wheel angle", -450, 450)
        self._number(wheel.throttle, "throttle", 0, 1)
        self._number(wheel.brake, "brake", 0, 1)
        mode = ControlMode(item.mode).value
        if type(item.expert) is not bool or not isinstance(item.reason, str):
            raise ValueError("expert must be a bool and reason must be a string")
        eligible = item.expert and wheel.connected and mode in {"manual", "takeover"}
        training_mode = mode if eligible else "assist"
        row = (wheel.timestamp_ns, wheel.angle_deg, wheel.throttle, wheel.brake, training_mode)
        if self._last_wheel is not None:
            if row[0] < self._last_wheel[0]:
                raise ValueError("wheel timestamp reversed")
            if row[0] == self._last_wheel[0] and row != self._last_wheel:
                raise ValueError("wheel values or training mode changed at the same timestamp")
        boundary = (mode, eligible)
        if boundary != self._boundary:
            # Do not let cached frames from before explicit human control acquire
            # expert labels merely because they arrived after the mode switch.
            self._eligible_since = wheel.timestamp_ns if eligible else None
            self._boundary = boundary
        event = (mode, training_mode, bool(eligible), item.reason)
        if event != self._last_event:
            writers["events"].writerow((wheel.timestamp_ns, mode, training_mode,
                                        int(eligible), item.reason))
            self._last_event = event
            self._count("events")
        if row != self._last_wheel:
            writers["wheel"].writerow(row)
            self._last_wheel = row
            self._count("wheel_samples")

        vehicle = item.vehicle
        if vehicle is not None:
            self._timestamp(vehicle.timestamp_ns, "telemetry timestamp")
            self._number(vehicle.speed_mps, "speed", 0, float("inf"))
            self._number(vehicle.rpm, "rpm", 0, float("inf"))
            self._number(vehicle.steering_input, "steering input", -128, 127)
            self._timestamp(vehicle.game_timestamp_ms, "game timestamp_ms")
            row = (vehicle.timestamp_ns, vehicle.speed_mps, int(bool(vehicle.is_race_on)),
                   vehicle.game_timestamp_ms, vehicle.rpm, vehicle.steering_input)
            if self._last_vehicle is not None:
                if row[0] < self._last_vehicle[0]:
                    raise ValueError("telemetry timestamp reversed")
                if row[0] == self._last_vehicle[0] and row != self._last_vehicle:
                    raise ValueError("telemetry values changed at the same timestamp")
            if row != self._last_vehicle:
                writers["telemetry"].writerow(row)
                self._last_vehicle = row
                self._count("telemetry_samples")

        if item.frame is not None:
            self._write_frame(item.frame, writers["frames"])

        if item.prediction is not None and item.prediction[0] != getattr(self, "_last_prediction_ns", None):
            generated, observed, angle, applied, throttle, brake = item.prediction
            self._timestamp(generated, "prediction timestamp")
            self._timestamp(observed, "prediction observation timestamp")
            writers["predictions"].writerow((generated, observed, round(float(angle), 3), round(float(applied), 3),
                                             "" if throttle is None else round(float(throttle), 4),
                                             "" if brake is None else round(float(brake), 4), mode))
            self._last_prediction_ns = generated

    def _write_frame(self, frame, writer):
        self._timestamp(frame.timestamp_ns, "frame timestamp")
        if type(frame.frame_id) is not int or frame.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        identity = (frame.frame_id, frame.timestamp_ns)
        if identity == self._last_frame:
            self._count("frames_skipped")
            return
        if self._last_frame is not None and (
            identity[0] <= self._last_frame[0] or identity[1] <= self._last_frame[1]
        ):
            raise ValueError("frame identity/timestamp reversed or reused inconsistently")
        self._last_frame = identity
        if self._eligible_since is not None and frame.timestamp_ns < self._eligible_since:
            self._count("frames_skipped")
            return
        pixels = frame.rgb
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                or pixels.ndim != 3 or pixels.shape[2] != 3
                or not pixels.shape[0] or not pixels.shape[1]):
            raise ValueError("recorded frame must be a nonempty uint8 HWC RGB array")
        relative = f"images/{frame.frame_id:010d}.png"
        # The worker is the only writer, and the destination is exclusively new.
        with (self.destination / relative).open("xb") as handle:
            Image.fromarray(pixels).save(handle, format="PNG")
        writer.writerow((frame.frame_id, relative, frame.timestamp_ns))
        self._count("frames_written")

    def _run(self):
        try:
            with ExitStack() as files:
                writers = {}
                for name, columns in self._FIELDS.items():
                    handle = files.enter_context((self.destination / f"{name}.csv").open(
                        "x", newline="", encoding="utf-8"))
                    writers[name] = csv.writer(handle)
                    writers[name].writerow(columns)
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        item = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        self._write_item(item, writers)
                    finally:
                        self._queue.task_done()
        except BaseException as error:
            self._fail(error)

    def close(self, completed=True):
        """Drain/close within a bounded join and atomically publish completion.

        Errors, queue drops, or a writer that does not exit leave completed=false
        and raise. Calling close(completed=False) is appropriate after a runtime
        exception; it preserves the recording for inspection, not training.
        """
        if type(completed) is not bool:
            raise ValueError("completed must be a bool")
        if not self._started:
            raise RuntimeError("recorder has not been started")
        if self._closed:
            self.check()
            return
        with self._lock:
            self._accepting = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.CLOSE_TIMEOUT_S)
            if self._thread.is_alive():
                self._fail(TimeoutError("recorder writer did not stop before shutdown deadline"))
        stats = self.stats
        if completed and not stats["error"] and not all(
            stats[key] for key in ("wheel_samples", "telemetry_samples", "frames_written")
        ):
            self._fail(ValueError("recording is missing wheel, telemetry, or image samples"))
        with self._lock:
            self._completed = completed and self._error is None
        try:
            self._write_metadata(self._completed)
        except BaseException as error:
            self._fail(error)
            with self._lock:
                self._completed = False
            # The initial metadata remains incomplete if final replacement failed.
        self._closed = True
        self.check()
