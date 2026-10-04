"""Nonblocking policy exchange with persistent inference-failure notifications."""

import math
import threading
import time

from forza_ai.contracts import ObservationUnavailable, SteeringCommand, DrivingPrediction, ModelObservation


class PolicyWorker:
    """Latest-observation/latest-command exchange, no queue of old predictions.

    CPU model inference can replace predict(), but must provide the original
    image timestamp in its observation. Generating a command never refreshes
    the observation timestamp. Invalidated input also discards any in-flight
    prediction, so a slow model cannot republish a pre-takeover/pause command.
    """

    def __init__(self, policy, hz=30.0, command_ttl_ns=150_000_000):
        if not math.isfinite(hz) or hz <= 0 or command_ttl_ns <= 0:
            raise ValueError("policy frequency and TTL must be positive")
        self.policy = policy
        self.period = 1.0 / hz
        self.command_ttl_ns = command_ttl_ns
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._completed_key = None
        self._taken_key = None
        self._started_ns = time.monotonic_ns()
        self._counts = dict(unique_frames=0, duplicate_publications=0, superseded_frames=0, model_calls=0)
        self._stop = threading.Event()
        self._observation = None
        self._command = None
        self._error = None
        self._generation = 0
        self._unavailable_reason = None
        self._failure_generation = 0
        self._failure_reason = None
        self._thread = threading.Thread(target=self._run, name="steering-policy", daemon=True)

    def start(self):
        self._thread.start()

    def publish(self, observation):
        with self._ready:
            key = self._frame_key(observation)
            previous = self._frame_key(self._observation)
            if key is not None:
                if key == previous:
                    self._counts["duplicate_publications"] += 1
                    return
                self._counts["unique_frames"] += 1
                if previous is not None and previous != self._taken_key:
                    self._counts["superseded_frames"] += 1
            self._observation = observation
            self._ready.notify()

    @staticmethod
    def _frame_key(observation):
        if isinstance(observation, ModelObservation):
            return observation.frame.frame_id, observation.frame.timestamp_ns
        return None

    def stats(self):
        with self._lock:
            seconds = max((time.monotonic_ns() - self._started_ns) / 1e9, 1e-9)
            return dict(self._counts, unique_frame_hz=self._counts["unique_frames"] / seconds,
                        model_call_hz=self._counts["model_calls"] / seconds,
                        maximum_hz=1 / self.period, rate_window="since worker construction")

    def invalidate(self, reason="input_unavailable"):
        with self._lock:
            self._generation += 1
            self._observation = None
            self._command = None
            self._unavailable_reason = reason
            self._completed_key = None
            self._taken_key = None
            self._ready.notify()

    @property
    def unavailable_reason(self):
        with self._lock:
            return self._unavailable_reason

    def latest(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError("policy failed") from self._error
            return self._command

    def failure_state(self) -> tuple[int, str | None]:
        """Return the latest persistent recoverable-failure event.

        Each consumer keeps its own last-seen generation. Reading does not
        acknowledge or clear the event; successful retries and observation
        invalidation cannot erase it. A control loop must disengage on a newer
        generation even when ``latest()`` already contains a successful retry.
        Unexpected policy errors remain fatal, including between motor ticks.
        """
        with self._lock:
            if self._error is not None:
                raise RuntimeError("policy failed") from self._error
            return self._failure_generation, self._failure_reason

    def _run(self):
        next_start = 0.0
        while not self._stop.is_set():
            with self._ready:
                while not self._stop.is_set():
                    observation = self._observation
                    key = self._frame_key(observation)
                    pending = observation is not None and (key is None or key != self._completed_key)
                    delay = next_start - time.monotonic()
                    if pending and delay <= 0:
                        break
                    self._ready.wait(timeout=max(0.0, delay) if pending else None)
                if self._stop.is_set():
                    return
                generation = self._generation
                self._taken_key = key
                self._counts["model_calls"] += 1
            # A maximum cadence, not a work queue: slow predictions naturally
            # reduce throughput and the next iteration takes only the latest input.
            next_start = time.monotonic() + self.period
            if observation is not None:
                try:
                    prediction_started_ns = time.monotonic_ns()
                    prediction = self.policy.predict(observation)
                    driving = isinstance(prediction, DrivingPrediction)
                    if bool(getattr(self.policy, 'driving', False)) != driving:
                        raise ValueError('policy output does not match its declared task')
                    target = prediction.angle_deg if driving else float(prediction)
                    if not math.isfinite(target):
                        raise ValueError("policy returned a non-finite target")
                    generated = time.monotonic_ns()
                    command = SteeringCommand(target, generated, observation.timestamp_ns,
                                               generated + self.command_ttl_ns,
                                               (generated - prediction_started_ns) / 1e6,
                                               prediction.throttle if driving else None,
                                               prediction.brake if driving else None)
                    with self._lock:
                        if generation == self._generation:
                            self._command = command
                            self._unavailable_reason = None
                            self._completed_key = key
                except ObservationUnavailable as error:
                    with self._lock:
                        # Failure notification is independent of the observation
                        # generation. Invalidating an old input may discard its
                        # result, but must not hide a transport/policy failure
                        # that finishes after that invalidation.
                        self._failure_generation += 1
                        self._failure_reason = str(error)
                        if generation == self._generation:
                            self._command = None
                            self._unavailable_reason = str(error)
                except Exception as error:
                    with self._lock:
                        self._error = error
                    return

    def close(self):
        self._stop.set()
        with self._ready:
            self._ready.notify_all()
        close_policy = getattr(self.policy, "close", None)
        if close_policy is not None:
            close_policy()  # e.g. interrupt a pending network response
        if self._thread.ident is not None:
            self._thread.join(timeout=0.5)
        # A hung model cannot block hardware cleanup; the daemon has no motor access.
