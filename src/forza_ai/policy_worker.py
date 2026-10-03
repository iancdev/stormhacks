"""Nonblocking policy exchange with persistent inference-failure notifications."""

import math
import threading
import time

from forza_ai.contracts import ObservationUnavailable, SteeringCommand


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
        with self._lock:
            self._observation = observation

    def invalidate(self, reason="input_unavailable"):
        with self._lock:
            self._generation += 1
            self._observation = None
            self._command = None
            self._unavailable_reason = reason

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
        while not self._stop.is_set():
            started = time.monotonic()
            with self._lock:
                observation = self._observation
                generation = self._generation
            if observation is not None:
                try:
                    prediction_started_ns = time.monotonic_ns()
                    target = float(self.policy.predict(observation))
                    if not math.isfinite(target):
                        raise ValueError("policy returned a non-finite target")
                    generated = time.monotonic_ns()
                    command = SteeringCommand(target, generated, observation.timestamp_ns,
                                               generated + self.command_ttl_ns,
                                               (generated - prediction_started_ns) / 1e6)
                    with self._lock:
                        if generation == self._generation:
                            self._command = command
                            self._unavailable_reason = None
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
            self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))

    def close(self):
        self._stop.set()
        close_policy = getattr(self.policy, "close", None)
        if close_policy is not None:
            close_policy()  # e.g. interrupt a pending network response
        if self._thread.ident is not None:
            self._thread.join(timeout=0.5)
        # A hung model cannot block hardware cleanup; the daemon has no motor access.
