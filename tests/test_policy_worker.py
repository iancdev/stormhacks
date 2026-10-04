"""Regression coverage for failures that recover between control-loop reads."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from forza_ai.contracts import CapturedFrame, ModelObservation, ObservationUnavailable, VehicleState
from forza_ai.inference_server import InferenceServer
from forza_ai.network import RemotePolicy
from forza_ai.policy_worker import PolicyWorker


class ScriptedPolicy:
    """Run fixed outcomes, then block so the last published result is stable."""

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.exhausted = threading.Event()
        self.release = threading.Event()

    def predict(self, observation):
        outcome = next(self.outcomes, None)
        if outcome is None:
            self.exhausted.set()
            assert self.release.wait(2), "test did not release policy"
            return 0
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.release.set()


def observation():
    return SimpleNamespace(timestamp_ns=time.monotonic_ns())


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.001)
    raise AssertionError("worker condition did not occur")


def test_immediate_success_does_not_erase_unobserved_failure():
    policy = ScriptedPolicy([ObservationUnavailable("link disconnected"), 7.5])
    worker = PolicyWorker(policy, hz=1000)
    sample = observation()
    worker.publish(sample)
    assert worker.failure_state() == (0, None)
    worker.start()
    try:
        # No consumer reads the failure until a successful retry is published.
        assert policy.exhausted.wait(2)
        command = worker.latest()
        assert command.target_angle_deg == 7.5
        assert command.observation_time_ns == sample.timestamp_ns
        assert worker.unavailable_reason is None
        assert worker.failure_state() == (1, "link disconnected")
        assert worker.failure_state() == (1, "link disconnected")
    finally:
        worker.close()


def test_each_failure_increments_and_invalidation_never_resets_history():
    policy = ScriptedPolicy([ObservationUnavailable("first failure"), 1,
                             ObservationUnavailable("second failure"), 2])
    worker = PolicyWorker(policy, hz=1000)
    worker.publish(observation())
    worker.start()
    try:
        assert policy.exhausted.wait(2)
        assert worker.latest().target_angle_deg == 2
        assert worker.failure_state() == (2, "second failure")
        worker.invalidate("manual_takeover")
        assert worker.latest() is None
        assert worker.unavailable_reason == "manual_takeover"
        assert worker.failure_state() == (2, "second failure")
        worker.publish(observation())
        assert worker.failure_state() == (2, "second failure")
    finally:
        worker.close()


def test_invalidated_inflight_failure_still_has_an_observable_event():
    entered, release = threading.Event(), threading.Event()

    class BlockedFailure:
        def predict(self, sample):
            entered.set()
            assert release.wait(2)
            raise ObservationUnavailable("connection closed after invalidation")

        def close(self):
            release.set()

    worker = PolicyWorker(BlockedFailure())
    worker.publish(observation())
    worker.start()
    try:
        assert entered.wait(2)
        worker.invalidate("manual_takeover")
        release.set()
        wait_for(lambda: worker.failure_state()[0] == 1)
        assert worker.latest() is None
        assert worker.unavailable_reason == "manual_takeover"
        assert worker.failure_state() == (1, "connection closed after invalidation")
    finally:
        worker.close()


def test_unexpected_policy_errors_remain_fatal_for_both_reads():
    error = ValueError("invalid model output")
    policy = ScriptedPolicy([error])
    worker = PolicyWorker(policy, hz=1000)
    worker.publish(observation())
    worker.start()
    try:
        worker._thread.join(2)
        assert not worker._thread.is_alive()
        for read in (worker.latest, worker.failure_state):
            with pytest.raises(RuntimeError, match="policy failed") as caught:
                read()
            assert caught.value.__cause__ is error
    finally:
        worker.close()


def test_loopback_server_failure_and_fast_reconnect_preserve_event():
    entered, release = threading.Event(), threading.Event()

    class DesktopPredictor:
        calls = 0

        def predict(self, pixels, speed):
            self.calls += 1
            if self.calls == 2:
                # Longer than the policy period, so its next attempt can start
                # immediately after the server closes the failed connection.
                time.sleep(.04)
                raise RuntimeError("test model service failure")
            if self.calls == 4:
                entered.set()
                assert release.wait(2)
            return 5 if self.calls == 1 else 9

    key = b"policy-worker-loopback-test-only"
    server = InferenceServer(DesktopPredictor(), port=0, key=key)
    server.start()
    remote = RemotePolicy(*server.address, key=key, timeout_s=.5)
    worker = PolicyWorker(remote, hz=1000)
    now = time.monotonic_ns()
    frame = CapturedFrame(1, now, np.zeros((66, 200, 3), dtype=np.uint8))
    vehicle = VehicleState(now - 1_000_000, 10, True, 1, 1000, 0)
    worker.publish(ModelObservation(frame, vehicle))
    worker.start()
    try:
        assert entered.wait(2), "worker did not reconnect and publish its recovery"
        assert worker.latest().target_angle_deg == 9
        assert worker.unavailable_reason is None
        assert worker.failure_state() == (1, "remote inference timed out or disconnected")
        assert str(server.last_error) == "test model service failure"
    finally:
        release.set()
        worker.close()
        server.close()
