import csv
import time

from forza_ai import runtime
from forza_ai.contracts import ActuationExpired
from forza_ai.inference_server import InferenceServer
from forza_ai.network import RemotePolicy
from forza_ai.policies.placeholder import FixedAnglePolicy
from forza_ai.simulation import SimulatedAdapter
from test_engagement_regressions import Clock, TickAdapter, install_clock_and_policy
from test_live_runtime import FakeCamera, FakeTelemetry


def test_recovered_command_cannot_erase_failure_before_control_reads_it(monkeypatch, tmp_path):
    adapter = TickAdapter(Clock(), lambda a: setattr(a, "buttons", (1,) if a.tick >= 5 else ()))
    install_clock_and_policy(monkeypatch, adapter)
    monkeypatch.setattr(runtime.PolicyWorker, "failure_state",
                        lambda self: (1, "transient disconnect") if adapter.tick >= 5 else (0, None))
    path = tmp_path / "control.csv"
    result = runtime.run(adapter, FixedAnglePolicy(10), assist=True, arm_button=1, duration=.08,
                         status_path=path)
    rows = list(csv.DictReader(path.open()))
    assert all(r["mode"] == "assist" for r in rows[:4])
    assert all(r["mode"] == "takeover" and r["reason"] == "inference_failure" for r in rows[4:])
    assert result["metrics"]["events"]["arm"] == 1
    assert not any(torque for tick, _, torque in adapter.motor_writes if tick >= 5)


def test_explicit_later_arm_can_resume_after_consumed_failure(monkeypatch):
    adapter = TickAdapter(Clock(), lambda a: setattr(a, "buttons", (1,) if a.tick >= 7 else ()))
    install_clock_and_policy(monkeypatch, adapter)
    monkeypatch.setattr(runtime.PolicyWorker, "failure_state",
                        lambda self: (1, "disconnect") if adapter.tick >= 5 else (0, None))
    result = runtime.run(adapter, FixedAnglePolicy(10), assist=True, arm_button=1, duration=.09)
    assert result["mode"] == "assist"
    assert result["metrics"]["events"]["arm"] == 2
    assert any(torque for tick, _, torque in adapter.motor_writes if tick >= 8)


def test_failure_during_virtual_write_is_seen_before_motor_start(monkeypatch):
    adapter = TickAdapter(Clock())
    install_clock_and_policy(monkeypatch, adapter)
    adapter.failed = False
    original = adapter.write_virtual_state

    def write(state):
        original(state)
        adapter.failed |= adapter.tick >= 5

    adapter.write_virtual_state = write
    monkeypatch.setattr(runtime.PolicyWorker, "failure_state",
                        lambda self: (1, "disconnect") if adapter.failed else (0, None))
    result = runtime.run(adapter, FixedAnglePolicy(10), assist=True, duration=.08)
    assert result["reason"] == "inference_failure"
    assert not any(torque for tick, _, torque in adapter.motor_writes if tick >= 5)


def test_absolute_native_deadline_is_propagated_and_expiry_latches(monkeypatch):
    class DelayedNative(TickAdapter):
        def set_torque_before(self, torque, deadline_ns):
            assert torque != 0
            # The wheel's 50ms budget is tighter than the command's 150ms TTL.
            assert deadline_ns == self.clock.now_ns + 50_000_000
            self.deadline = deadline_ns
            self.clock.now_ns += 300_000_000
            raise ActuationExpired("native update exhausted command validity")

    adapter = DelayedNative(Clock())
    install_clock_and_policy(monkeypatch, adapter)
    result = runtime.run(adapter, FixedAnglePolicy(10), assist=True, duration=.4)
    assert result["mode"] == "fault"
    assert result["reason"] == "actuation_deadline_expired"
    assert result["max_abs_torque"] == 0
    assert all(torque == 0 for _, _, torque in adapter.motor_writes)


def test_real_transient_server_failure_requires_rearm_even_after_reconnect():
    class Predictor:
        calls = 0

        def predict(self, image, speed):
            self.calls += 1
            if self.calls == 3:
                time.sleep(.04)  # longer than policy period, shorter than command TTL
                raise RuntimeError("transient inference-service failure")
            return 5.0

    predictor = Predictor()
    server = InferenceServer(predictor, port=0, key="test-only")
    server.start()
    try:
        result = runtime.run(SimulatedAdapter(), RemotePolicy(*server.address, key="test-only"),
                             camera=FakeCamera(), receiver=FakeTelemetry(), assist=True, duration=.5)
        assert predictor.calls > 3  # the service did recover and return new commands
        assert result["max_abs_torque"] > 0
        assert result["mode"] == "takeover" and result["reason"] == "inference_failure"
        assert result["metrics"]["mode_seconds"]["takeover"] > 0
        assert result["metrics"]["events"]["arm"] == 1
    finally:
        server.close()
