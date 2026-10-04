"""Deterministic regressions for arming, UI queues, and expert-label handover."""

import csv
from dataclasses import replace
import queue

from forza_ai.contracts import CapturedFrame, SteeringCommand, VehicleState
from forza_ai.policies.placeholder import FixedAnglePolicy
from forza_ai.runtime import run
from forza_ai.simulation import SimulatedAdapter


class Clock:
    def __init__(self):
        self.now_ns = 1_000_000_000

    def monotonic_ns(self):
        return self.now_ns

    def monotonic(self):
        return self.now_ns / 1e9

    def sleep(self, seconds):
        self.now_ns += round(seconds * 1e9)


class TickAdapter(SimulatedAdapter):
    def __init__(self, clock, on_tick=lambda adapter: None, *, sample_age_ns=0,
                 output_delay_ns=0, motor_delay_ns=0):
        super().__init__()
        self.clock = clock
        self.on_tick = on_tick
        self.tick = 0
        self.sample_age_ns = sample_age_ns
        self.output_delay_ns = output_delay_ns
        self.motor_delay_ns = motor_delay_ns
        self.motor_writes = []

    def read_state(self, now_ns):
        self.tick += 1
        self.on_tick(self)
        return replace(super().read_state(now_ns), timestamp_ns=now_ns - self.sample_age_ns)

    def write_virtual_state(self, state):
        self.clock.now_ns += self.output_delay_ns
        super().write_virtual_state(state)

    def set_torque(self, torque):
        self.clock.now_ns += self.motor_delay_ns
        super().set_torque(torque)
        self.motor_writes.append((self.tick, self.clock.now_ns, torque))


def install_clock_and_policy(monkeypatch, adapter, command_state=lambda tick: "fresh"):
    """Use the real controller/runtime; avoid scheduler-dependent worker timing."""
    clock = adapter.clock

    class Worker:
        unavailable_reason = None

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def publish(self, observation):
            pass

        def invalidate(self, reason):
            pass

        def latest(self):
            state = command_state(adapter.tick)
            if state == "missing":
                return None
            generated = clock.now_ns - (300_000_000 if state == "stale" else 0)
            return SteeringCommand(10, generated, generated, generated + 150_000_000)

        def failure_state(self):
            return 0, None

    monkeypatch.setattr("forza_ai.runtime.time", clock)
    monkeypatch.setattr("forza_ai.simulation.time", clock)
    monkeypatch.setattr("forza_ai.runtime.PolicyWorker", Worker)


def test_wheel_edges_never_block_on_full_external_queue(monkeypatch):
    class BlockingForbiddenQueue(queue.Queue):
        def put(self, item, block=True, timeout=None):
            # Turn the old deadlock into an immediate test failure.
            assert block is False, "wheel loop attempted a blocking queue put"
            return super().put(item, block=block, timeout=timeout)

    events = BlockingForbiddenQueue(maxsize=1)

    def buttons(adapter):
        if adapter.tick == 2:
            events.put_nowait("unrecognized_external_event")
            adapter.buttons = (1, 2)

    adapter = TickAdapter(Clock(), buttons)
    install_clock_and_policy(monkeypatch, adapter)
    result = run(adapter, FixedAnglePolicy(10), command_queue=events,
                 arm_button=1, route_button=2, duration=0.05)
    assert result["mode"] == "assist"
    assert result["metrics"]["events"]["arm"] == 1
    assert result["metrics"]["routes"]["attempts"] == 1
    assert events.empty()
    assert adapter.closed


def test_held_arm_from_failure_tick_cannot_reengage_on_recovered_command(monkeypatch, tmp_path):
    def buttons(adapter):
        adapter.buttons = (1,) if adapter.tick >= 5 else ()

    adapter = TickAdapter(Clock(), buttons)
    install_clock_and_policy(monkeypatch, adapter,
                             lambda tick: "missing" if tick == 5 else "fresh")
    path = tmp_path / "status.csv"
    result = run(adapter, FixedAnglePolicy(10), assist=True, arm_button=1,
                 duration=0.08, status_path=path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows[:4]] == ["assist"] * 4
    assert rows[4]["mode"] == "takeover" and rows[4]["reason"] == "no_command"
    assert all(row["mode"] == "takeover" for row in rows[5:])
    assert result["metrics"]["events"]["arm"] == 1
    assert result["metrics"]["human_interventions"] == 0


def test_pending_arm_waits_for_fresh_command_instead_of_consuming_stale_one(monkeypatch, tmp_path):
    adapter = TickAdapter(Clock())
    install_clock_and_policy(monkeypatch, adapter,
                             lambda tick: "stale" if tick <= 3 else "fresh")
    path = tmp_path / "status.csv"
    result = run(adapter, FixedAnglePolicy(10), assist=True, duration=0.06, status_path=path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows[:3]] == ["manual"] * 3
    assert rows[3]["mode"] == "assist"
    assert result["mode"] == "assist"
    assert result["metrics"]["events"]["arm"] == 1


def test_external_event_flood_cannot_starve_control_ticks(monkeypatch):
    class Flood:
        reads = 0

        def get_nowait(self):
            self.reads += 1
            assert self.reads <= 512, "unbounded event drain starved the motor loop"
            return "unrecognized_external_event"

    adapter, events = TickAdapter(Clock()), Flood()
    install_clock_and_policy(monkeypatch, adapter)
    result = run(adapter, FixedAnglePolicy(10), command_queue=events, assist=True, duration=0.03)
    assert result["ticks"] >= 3
    assert events.reads == 64 * result["ticks"]
    assert result["max_abs_torque"] > 0


def test_expert_labels_are_sampled_after_motor_release_plus_settle_window(monkeypatch):
    clock, events = Clock(), queue.Queue()

    def takeover(adapter):
        if adapter.tick == 2:
            events.put_nowait("manual")

    adapter = TickAdapter(clock, takeover, sample_age_ns=20_000_000,
                          output_delay_ns=15_000_000, motor_delay_ns=10_000_000)
    install_clock_and_policy(monkeypatch, adapter)

    class Camera:
        def start(self):
            pass

        def close(self):
            pass

        def latest(self):
            return CapturedFrame(adapter.tick, clock.now_ns, None)

    class Telemetry:
        def start(self):
            pass

        def close(self):
            pass

        def latest(self):
            return VehicleState(clock.now_ns, 10, True, adapter.tick, 1000, 0)

    class Recorder:
        stats = {}

        def __init__(self):
            self.samples = []

        def start(self):
            pass

        def close(self, **kwargs):
            pass

        def check(self):
            pass

        def submit(self, wheel, vehicle, frame, mode, *, expert, reason):
            self.samples.append((wheel.timestamp_ns, expert))

    recorder = Recorder()
    run(adapter, FixedAnglePolicy(10), duration=0.2, camera=Camera(), receiver=Telemetry(),
        recorder=recorder, takeover_settle_ms=40, command_queue=events)
    release_ns = next(timestamp for tick, timestamp, torque in adapter.motor_writes if tick == 2)
    settle_boundary = release_ns + 40_000_000
    assert any(expert for _, expert in recorder.samples), "test must include settled human samples"
    assert any(not expert and timestamp < settle_boundary for timestamp, expert in recorder.samples)
    assert all(timestamp >= settle_boundary for timestamp, expert in recorder.samples if expert)
