"""Direct-vJoy fallback: the AI target reaches vJoy steering, the motor never moves."""

from forza_ai.contracts import ControlMode
from forza_ai.policies.placeholder import FixedAnglePolicy
from forza_ai.runtime import run
from forza_ai.simulation import SimulatedAdapter


class RecordingAdapter(SimulatedAdapter):
    """Keeps every vJoy write and every torque request; can force the wheel angle."""

    def __init__(self, forced_angle_after=None):
        super().__init__()
        self.writes = []
        self.torques = []
        self.reads = 0
        self.forced_angle_after = forced_angle_after   # (read count, angle)

    def read_state(self, now_ns):
        state = super().read_state(now_ns)
        self.reads += 1
        if self.forced_angle_after and self.reads >= self.forced_angle_after[0]:
            self.angle = self.forced_angle_after[1]
            state = super().read_state(now_ns)
        return state

    def write_virtual_state(self, state):
        super().write_virtual_state(state)
        self.writes.append(state.angle_deg)

    def set_torque(self, torque):
        self.torques.append(torque)
        super().set_torque(torque)


def test_direct_vjoy_sends_target_to_vjoy_and_never_uses_motor():
    adapter = RecordingAdapter()
    summary = run(adapter, FixedAnglePolicy(5.0), duration=1.0, assist=True, direct_vjoy=True)
    assert summary["mode"] == ControlMode.ASSIST.value
    assert summary["actuation"] == "direct_vjoy"
    assert summary["max_abs_torque"] == 0.0
    assert all(t == 0.0 for t in adapter.torques)
    assert adapter.angle == 0.0                       # physical wheel untouched
    assert abs(adapter.writes[-1] - 5.0) < 1e-6       # vJoy follows the AI target
    # rate-limited ramp from the wheel position, no jump on engagement
    steps = [b - a for a, b in zip(adapter.writes, adapter.writes[1:])]
    assert max(steps) <= 60.0 * 0.2 + 1e-6


def test_turning_the_wheel_takes_over_and_restores_passthrough():
    adapter = RecordingAdapter(forced_angle_after=(60, 30.0))
    summary = run(adapter, FixedAnglePolicy(5.0), duration=1.0, assist=True, direct_vjoy=True,
                  direct_override_deg=20.0)
    assert summary["mode"] == ControlMode.TAKEOVER.value
    assert summary["reason"] == "manual_takeover"
    assert adapter.writes[-1] == 30.0                 # back to the human's wheel
    assert all(t == 0.0 for t in adapter.torques)


def test_not_engaged_forwards_measured_wheel():
    adapter = RecordingAdapter()
    run(adapter, FixedAnglePolicy(5.0), duration=0.3, assist=False, direct_vjoy=True)
    assert set(adapter.writes) == {0.0}


def test_motor_mode_still_forwards_measured_wheel_only():
    adapter = RecordingAdapter()
    summary = run(adapter, FixedAnglePolicy(5.0), duration=1.0, assist=True)
    assert summary["actuation"] == "motor"
    assert summary["max_abs_torque"] > 0.0
    assert all(abs(w - 5.0) > 1e-9 or w == adapter.angle for w in adapter.writes[:1])
    assert max(adapter.writes) < 5.0 + 1.0            # vJoy shows the physical wheel, not the target


def test_steering_only_direct_output_uses_native_deadline_and_restores_human_on_expiry():
    from forza_ai.contracts import ActuationExpired
    class ExpiringAdapter(RecordingAdapter):
        deadlines = 0
        def write_virtual_state_before(self, state, deadline_ns, *, physical_pedals=False):
            assert physical_pedals
            self.deadlines += 1
            assert deadline_ns > 0
            raise ActuationExpired("native preparation stalled")
    adapter = ExpiringAdapter()
    result = run(adapter, FixedAnglePolicy(5), duration=.15, assist=True, direct_vjoy=True)
    assert adapter.deadlines == 1
    assert result["mode"] == ControlMode.FAULT.value
    assert result["reason"] == "actuation_deadline_expired"
    assert adapter.writes[-1] == adapter.angle == 0
    assert all(t == 0 for t in adapter.torques)


def test_deadline_writer_preserves_human_overlapping_pedals_only_when_explicit():
    import time
    import pytest
    from forza_ai.contracts import WheelState
    from test_hardware import Rig
    rig = Rig()
    adapter = rig.adapter()
    try:
        sample = WheelState(time.monotonic_ns(), 0, .4, .3)
        with pytest.raises(ValueError, match="invalid driving"):
            adapter.write_virtual_state_before(sample, time.monotonic_ns() + 1_000_000_000)
        adapter.write_virtual_state_before(sample, time.monotonic_ns() + 1_000_000_000,
                                           physical_pedals=True)
        assert rig.virtual_axes[0x31] < 32768 and rig.virtual_axes[0x32] < 32768
    finally:
        adapter.close()


def test_mirror_wheel_drives_vjoy_from_ai_and_moves_the_wheel_with_the_motor():
    adapter = RecordingAdapter()
    summary = run(adapter, FixedAnglePolicy(10.0), duration=1.5, assist=True, direct_vjoy=True, mirror_wheel=True)
    assert summary["mode"] == ControlMode.ASSIST.value
    assert abs(adapter.writes[-1] - 10.0) < 1e-6        # Forza gets the AI's angle directly
    assert summary["max_abs_torque"] > 0.0              # the motor is used to mirror it
    assert adapter.angle > 5.0                          # and the physical wheel follows


def test_mirror_wheel_grab_far_from_target_takes_over():
    adapter = RecordingAdapter(forced_angle_after=(60, -40.0))   # human holds the wheel at -40 deg
    summary = run(adapter, FixedAnglePolicy(10.0), duration=1.5, assist=True, direct_vjoy=True, mirror_wheel=True)
    assert summary["mode"] == ControlMode.TAKEOVER.value
    assert summary["reason"] == "manual_takeover"
    assert adapter.writes[-1] == -40.0                  # back to the human's wheel


def test_mirror_wheel_requires_direct_vjoy():
    import pytest
    with pytest.raises(ValueError):
        run(RecordingAdapter(), FixedAnglePolicy(5.0), duration=0.1, assist=True, mirror_wheel=True)
