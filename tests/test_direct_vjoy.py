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
