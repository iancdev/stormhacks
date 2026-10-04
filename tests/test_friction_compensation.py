"""Static-friction compensation in the steering controller (off by default)."""

import pytest

from forza_ai.contracts import SteeringCommand, WheelState
from forza_ai.control import SteeringConfig, SteeringController


def torque_for(config, target, angle, ticks=40):
    controller = SteeringController(config)
    now = 1_000_000_000
    status = None
    for i in range(ticks):
        wheel = WheelState(now, angle, 0, 0)
        command = SteeringCommand(target, now, now, now + 100_000_000)
        status = controller.step(wheel, command, now, engage=(i == 0))
        now += 10_000_000
    return status.torque, status.target_angle_deg


def test_default_is_plain_pd():
    torque, _ = torque_for(SteeringConfig(kp=0.01, kd=0.0, torque_limit=0.3), 5.0, 0.0)
    assert torque == pytest.approx(0.01 * 5.0)


def test_breakaway_push_added_outside_deadband_in_the_error_direction():
    config = SteeringConfig(kp=0.01, kd=0.0, torque_limit=0.3, friction_ff=0.17, friction_deadband_deg=1.5)
    right, _ = torque_for(config, 5.0, 0.0)
    left, _ = torque_for(config, -5.0, 0.0)
    assert right == pytest.approx(0.01 * 5.0 + 0.17)
    assert left == pytest.approx(-(0.01 * 5.0 + 0.17))


def test_no_breakaway_push_inside_deadband_and_limit_still_applies():
    config = SteeringConfig(kp=0.01, kd=0.0, torque_limit=0.3, friction_ff=0.17, friction_deadband_deg=1.5)
    near, _ = torque_for(config, 1.0, 0.0)
    assert near == pytest.approx(0.01 * 1.0)
    far, _ = torque_for(config, 60.0, 0.0, ticks=200)
    assert far == pytest.approx(0.3)


def test_negative_friction_rejected():
    with pytest.raises(ValueError):
        SteeringConfig(friction_ff=-0.1)


def test_steer_gain_scales_the_target_before_limits():
    config = SteeringConfig(kp=0.01, kd=0.0, torque_limit=0.3, steer_gain=1.6, target_rate_deg_s=1000)
    _, target = torque_for(config, 10.0, 0.0)
    assert target == pytest.approx(16.0)
    capped = SteeringConfig(steer_gain=2.0, target_limit_deg=30.0, target_rate_deg_s=1000)
    _, target = torque_for(capped, 20.0, 0.0)
    assert target == pytest.approx(30.0)


def test_steer_gain_bounds():
    with pytest.raises(ValueError):
        SteeringConfig(steer_gain=0)
    with pytest.raises(ValueError):
        SteeringConfig(steer_gain=5)
