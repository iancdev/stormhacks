import math
import unittest

from forza_ai.contracts import ControlMode, SteeringCommand, WheelState
from forza_ai.control import SteeringConfig, SteeringController


def wheel(now, angle=0.0, **kwargs):
    return WheelState(now, angle, 0.0, 0.0, **kwargs)


def command(now, target=30.0, ttl=150_000_000, observation=None):
    return SteeringCommand(target, now, now if observation is None else observation, now + ttl)


class SteeringTests(unittest.TestCase):
    def test_no_torque_until_engaged_then_slew_limited(self):
        c = SteeringController()
        start = 1_000_000_000
        self.assertEqual(c.step(wheel(start), command(start), start).torque, 0)
        t = start + 10_000_000
        self.assertEqual(c.step(wheel(t), command(t), t, engage=True).torque, 0)
        t += 10_000_000
        status = c.step(wheel(t), command(t), t)
        self.assertAlmostEqual(status.target_angle_deg, 0.6)
        self.assertGreater(status.torque, 0)

    def test_target_and_torque_clamped(self):
        c = SteeringController(SteeringConfig(kp=1, kd=0, target_limit_deg=10))
        start = 1_000_000_000
        c.step(wheel(start), command(start, 1000), start, engage=True)
        for i in range(1, 101):
            t = start + i * 10_000_000
            status = c.step(wheel(t), command(t, 1000), t)
        self.assertEqual(status.target_angle_deg, 10)
        self.assertEqual(status.torque, 0.15)

    def test_takeover_latches_and_requires_explicit_reengagement(self):
        c = SteeringController()
        t = 1_000_000_000
        c.step(wheel(t), command(t), t, engage=True)
        t += 10_000_000
        s = c.step(wheel(t), command(t), t, engage=True, takeover=True)
        self.assertEqual(s.mode, ControlMode.TAKEOVER)
        self.assertEqual(s.torque, 0)
        t += 10_000_000
        self.assertEqual(c.step(wheel(t), command(t), t).mode, ControlMode.TAKEOVER)
        t += 10_000_000
        self.assertEqual(c.step(wheel(t), command(t), t, engage=True).mode, ControlMode.ASSIST)

    def test_expired_command_latches_even_when_new_command_arrives(self):
        c = SteeringController()
        t = 1_000_000_000
        cmd = command(t, ttl=10_000_000)
        c.step(wheel(t), cmd, t, engage=True)
        t += 10_000_000
        s = c.step(wheel(t), cmd, t)
        self.assertEqual(s.reason, "command_expired")
        self.assertEqual(s.torque, 0)
        t += 10_000_000
        self.assertEqual(c.step(wheel(t), command(t), t).mode, ControlMode.TAKEOVER)

    def test_fresh_command_cannot_refresh_stale_observation(self):
        t = 1_000_000_000
        c = SteeringController()
        s = c.step(wheel(t), command(t, observation=t - 300_000_000), t, engage=True)
        self.assertEqual(s.reason, "stale_observation")
        self.assertEqual(s.torque, 0)

    def test_nonfinite_future_and_reordered_commands(self):
        t = 1_000_000_000
        for cmd, reason in [(command(t, math.nan), "invalid_target"),
                            (command(t + 1), "stale_command"),
                            (command(t, observation=t + 1), "future_observation")]:
            with self.subTest(reason=reason):
                c = SteeringController()
                s = c.step(wheel(t), cmd, t, engage=True)
                self.assertEqual(s.reason, reason)
                self.assertEqual(s.torque, 0)
        c = SteeringController()
        c.step(wheel(t), command(t), t, engage=True)
        s = c.step(wheel(t + 10_000_000), command(t - 1), t + 10_000_000)
        self.assertEqual(s.reason, "command_time_reversed")

    def test_invalid_disconnected_stale_wheel_disengages(self):
        t = 1_000_000_000
        for state in (wheel(t, connected=False), wheel(t, angle=math.nan),
                      wheel(t - 100_000_000), wheel(t, angle=500),
                      WheelState(t, 0, 1.1, 0)):
            with self.subTest(state=state):
                c = SteeringController()
                s = c.step(state, command(t), t, engage=True)
                self.assertEqual(s.mode, ControlMode.FAULT)
                self.assertEqual(s.torque, 0)

    def test_slow_control_loop_latches(self):
        t = 1_000_000_000
        c = SteeringController()
        c.step(wheel(t), command(t), t, engage=True)
        t += 150_000_000
        s = c.step(wheel(t), command(t), t)
        self.assertEqual(s.reason, "control_loop_gap")
        self.assertEqual(s.torque, 0)

    def test_derivative_damps_motion_and_repeated_samples_dont_divide_by_zero(self):
        t = 1_000_000_000
        c = SteeringController(SteeringConfig(kp=0, kd=0.01))
        c.step(wheel(t), command(t), t, engage=True)
        t += 10_000_000
        s = c.step(wheel(t, angle=1), command(t), t)
        self.assertLess(s.torque, 0)
        s = c.step(wheel(t, angle=1), command(t), t + 1_000_000)
        self.assertTrue(math.isfinite(s.torque))

    def test_invalid_config(self):
        for kwargs in ({"kp": math.nan}, {"torque_limit": 0}, {"torque_limit": 2},
                       {"max_tick_gap_ns": 0}, {"target_limit_deg": 500}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SteeringConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
