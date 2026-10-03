import json
from pathlib import Path
import queue
import tempfile
import time
import unittest

import numpy as np

from forza_ai.contracts import CapturedFrame, ObservationUnavailable, VehicleState
from forza_ai.data.sessions import load_session
from forza_ai.recording import SessionRecorder
from forza_ai.runtime import run
from forza_ai.simulation import SimulatedAdapter


class Camera:
    def start(self):
        self.index = 0
        self.closed = False

    def latest(self):
        self.index += 1
        return CapturedFrame(self.index, time.monotonic_ns() - 2_000_000,
                             np.full((66, 200, 3), self.index % 255, dtype=np.uint8))

    def close(self):
        self.closed = True


class Telemetry:
    def start(self):
        self.closed = False

    def latest(self):
        return self.at_or_before(time.monotonic_ns())

    def at_or_before(self, timestamp_ns):
        return VehicleState(timestamp_ns - 1_000_000, 10.0, True, 1, 1000.0, 0)

    def close(self):
        self.closed = True


class TimedAdapter(SimulatedAdapter):
    button_count = 4

    def __init__(self, button_schedule=lambda elapsed: (), interrupt_after=None):
        super().__init__()
        self.schedule = button_schedule
        self.started = None
        self.interrupt_after = interrupt_after

    def read_state(self, now_ns):
        self.started = now_ns if self.started is None else self.started
        elapsed = (now_ns - self.started) / 1e9
        if self.interrupt_after is not None and elapsed >= self.interrupt_after:
            raise KeyboardInterrupt
        self.buttons = self.schedule(elapsed)
        return super().read_state(now_ns)


class Policy:
    requires_camera = True

    def __init__(self, unavailable_after=None):
        self.started = None
        self.unavailable_after = unavailable_after

    def predict(self, observation):
        self.started = time.monotonic() if self.started is None else self.started
        if self.unavailable_after is not None and time.monotonic() - self.started >= self.unavailable_after:
            raise ObservationUnavailable("network disconnected")
        return 5.0


class OperatorWorkflowTests(unittest.TestCase):
    def test_arm_edge_and_human_takeover_are_counted_once(self):
        def buttons(elapsed):
            if .04 <= elapsed < .13:
                return (1,)
            if elapsed >= .18:
                return (0,)
            return ()

        result = run(TimedAdapter(buttons), Policy(), camera=Camera(), receiver=Telemetry(),
                     duration=.3, arm_button=1, takeover_button=0)
        self.assertGreater(result["max_abs_torque"], 0)
        self.assertEqual(result["mode"], "takeover")
        self.assertEqual(result["metrics"]["events"]["arm"], 1)
        self.assertEqual(result["metrics"]["human_interventions"], 1)

    def test_held_arm_cannot_reengage_after_policy_failure(self):
        adapter = TimedAdapter(lambda elapsed: (1,) if elapsed >= .04 else ())
        result = run(adapter, Policy(unavailable_after=.12), camera=Camera(), receiver=Telemetry(),
                     duration=.3, arm_button=1, takeover_button=0)
        self.assertGreater(result["max_abs_torque"], 0)
        self.assertEqual(result["mode"], "takeover")
        self.assertEqual(result["metrics"]["events"]["arm"], 1)
        self.assertEqual(result["metrics"]["human_interventions"], 0)

    def test_startup_held_arm_does_not_engage(self):
        result = run(TimedAdapter(lambda elapsed: (1,)), Policy(), camera=Camera(), receiver=Telemetry(),
                     duration=.12, arm_button=1, takeover_button=0)
        self.assertEqual(result["max_abs_torque"], 0)
        self.assertEqual(result["metrics"]["events"]["arm"], 0)

    def test_takeover_wins_simultaneous_arm_and_shadow_never_arms(self):
        for shadow in (False, True):
            with self.subTest(shadow=shadow):
                result = run(TimedAdapter(lambda elapsed: (0, 1) if elapsed >= .04 else ()), Policy(),
                             camera=Camera(), receiver=Telemetry(), duration=.12,
                             arm_button=1, takeover_button=0, shadow=shadow)
                self.assertEqual(result["max_abs_torque"], 0)

    def test_route_button_toggles_once_per_press(self):
        def buttons(elapsed):
            return (2,) if .03 <= elapsed < .09 or elapsed >= .14 else ()

        result = run(TimedAdapter(buttons), Policy(), camera=Camera(), receiver=Telemetry(),
                     duration=.22, route_button=2, shadow=True)
        self.assertEqual(result["metrics"]["routes"]["attempts"], 1)
        self.assertEqual(result["metrics"]["routes"]["completed"], 1)
        self.assertFalse(result["metrics"]["routes"]["active"])

    def test_only_explicit_takeover_becomes_expert_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "session"
            adapter = TimedAdapter(lambda elapsed: (0,) if elapsed >= .15 else ())
            recorder = SessionRecorder(destination)
            result = run(adapter, Policy(), camera=Camera(), receiver=Telemetry(),
                         assist=True, takeover_button=0, duration=.4,
                         recorder=recorder, takeover_settle_ms=50)
            self.assertTrue(result["recording"]["completed"])
            session = load_session(destination)
            self.assertTrue(session.samples)
            self.assertEqual({sample.control_mode for sample in session.samples}, {"takeover"})
            self.assertGreater(session.rejected.get("nonexpert_or_mode_boundary", 0), 0)

    def test_timeout_is_not_an_expert_takeover(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "session"
            result = run(TimedAdapter(), Policy(unavailable_after=.10), camera=Camera(), receiver=Telemetry(),
                         assist=True, duration=.28, recorder=SessionRecorder(destination))
            self.assertTrue(result["recording"]["completed"])
            self.assertEqual(result["metrics"]["human_interventions"], 0)
            self.assertFalse(load_session(destination).samples)

    def test_operator_stop_finalizes_manual_recording_and_unlimited_report(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, report = Path(directory) / "session", Path(directory) / "run.json"
            with self.assertRaises(KeyboardInterrupt):
                run(TimedAdapter(interrupt_after=.16), Policy(), camera=Camera(), receiver=Telemetry(),
                    duration=0, shadow=True, recorder=SessionRecorder(destination), record_manual=True,
                    run_report=report)
            self.assertTrue(load_session(destination).samples)
            saved = json.loads(report.read_text())
            self.assertIsNone(saved["error"])
            self.assertTrue(saved["recording"]["completed"])
            self.assertEqual(saved["stop_reason"], "keyboard_interrupt")
            self.assertGreater(saved["metrics"]["ticks"], 0)

    def test_colliding_buttons_rejected_before_loop(self):
        adapter = TimedAdapter()
        with self.assertRaisesRegex(ValueError, "different"):
            run(adapter, Policy(), arm_button=0, takeover_button=0, camera=Camera(), receiver=Telemetry())
        self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
