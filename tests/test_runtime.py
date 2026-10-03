import queue
import time
import unittest

from forza_ai.contracts import WheelState
from forza_ai.policies.placeholder import FixedAnglePolicy, TestObservation
from forza_ai.runtime import PolicyWorker, run
from forza_ai.simulation import SimulatedAdapter


class RuntimeTests(unittest.TestCase):
    def test_assist_runtime_forwards_measured_angle_and_cleans_up(self):
        class Spy(SimulatedAdapter):
            def __init__(self):
                super().__init__()
                self.forwarded = []

            def write_virtual_state(self, state):
                self.forwarded.append(state)
                super().write_virtual_state(state)

        adapter = Spy()
        summary = run(adapter, FixedAnglePolicy(10), assist=True, duration=0.25)
        self.assertEqual(summary["mode"], "assist")
        self.assertGreater(summary["max_abs_torque"], 0)
        self.assertLessEqual(summary["max_abs_torque"], 0.15)
        self.assertEqual(adapter.forwarded[0].angle_deg, 0)
        self.assertNotEqual(adapter.forwarded[-1].angle_deg, 10)
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.torque, 0)

    def test_button_blocks_initial_arm(self):
        adapter = SimulatedAdapter()
        adapter.buttons = (2,)
        result = run(adapter, FixedAnglePolicy(10), assist=True, takeover_button=2, duration=0.08)
        self.assertEqual(result["mode"], "takeover")
        self.assertEqual(result["max_abs_torque"], 0)

    def test_io_error_still_releases_hardware(self):
        class Broken(SimulatedAdapter):
            def write_virtual_state(self, state):
                raise RuntimeError("unplugged")

        adapter = Broken()
        with self.assertRaisesRegex(RuntimeError, "unplugged"):
            run(adapter, FixedAnglePolicy(), duration=0.05)
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.torque, 0)

    def test_configuration_error_also_cleans_up(self):
        adapter = SimulatedAdapter()
        with self.assertRaises(ValueError):
            run(adapter, FixedAnglePolicy(), control_hz=0)
        self.assertTrue(adapter.closed)

    def test_missing_takeover_button_rejected_before_motor_commands(self):
        adapter = SimulatedAdapter()
        adapter.button_count = 3
        with self.assertRaisesRegex(ValueError, "absent"):
            run(adapter, FixedAnglePolicy(), assist=True, takeover_button=3)
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.angle, 0)

    def test_quit_event_cleans_up(self):
        events = queue.Queue()
        events.put("quit")
        adapter = SimulatedAdapter()
        run(adapter, FixedAnglePolicy(), command_queue=events, duration=0)
        self.assertTrue(adapter.closed)

    def test_worker_retains_source_timestamp(self):
        worker = PolicyWorker(FixedAnglePolicy())
        source = time.monotonic_ns() - 300_000_000
        worker.publish(TestObservation(WheelState(source, 0, 0, 0)))
        worker.start()
        try:
            end = time.monotonic() + 0.5
            cmd = worker.latest()
            while cmd is None and time.monotonic() < end:
                time.sleep(0.005)
                cmd = worker.latest()
            self.assertIsNotNone(cmd)
            self.assertEqual(cmd.observation_time_ns, source)
            self.assertGreater(cmd.generated_time_ns, source)
        finally:
            worker.close()

    def test_policy_failure_propagates_and_releases_hardware(self):
        class BrokenPolicy:
            def predict(self, observation):
                raise ValueError("bad model")

        adapter = SimulatedAdapter()
        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            run(adapter, BrokenPolicy(), assist=True, duration=0.2)
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.torque, 0)

    def test_output_delay_cannot_use_aged_wheel_for_torque(self):
        from dataclasses import replace

        class DelayedAdapter(SimulatedAdapter):
            def __init__(self):
                super().__init__()
                self.last_sample = None
                self.invalid_outputs = []

            def read_state(self, now_ns):
                self.last_sample = replace(super().read_state(now_ns), timestamp_ns=now_ns - 35_000_000)
                return self.last_sample

            def write_virtual_state(self, state):
                super().write_virtual_state(state)
                time.sleep(0.025)

            def set_torque(self, torque):
                if torque and time.monotonic_ns() - self.last_sample.timestamp_ns > 50_000_000:
                    self.invalid_outputs.append(torque)
                super().set_torque(torque)

        adapter = DelayedAdapter()
        result = run(adapter, FixedAnglePolicy(10), assist=True, duration=0.18)
        self.assertFalse(adapter.invalid_outputs)
        self.assertEqual(result["reason"], "stale_wheel")
        self.assertEqual(result["max_abs_torque"], 0)


if __name__ == "__main__":
    unittest.main()
