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


if __name__ == "__main__":
    unittest.main()
