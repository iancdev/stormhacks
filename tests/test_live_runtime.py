import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from forza_ai.contracts import CapturedFrame, ModelObservation, VehicleState, WheelState
from forza_ai.policies.placeholder import SweepPolicy, TestObservation as PlaceholderObservation
from forza_ai.runtime import PolicyWorker, run
from forza_ai.simulation import SimulatedAdapter


class FakeCamera:
    def __init__(self):
        self.closed = False
        self.started = False
        self.frame = None

    def start(self):
        self.started = True

    def latest(self):
        self.frame = CapturedFrame(time.monotonic_ns(), time.monotonic_ns() - 1_000_000,
                                   np.zeros((66, 200, 3), dtype=np.uint8))
        return self.frame

    def close(self):
        self.closed = True


class FakeTelemetry:
    def __init__(self, pause_after=None):
        self.closed = False
        self.pause_after = pause_after

    def start(self):
        self.started = time.monotonic()

    def latest(self):
        race_on = self.pause_after is None or time.monotonic() - self.started < self.pause_after
        return VehicleState(time.monotonic_ns(), 10, race_on, 1, 1000, 0)

    def at_or_before(self, timestamp_ns):
        return VehicleState(timestamp_ns - 1_000_000, 10, True, 1, 1000, 0)

    def close(self):
        self.closed = True


class FakeModelPolicy:
    requires_camera = True

    def predict(self, observation):
        assert isinstance(observation, ModelObservation)
        return 5.0


class LiveRuntimeTests(unittest.TestCase):
    def test_sweep_has_finite_sequence_then_holds_center(self):
        policy = SweepPolicy(5, hold_seconds=2)
        angles = []
        for second in (0, 2, 4, 6, 8, 100):
            state = WheelState(1_000_000_000 + second * 1_000_000_000, 0, 0, 0)
            angles.append(policy.predict(PlaceholderObservation(state)))
        self.assertEqual(angles, [0, 5, 0, -5, 0, 0])

    def test_model_drives_controller_and_resources_close(self):
        adapter, camera, receiver = SimulatedAdapter(), FakeCamera(), FakeTelemetry()
        result = run(adapter, FakeModelPolicy(), camera=camera, receiver=receiver,
                     assist=True, duration=0.2)
        self.assertGreater(result["max_abs_torque"], 0)
        self.assertEqual(result["predicted_angle_deg"], 5)
        self.assertTrue(all((adapter.closed, camera.closed, receiver.closed)))

    def test_shadow_predicts_without_motor_output(self):
        result = run(SimulatedAdapter(), FakeModelPolicy(), camera=FakeCamera(), receiver=FakeTelemetry(),
                     shadow=True, duration=0.15)
        self.assertEqual(result["predicted_angle_deg"], 5)
        self.assertEqual(result["max_abs_torque"], 0)
        self.assertEqual(result["mode"], "manual")

    def test_pausing_game_latches_takeover(self):
        result = run(SimulatedAdapter(), FakeModelPolicy(), camera=FakeCamera(),
                     receiver=FakeTelemetry(pause_after=0.12), assist=True, duration=0.25)
        self.assertEqual(result["mode"], "takeover")
        self.assertIn(result["reason"], ("race_inactive", "telemetry_unavailable_or_paused"))
        self.assertIsNone(result["predicted_angle_deg"])

    def test_invalidated_inflight_prediction_cannot_republish(self):
        entered, release = threading.Event(), threading.Event()

        class BlockedPolicy:
            def predict(self, observation):
                entered.set()
                release.wait(timeout=1)
                return 5.0

        worker = PolicyWorker(BlockedPolicy())
        worker.publish(PlaceholderObservation(WheelState(time.monotonic_ns(), 0, 0, 0)))
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            worker.invalidate("paused")
            release.set()
            worker.close()
            self.assertIsNone(worker.latest())
        finally:
            release.set()
            worker.close()

    def test_fault_preserves_status_after_motor_cleanup(self):
        class BrokenCamera(FakeCamera):
            def latest(self):
                raise RuntimeError("capture stopped")

        adapter, camera, receiver = SimulatedAdapter(), BrokenCamera(), FakeTelemetry()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.csv"
            with self.assertRaisesRegex(RuntimeError, "capture stopped"):
                run(adapter, FakeModelPolicy(), camera=camera, receiver=receiver,
                    assist=True, duration=0.1, status_path=path)
            report = json.loads(path.with_suffix(".json").read_text())
            self.assertIn("capture stopped", report["error"])
            self.assertEqual(adapter.torque, 0)
            self.assertTrue(all((adapter.closed, camera.closed, receiver.closed)))

    def test_cleanup_error_is_saved_and_other_resources_close(self):
        class BadClose(SimulatedAdapter):
            def close(self):
                super().close()
                raise RuntimeError("release failed")

        camera, receiver = FakeCamera(), FakeTelemetry()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.csv"
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                run(BadClose(), FakeModelPolicy(), camera=camera, receiver=receiver,
                    shadow=True, duration=0.05, status_path=path)
            report = json.loads(path.with_suffix(".json").read_text())
            self.assertIn("release failed", report["error"])
            self.assertTrue(report["cleanup_errors"])
            self.assertTrue(camera.closed and receiver.closed)

    def test_late_telemetry_read_cannot_apply_aged_wheel_torque(self):
        from dataclasses import replace

        class DelayedTelemetry(FakeTelemetry):
            def latest(self):
                time.sleep(0.020)
                return super().latest()

        class OldWheel(SimulatedAdapter):
            def __init__(self):
                super().__init__()
                self.sample = None
                self.invalid = []

            def read_state(self, now_ns):
                self.sample = replace(super().read_state(now_ns), timestamp_ns=now_ns - 25_000_000)
                return self.sample

            def set_torque(self, torque):
                if torque and time.monotonic_ns() - self.sample.timestamp_ns > 50_000_000:
                    self.invalid.append(torque)
                super().set_torque(torque)

        adapter = OldWheel()
        result = run(adapter, FakeModelPolicy(), camera=FakeCamera(), receiver=DelayedTelemetry(),
                     assist=True, duration=0.2)
        self.assertFalse(adapter.invalid)
        self.assertEqual(result["max_abs_torque"], 0)

    def test_losing_foreground_blocks_model_torque(self):
        class NotGame:
            def is_active(self):
                return False

        result = run(SimulatedAdapter(), FakeModelPolicy(), camera=FakeCamera(), receiver=FakeTelemetry(),
                     assist=True, duration=0.1, foreground_guard=NotGame())
        self.assertEqual(result["max_abs_torque"], 0)
        self.assertEqual(result["input_status"], "game_not_foreground")

    def test_exported_predictor_is_exercised_by_live_runtime(self):
        # Synthetic weights test the actual inference interface, not driving skill.
        import torch
        from forza_ai.policies.live import LiveModelPolicy
        from forza_ai.policies.steering_model import Preprocessing, SteeringModel
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {"format_version": 1, "architecture": "pilotnet_speed_v1",
                        "image_stage": "road_crop", "preprocessing": Preprocessing().to_dict()}
            (path / "metadata.json").write_text(json.dumps(metadata))
            model = SteeringModel()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.head[-2].bias.fill_(0.01)
            torch.save(model.state_dict(), path / "model.pt")
            result = run(SimulatedAdapter(), LiveModelPolicy(path), camera=FakeCamera(),
                         receiver=FakeTelemetry(), assist=True, duration=0.3)
            self.assertAlmostEqual(result["predicted_angle_deg"], 4.49985, places=3)
            self.assertGreater(result["max_abs_torque"], 0)


if __name__ == "__main__":
    unittest.main()
