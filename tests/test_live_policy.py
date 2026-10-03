import math
import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from forza_ai.capture import DXCamCapture
from forza_ai.contracts import CapturedFrame, ModelObservation, ObservationUnavailable, VehicleState
from forza_ai.policies.live import LiveModelPolicy


class FakeCamera:
    """One-shot DXcam behavior: None without a new frame, cached pixels otherwise."""

    def __init__(self):
        self.is_capturing = False
        self.frames = queue.Queue()
        self.cached = None
        self.calls = []
        self.release_count = 0
        self.release_thread = None
        self.grab_thread = None
        self.inside_grab = threading.Event()
        self.allow_grab = threading.Event()
        self.allow_grab.set()
        self.delay = 0.0

    def grab(self, **kwargs):
        self.calls.append(kwargs)
        self.grab_thread = threading.get_ident()
        self.inside_grab.set()
        try:
            if not self.allow_grab.wait(timeout=3.0):
                raise RuntimeError("test grab timed out")
            if self.delay:
                time.sleep(self.delay)
            try:
                value = self.frames.get_nowait()
            except queue.Empty:
                return None if kwargs.get("new_frame_only") else self.cached
            if isinstance(value, BaseException):
                raise value
            self.cached = value
            return value
        finally:
            self.inside_grab.clear()

    def release(self):
        if self.inside_grab.is_set():
            raise AssertionError("release raced an active grab")
        self.release_count += 1
        self.release_thread = threading.get_ident()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.camera = FakeCamera()
        self.factory_patch = patch("forza_ai.capture._create_camera", return_value=self.camera)
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)
        self.capture = DXCamCapture((10, 20, 14, 23), fps=120, output_idx=2)
        self.addCleanup(self.capture.close)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.002)
        self.fail("capture condition did not occur")

    def test_new_frames_only_and_owned_rgb(self):
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
        pixels[:, :, 0] = 211  # RGB must reach consumers without a BGR swap.
        self.camera.frames.put(pixels)
        self.capture.start()
        frame = self.wait_for(self.capture.latest)
        self.factory.assert_called_once_with((10, 20, 14, 23), 2)
        self.assertEqual(self.camera.calls[0], {"region": (10, 20, 14, 23), "copy": False, "new_frame_only": True})
        np.testing.assert_array_equal(frame.rgb, pixels)
        self.assertFalse(np.shares_memory(frame.rgb, pixels))
        self.assertFalse(frame.rgb.flags.writeable)
        pixels[:] = 0
        self.wait_for(lambda: len(self.camera.calls) >= 3)
        self.assertIs(self.capture.latest(), frame)
        self.assertEqual(frame.rgb[0, 0, 0], 211)
        self.assertIsNone(self.capture.latest(frame.timestamp_ns + 250_000_001))
        self.assertIsNone(self.capture.latest(frame.timestamp_ns - 1))
        self.assertIs(self.capture.latest(frame.timestamp_ns, max_age_ns=0), frame)

    def test_new_identical_pixels_are_not_confused_with_cached_reads(self):
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
        self.camera.frames.put(pixels)
        self.capture.start()
        first = self.wait_for(self.capture.latest)
        self.camera.frames.put(pixels.copy())
        second = self.wait_for(lambda: self._later_frame(first.frame_id))
        self.assertEqual(second.frame_id, first.frame_id + 1)
        self.assertGreater(second.timestamp_ns, first.timestamp_ns)
        np.testing.assert_array_equal(first.rgb, second.rgb)

    def _later_frame(self, frame_id):
        frame = self.capture.latest()
        return frame if frame is not None and frame.frame_id > frame_id else None

    def test_grab_latency_counts_against_freshness(self):
        self.camera.delay = 0.03
        self.camera.frames.put(np.zeros((3, 4, 3), dtype=np.uint8))
        self.capture.start()
        frame = self.wait_for(self.capture.latest)
        self.assertGreaterEqual(time.monotonic_ns() - frame.timestamp_ns, 25_000_000)
        self.assertIsNone(self.capture.latest(max_age_ns=20_000_000))

    def test_clean_release_and_restart(self):
        self.capture.start()
        self.wait_for(lambda: self.camera.calls)
        self.capture.start()
        self.factory.assert_called_once()
        self.capture.close()
        self.capture.close()
        self.assertEqual(self.camera.release_count, 1)
        self.assertEqual(self.camera.release_thread, self.camera.grab_thread)
        self.assertIsNone(self.capture.latest())
        self.camera = FakeCamera()
        self.factory.return_value = self.camera
        self.capture.start()
        self.assertIsNone(self.capture.latest())

    def test_release_waits_for_grab_and_latest_does_not_wait(self):
        self.camera.allow_grab.clear()
        self.capture.start()
        self.assertTrue(self.camera.inside_grab.wait(timeout=1.0))
        started = time.monotonic()
        self.assertIsNone(self.capture.latest())
        self.assertLess(time.monotonic() - started, 0.1)
        closed = threading.Event()
        closer = threading.Thread(target=lambda: (self.capture.close(), closed.set()))
        closer.start()
        try:
            self.assertFalse(closed.wait(timeout=0.03))
            self.assertEqual(self.camera.release_count, 0)
            self.camera.allow_grab.set()
            closer.join(timeout=1.0)
            self.assertTrue(closed.is_set())
            self.assertEqual(self.camera.release_count, 1)
        finally:
            self.camera.allow_grab.set()
            closer.join(timeout=1.0)

    def test_close_timeout_does_not_release_under_active_grab(self):
        self.camera.allow_grab.clear()
        self.capture.start()
        self.assertTrue(self.camera.inside_grab.wait(timeout=1.0))
        try:
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                self.capture.close()
            self.assertEqual(self.camera.release_count, 0)
            with self.assertRaisesRegex(RuntimeError, "still stopping"):
                self.capture.start()
        finally:
            self.camera.allow_grab.set()
            self.capture.close()
        self.assertEqual(self.camera.release_count, 1)

    def test_startup_errors_and_threaded_grab_errors_propagate(self):
        self.factory.side_effect = OSError("display missing")
        with self.assertRaisesRegex(RuntimeError, "startup failed") as error:
            self.capture.start()
        self.assertIsInstance(error.exception.__cause__, OSError)
        self.capture.close()
        self.factory.side_effect = None
        self.capture.start()
        self.camera.frames.put(OSError("capture failed"))
        self.wait_for(lambda: self.camera.release_count == 1)
        with self.assertRaisesRegex(RuntimeError, "screen capture failed") as error:
            self.capture.latest()
        self.assertIsInstance(error.exception.__cause__, OSError)

    def test_existing_threaded_camera_is_rejected_without_releasing_it(self):
        self.camera.is_capturing = True
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            self.capture.start()
        self.assertEqual(self.camera.release_count, 0)
        self.assertFalse(self.camera.calls)

    def test_malformed_capture_is_fatal_and_releases_camera(self):
        self.capture.start()
        self.camera.frames.put(np.zeros((3, 4, 4), dtype=np.uint8))
        self.wait_for(lambda: self.camera.release_count == 1)
        with self.assertRaises(RuntimeError) as error:
            self.capture.latest()
        self.assertIsInstance(error.exception.__cause__, ValueError)

    def test_configuration_validation(self):
        for region in (None, (), (0, 0, 1), (-1, 0, 2, 2), (0, 0, 0, 2), (0, 3, 2, 2), (0.0, 0, 2, 2), (False, 0, 2, 2)):
            with self.subTest(region=region), self.assertRaises(ValueError):
                DXCamCapture(region)
        for fps in (0, -1, math.nan, math.inf, 241, True):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                DXCamCapture((0, 0, 2, 2), fps=fps)
        with self.assertRaises(ValueError):
            DXCamCapture((0, 0, 2, 2), output_idx=-1)
        for value in (-1, math.nan, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.capture.latest(max_age_ns=value)

    def test_dxcam_create_is_lazy_and_requests_rgb(self):
        # Undo the injected factory only for this explicit Windows API test.
        self.factory_patch.stop()
        from forza_ai.capture import _create_camera
        dxcam = Mock()
        with patch("forza_ai.capture.sys.platform", "win32"), patch.dict("sys.modules", {"dxcam": dxcam}):
            _create_camera((0, 0, 2, 2), 1)
        dxcam.create.assert_called_once_with(region=(0, 0, 2, 2), output_idx=1, output_color="RGB", backend="dxgi")


class LivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.zeros((9, 20, 3), dtype=np.uint8)
        self.predictor = Mock()
        self.predictor.predict.return_value = -37.5
        self.policy = LiveModelPolicy(predictor=self.predictor)

    def observation(self, *, frame_time=1_000_000_000, telemetry_time=990_000_000, speed=20.0, race=True, pixels=None):
        frame = CapturedFrame(7, frame_time, self.pixels if pixels is None else pixels)
        vehicle = VehicleState(telemetry_time, speed, race, 900, 1000, 12)
        return ModelObservation(frame, vehicle)

    def test_rgb_road_crop_speed_and_degrees_pass_directly(self):
        observation = self.observation()
        self.assertEqual(self.policy.predict(observation), -37.5)
        self.assertIs(self.predictor.predict.call_args.args[0], self.pixels)
        self.assertEqual(self.predictor.predict.call_args.args[1], 20.0)
        self.assertTrue(self.policy.requires_camera)
        self.assertEqual(observation.timestamp_ns, observation.frame.timestamp_ns)

    def test_pause_stale_and_future_telemetry_are_unavailable(self):
        for kwargs in ({"race": False}, {"telemetry_time": 899_999_999}, {"telemetry_time": 1_000_000_001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ObservationUnavailable):
                self.policy.predict(self.observation(**kwargs))
        self.predictor.predict.assert_not_called()
        self.assertEqual(self.policy.predict(self.observation(telemetry_time=900_000_000)), -37.5)
        self.assertEqual(self.policy.predict(self.observation(telemetry_time=1_000_000_000)), -37.5)

    def test_bad_rgb_speed_and_prediction_are_fatal(self):
        for pixels in (np.zeros((2, 2, 3)), np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((0, 2, 3), dtype=np.uint8), [1, 2]):
            with self.subTest(pixels=type(pixels)), self.assertRaises(ValueError):
                self.policy.predict(self.observation(pixels=pixels))
        for speed in (-1, math.nan, math.inf):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                self.policy.predict(self.observation(speed=speed))
        self.predictor.predict.return_value = math.nan
        with self.assertRaises(ValueError):
            self.policy.predict(self.observation())
        self.predictor.predict.side_effect = OSError("model unavailable")
        with self.assertRaisesRegex(OSError, "model unavailable"):
            self.policy.predict(self.observation())

    def test_constructor_and_lazy_predictor_loading(self):
        for kwargs in ({}, {"export_path": "model", "predictor": self.predictor}, {"predictor": self.predictor, "max_telemetry_age_ns": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LiveModelPolicy(**kwargs)
        module = Mock()
        with patch.dict("sys.modules", {"forza_ai.policies.predictor": module}):
            policy = LiveModelPolicy("/model/export")
        module.SteeringPredictor.assert_called_once_with("/model/export")
        self.assertIs(policy.predictor, module.SteeringPredictor.return_value)


if __name__ == "__main__":
    unittest.main()
