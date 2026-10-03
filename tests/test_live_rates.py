import unittest
from forza_ai.contracts import CapturedFrame
from forza_ai.reporting import LiveRates


class LiveRateTests(unittest.TestCase):
    def test_cached_frames_and_predictions_are_not_new_events(self):
        rates = LiveRates(window_ns=1_000_000_000)
        for index in range(201):
            now = 1_000_000_000 + index * 10_000_000
            source = 1_000_000_000 + (index // 5) * 50_000_000
            rates.update(now, frame=CapturedFrame(index // 5, source, None),
                         prediction_id=source, has_camera=True)
        result = rates.summary()
        self.assertEqual(result["control_hz"], 100)
        self.assertEqual(result["capture_fps"], 20)
        self.assertEqual(result["policy_fps"], 20)

    def test_stopped_inputs_decay_and_absent_camera_is_unknown(self):
        rates = LiveRates(window_ns=1_000_000_000)
        frame = CapturedFrame(1, 1_000_000_000, None)
        rates.update(1_000_000_000, frame=frame, prediction_id=1_000_000_000, has_camera=True)
        self.assertIsNone(rates.summary()["control_hz"])
        rates.update(3_000_000_000, frame=frame, prediction_id=1_000_000_000, has_camera=True)
        self.assertEqual(rates.summary()["capture_fps"], 0)
        self.assertEqual(rates.summary()["policy_fps"], 0)
        rates.update(3_010_000_000, has_camera=False)
        self.assertIsNone(rates.summary()["capture_fps"])

    def test_storage_is_bounded_and_reversed_clock_is_ignored(self):
        rates = LiveRates(window_ns=10**15)
        for index in range(10_000):
            rates.update(index)
        self.assertLessEqual(len(rates._events["control"]), 4096)
        before = rates.summary()
        rates.update(0)
        self.assertEqual(rates.summary(), before)

    def test_future_sources_do_not_count_as_received(self):
        rates = LiveRates()
        rates.update(1_000_000_000, frame=CapturedFrame(1, 3_000_000_000, None),
                     prediction_id=3_000_000_000, has_camera=True)
        rates.update(2_000_000_000, has_camera=True)
        self.assertEqual(rates.summary()["capture_fps"], 0)
        self.assertEqual(rates.summary()["policy_fps"], 0)


if __name__ == "__main__":
    unittest.main()
