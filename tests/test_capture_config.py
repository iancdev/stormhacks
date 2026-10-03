"""Exact live preprocessing compatibility with the received recorder."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from forza_ai.capture_config import CaptureConfig


def recorder_module():
    path = Path(__file__).resolve().parents[1] / "record.py"
    spec = importlib.util.spec_from_file_location("capture_config_reference_recorder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("cv2") is not None, "OpenCV required for recorder parity checks")
class RecorderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorder = recorder_module()

    def check_parity(self, raw_config):
        config = CaptureConfig.from_dict(raw_config)
        left, top, right, bottom = config.region
        image = np.random.default_rng(71).integers(0, 256, (bottom - top, right - left, 3), dtype=np.uint8)
        original = image.copy()
        expected_size = self.recorder.save_size(raw_config)
        expected_masks = self.recorder.masks_in_crop(raw_config)
        self.assertEqual(config.output_size, expected_size)
        self.assertEqual(config.masks_in_crop, tuple(expected_masks))
        expected = self.recorder.process(image, expected_masks, expected_size)
        actual = config.transform(image)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(actual.shape, (expected_size[1], expected_size[0], 3))
        self.assertEqual(actual.dtype, np.uint8)
        return actual

    def test_crop_masks_resize_match_recorder_including_outside_masks(self):
        self.check_parity({"monitor": 1, "crop": [100, 50, 171, 89], "save_width": 32,
                           "masks": [[110, 55, 121, 65], [80, 40, 110, 58],
                                     [165, 80, 200, 120], [0, 0, 10, 10],
                                     [180, 60, 190, 70], [120, 95, 130, 105]]})

    def test_no_mask_resize_matches_recorder(self):
        self.check_parity({"monitor": 0, "crop": [0, 0, 960, 301], "save_width": 320, "masks": []})

    def test_even_height_rounding_ties_match_recorder(self):
        for crop_height, expected in ((5, 4), (7, 8), (9, 8), (11, 12)):
            raw = {"monitor": 0, "crop": [0, 0, 4, crop_height], "save_width": 4, "masks": []}
            with self.subTest(height=crop_height):
                actual = self.check_parity(raw)
                self.assertEqual(actual.shape[0], expected)

    def test_complete_offscreen_mask_clips_exactly_and_blackens_crop(self):
        actual = self.check_parity({"monitor": 0, "crop": [5, 5, 15, 15], "save_width": 8,
                                   "masks": [[-20, -20, 20, 20]]})
        self.assertFalse(np.any(actual))

    def test_rgb_channels_match_bgr_recording_after_decode_conversion(self):
        config = CaptureConfig.from_dict({"crop": [2, 4, 8, 8], "masks": [[2, 4, 4, 6]], "save_width": 3})
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 200, 50, 10
        bgr_recorded = self.recorder.process(rgb[..., ::-1], config.masks_in_crop, config.output_size)
        live_rgb = config.transform(rgb)
        np.testing.assert_array_equal(live_rgb, bgr_recorded[..., ::-1])
        self.assertGreater(int(live_rgb[-1, -1, 0]), int(live_rgb[-1, -1, 2]))

    def test_read_only_input_and_repeat_calls_are_deterministic(self):
        config = CaptureConfig.from_dict({"crop": [0, 0, 6, 4], "save_width": 3})
        rgb = np.full((4, 6, 3), 255, dtype=np.uint8)
        rgb.flags.writeable = False
        np.testing.assert_array_equal(config.transform(rgb), config.transform(rgb))


class ConfigValidationTests(unittest.TestCase):
    def test_from_json_defaults_and_monitor_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps({"monitor": 2, "crop": [10, 20, 650, 260]}))
            config = CaptureConfig.from_json(path)
        self.assertEqual(config.region, (10, 20, 650, 260))
        self.assertEqual(config.output_idx, 2)
        self.assertEqual(config.masks, ())
        self.assertEqual(config.output_size, (320, 120))

    def test_config_owns_immutable_box_copies(self):
        raw = {"crop": [0, 0, 4, 4], "masks": [[0, 0, 2, 2]]}
        config = CaptureConfig.from_dict(raw)
        raw["crop"][2] = 100
        raw["masks"][0][2] = 100
        self.assertEqual(config.region, (0, 0, 4, 4))
        self.assertEqual(config.masks, ((0, 0, 2, 2),))

    def test_invalid_root_crop_and_unknown_fields(self):
        for raw in ([], None, {}, {"crop": None}, {"crop": "0,0,4,4"},
                    {"crop": [0, 0, 0, 2]}, {"crop": [0, 2, 4, 1]},
                    {"crop": [-1, 0, 4, 4]}, {"crop": [0, 0, 4]},
                    {"crop": [0, 0, 4, 4], "save_witdh": 320}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                CaptureConfig.from_dict(raw)

    def test_non_integer_and_nonfinite_configuration_rejected(self):
        for invalid in (True, 3.0, float("nan"), float("inf"), "3", None):
            for key in ("monitor", "save_width"):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    CaptureConfig.from_dict({"crop": [0, 0, 4, 4], key: invalid})
            with self.subTest(crop=invalid), self.assertRaises(ValueError):
                CaptureConfig.from_dict({"crop": [0, 0, invalid, 4]})
            with self.subTest(mask=invalid), self.assertRaises(ValueError):
                CaptureConfig.from_dict({"crop": [0, 0, 4, 4], "masks": [[0, 0, invalid, 2]]})

    def test_invalid_masks_and_zero_height_output_rejected(self):
        for masks in (None, "mask", [[0, 0, 2]], [[0, 0, 0, 2]], [[3, 0, 2, 2]]):
            with self.subTest(masks=masks), self.assertRaises(ValueError):
                CaptureConfig.from_dict({"crop": [0, 0, 4, 4], "masks": masks})
        with self.assertRaisesRegex(ValueError, "computed saved height"):
            CaptureConfig.from_dict({"crop": [0, 0, 100, 1], "save_width": 2})
        with self.assertRaises(ValueError):
            CaptureConfig.from_dict({"crop": [0, 0, 4, 4], "save_width": 0})
        with self.assertRaises(ValueError):
            CaptureConfig.from_dict({"crop": [0, 0, 4, 4], "monitor": -1})

    def test_transform_rejects_wrong_shape_channels_and_dtype(self):
        config = CaptureConfig.from_dict({"crop": [10, 20, 14, 24]})
        for image in (None, [], np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4), np.uint8),
                      np.zeros((4, 4, 3), np.float32), np.zeros((5, 4, 3), np.uint8)):
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                config.transform(image)

    def test_missing_opencv_explains_install_action(self):
        config = CaptureConfig.from_dict({"crop": [0, 0, 4, 4]})
        with patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "opencv-python"):
                config.transform(np.zeros((4, 4, 3), np.uint8))

    def test_importing_configuration_does_not_import_opencv(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; from forza_ai.capture_config import CaptureConfig; "
                                 "CaptureConfig.from_dict({'crop': [0, 0, 4, 4]}); "
                                 "assert 'cv2' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_json_reports_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{")
            with self.assertRaisesRegex(ValueError, "broken.json"):
                CaptureConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
