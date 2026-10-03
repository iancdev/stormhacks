"""Recorder-compatible live RGB preprocessing from ``config/capture.json``.

The capture backend first grabs ``region`` from ``output_idx`` as RGB. Pass that
already-cropped array to ``transform``: masks use absolute screen coordinates,
then the result is resized exactly as ``record.py`` does before JPEG encoding.
OpenCV is loaded only when transforming a frame; no hardware is opened here.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


_MAX_CV_INT = 2**31 - 1


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= _MAX_CV_INT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_MAX_CV_INT}]")
    return value


def _box(value, name, allow_negative=False):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain [left, top, right, bottom]")
    lower = -_MAX_CV_INT if allow_negative else 0
    box = tuple(_integer(coordinate, f"{name} coordinate", lower) for coordinate in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{name} must have positive width and height")
    return box


@dataclass(frozen=True)
class CaptureConfig:
    region: tuple[int, int, int, int]
    output_idx: int = 0
    masks: tuple[tuple[int, int, int, int], ...] = ()
    save_width: int = 320
    # Recorder-only synchronization patch; never included in the model road view.
    hud_box: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "region", _box(self.region, "crop"))
        _integer(self.output_idx, "monitor")
        _integer(self.save_width, "save_width", minimum=1)
        if self.hud_box is not None:
            object.__setattr__(self, "hud_box", _box(self.hud_box, "hud_box"))
        if not isinstance(self.masks, (list, tuple)):
            raise ValueError("masks must be a list of absolute screen rectangles")
        object.__setattr__(self, "masks", tuple(
            _box(mask, f"masks[{index}]", allow_negative=True)
            for index, mask in enumerate(self.masks)
        ))
        # Match the recorder's arithmetic and Python's tie-to-even round exactly.
        # Reject a crop/aspect ratio whose rounded height would be zero or could
        # not be represented by OpenCV's signed integer Size argument.
        _integer(self.output_size[1], "computed saved height", minimum=1)

    @classmethod
    def from_dict(cls, config):
        if not isinstance(config, dict):
            raise ValueError("capture config must be a JSON object")
        unknown = set(config) - {"monitor", "crop", "masks", "save_width", "hud_box"}
        if unknown:
            raise ValueError("unknown capture config fields: " + ", ".join(sorted(map(str, unknown))))
        if config.get("crop") is None:
            raise ValueError("capture config needs a crop; run python record.py setup first")
        return cls(region=config["crop"], output_idx=config.get("monitor", 0),
                   masks=config.get("masks", []), save_width=config.get("save_width", 320),
                   hud_box=config.get("hud_box"))

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid capture JSON in {path}: {error.msg}") from error
        return cls.from_dict(config)

    @property
    def output_size(self) -> tuple[int, int]:
        """Recorder ``save_size`` as OpenCV (width, height), with even height."""
        left, top, right, bottom = self.region
        height = int(round(self.save_width * (bottom - top) / (right - left) / 2)) * 2
        return self.save_width, height

    @property
    def masks_in_crop(self) -> tuple[tuple[int, int, int, int], ...]:
        left, top = self.region[:2]
        return tuple((x0 - left, y0 - top, x1 - left, y1 - top)
                     for x0, y0, x1, y1 in self.masks)

    def transform(self, rgb_cropped: np.ndarray) -> np.ndarray:
        """Copy, mask, and resize a uint8 HWC RGB crop without swapping channels."""
        left, top, right, bottom = self.region
        shape = (bottom - top, right - left, 3)
        if (not isinstance(rgb_cropped, np.ndarray) or rgb_cropped.dtype != np.uint8
                or rgb_cropped.shape != shape):
            raise ValueError(f"expected a uint8 HWC RGB crop with shape {shape}")
        try:
            import cv2
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "OpenCV is required for recorder-compatible capture preprocessing. "
                "Install the hardware extra on Windows (python -m pip install '.[hardware]') "
                "or install opencv-python on this device."
            ) from error
        out = rgb_cropped.copy()
        for x0, y0, x1, y1 in self.masks_in_crop:
            # Keep the recorder's clipping semantics, including fully or partly
            # out-of-crop masks. NumPy bounds the upper slice ends automatically.
            out[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 0
        return cv2.resize(out, self.output_size, interpolation=cv2.INTER_AREA)
