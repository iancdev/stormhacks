"""PilotNet-style image encoder with speed fusion; output is normalized angle."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn


@dataclass(frozen=True)
class Preprocessing:
    # Normalized [left, top, right, bottom] region of the original RGB frame.
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    width: int = 200
    height: int = 66
    angle_scale_deg: float = 450.0
    speed_scale_mps: float = 50.0

    def __post_init__(self):
        if len(self.crop) != 4:
            raise ValueError("crop must contain left, top, right, bottom")
        l, t, r, b = self.crop
        if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
            raise ValueError("crop must be a nonempty normalized rectangle")
        if (self.width, self.height) != (200, 66):
            raise ValueError("model requires a 200x66 input")
        if not (0 < self.angle_scale_deg < float('inf') and 0 < self.speed_scale_mps < float('inf')):
            raise ValueError("angle and speed scales must be finite and positive")

    def to_dict(self):
        return asdict(self)


def preprocess_rgb(rgb: np.ndarray, config: Preprocessing) -> torch.Tensor:
    """uint8 HWC RGB -> float32 CHW, bilinear resize, pixel / 127.5 - 1."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected uint8 HWC RGB image with three channels")
    height, width = rgb.shape[:2]
    if not height or not width:
        raise ValueError("empty image")
    l, t, r, b = config.crop
    box = (int(l * width), int(t * height), int(r * width), int(b * height))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("crop is empty at this image resolution")
    image = Image.fromarray(rgb).crop(box).resize(
        (config.width, config.height), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


class SteeringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(1152 + 1, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(), nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1), nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.encoder(image), speed.reshape(-1, 1)), dim=1)).squeeze(1)
