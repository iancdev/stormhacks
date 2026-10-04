"""Portable CPU inference. Input is already a road crop, never a raw full screen."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from forza_ai.policies.steering_model import Preprocessing, SteeringModel, DrivingModel, preprocess_rgb
from forza_ai.contracts import DrivingPrediction


class SteeringPredictor:
    def __init__(self, artifact: str | Path):
        artifact = Path(artifact)
        metadata = json.loads((artifact / 'metadata.json').read_text())
        if metadata.get('format_version') != 1 or metadata.get('architecture') != 'pilotnet_speed_v1':
            raise ValueError('unsupported model artifact')
        if metadata.get('image_stage') != 'road_crop':
            raise ValueError('artifact must consume road crops')
        self.preprocessing = Preprocessing(**metadata['preprocessing'])
        self.metadata = metadata
        self.model = SteeringModel().cpu().eval()
        self.model.load_state_dict(torch.load(artifact / 'model.pt', map_location='cpu', weights_only=True))

    def predict(self, road_crop_rgb: np.ndarray, speed_mps: float) -> float:
        if not math.isfinite(speed_mps) or speed_mps < 0:
            raise ValueError('speed_mps must be finite and nonnegative')
        config = self.preprocessing
        image = preprocess_rgb(road_crop_rgb, config).unsqueeze(0)
        speed = torch.tensor([speed_mps / config.speed_scale_mps], dtype=torch.float32)
        with torch.inference_mode():
            angle = float(self.model(image, speed)[0]) * config.angle_scale_deg
        if not math.isfinite(angle):
            raise ValueError('non-finite predicted angle')
        return angle


class DrivingPredictor(SteeringPredictor):
    driving = True

    def __init__(self, artifact):
        artifact = Path(artifact)
        metadata = json.loads((artifact / 'metadata.json').read_text())
        if (metadata.get('format_version'), metadata.get('architecture')) != (2, 'pilotnet_driving_v2'):
            raise ValueError('driving mode requires a v2 driving artifact')
        if metadata.get('image_stage') != 'road_crop' or metadata.get('output', {}).get('fields') != ['angle_deg', 'throttle', 'brake']:
            raise ValueError('unsupported driving artifact contract')
        self.preprocessing = Preprocessing(**metadata['preprocessing'])
        self.metadata = metadata
        self.model = DrivingModel().cpu().eval()
        self.model.load_state_dict(torch.load(artifact / 'model.pt', map_location='cpu', weights_only=True))

    def predict(self, road_crop_rgb, speed_mps):
        if not math.isfinite(speed_mps) or speed_mps < 0:
            raise ValueError('speed_mps must be finite and nonnegative')
        config = self.preprocessing
        image = preprocess_rgb(road_crop_rgb, config).unsqueeze(0)
        speed = torch.tensor([speed_mps / config.speed_scale_mps], dtype=torch.float32)
        with torch.inference_mode():
            angle, throttle, brake = self.model(image, speed)[0].tolist()
        return DrivingPrediction(angle * config.angle_scale_deg, throttle, brake)


def load_predictor(artifact):
    metadata = json.loads((Path(artifact) / 'metadata.json').read_text())
    return DrivingPredictor(artifact) if metadata.get('format_version') == 2 else SteeringPredictor(artifact)
