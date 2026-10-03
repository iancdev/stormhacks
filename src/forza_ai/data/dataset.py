import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from forza_ai.policies.steering_model import Preprocessing, preprocess_rgb


class SteeringDataset(Dataset):
    def __init__(self, sessions, preprocessing: Preprocessing):
        self.samples = [sample for session in sessions for sample in session.samples]
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            pixels = np.asarray(image.convert('RGB'))
        config = self.preprocessing
        return (preprocess_rgb(pixels, config),
                torch.tensor(sample.speed_mps / config.speed_scale_mps, dtype=torch.float32),
                torch.tensor(sample.angle_deg / config.angle_scale_deg, dtype=torch.float32))
