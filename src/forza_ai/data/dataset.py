from dataclasses import asdict
import hashlib
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from forza_ai.policies.steering_model import Preprocessing, preprocess_rgb


class SteeringDataset(Dataset):
    def __init__(self, sessions, preprocessing: Preprocessing, cache=None, alignment=None):
        sessions = list(sessions)
        self.samples = [sample for session in sessions for sample in session.samples]
        self.preprocessing = preprocessing
        self.cache = cache
        self.cache_keys = []
        if cache is not None:
            for session in sessions:
                # Freshly validated content plus exact accepted rows prevent stale
                # reuse after filtering/alignment/preprocessing or label changes.
                identity = {
                    'path': str(session.path.resolve()), 'fingerprint': session.fingerprint,
                    'session': session.session_id, 'group': session.group,
                    'preprocessing': preprocessing.to_dict(),
                    'alignment': asdict(alignment) if alignment is not None else None,
                    'accepted': [(str(s.image_path), s.capture_time_ns, s.angle_deg,
                                  s.speed_mps, s.control_mode) for s in session.samples],
                }
                namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                self.cache_keys.extend((namespace, i) for i in range(len(session.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.cache is not None:
            cached = self.cache.get(self.cache_keys[index])
            if cached is not None:
                return cached
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            pixels = np.asarray(image.convert('RGB'))
        config = self.preprocessing
        result = (preprocess_rgb(pixels, config),
                torch.tensor(sample.speed_mps / config.speed_scale_mps, dtype=torch.float32),
                torch.tensor(sample.angle_deg / config.angle_scale_deg, dtype=torch.float32))
        if self.cache is not None:
            self.cache.put(self.cache_keys[index], result)
        return result
