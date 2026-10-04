from dataclasses import asdict
import hashlib
import json
import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from forza_ai.policies.steering_model import Preprocessing, preprocess_rgb


SAMPLE_DTYPE = torch.float32


def cache_keys_for_sessions(sessions, preprocessing, alignment=None, task="steering"):
    keys = []
    for session in sessions:
        # Freshly validated content plus exact accepted rows prevent stale
        # reuse after filtering/alignment/preprocessing or label changes.
        identity = {
            'path': str(session.path.resolve()), 'fingerprint': session.fingerprint,
            'session': session.session_id, 'group': session.group,
            'preprocessing': preprocessing.to_dict(), 'task': task,
            'alignment': asdict(alignment) if alignment is not None else None,
            'accepted': [(str(s.image_path), s.capture_time_ns, s.angle_deg,
                          s.speed_mps, s.control_mode, s.throttle, s.brake) for s in session.samples],
        }
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        keys.extend((namespace, i) for i in range(len(session.samples)))
    return keys


def estimate_cache_bytes(sessions, preprocessing, alignment=None, task="steering"):
    # Same namespace/index keys as the dataset; duplicate references count once.
    # The supported preprocessing contract is fixed RGB CHW plus two scalars.
    if (preprocessing.width, preprocessing.height) != (200, 66):
        return None
    count = len(set(cache_keys_for_sessions(sessions, preprocessing, alignment, task)))
    item_bytes = torch.empty((), dtype=SAMPLE_DTYPE, device='cpu').element_size()
    return count * (3 * preprocessing.height * preprocessing.width + (4 if task == "driving" else 2)) * item_bytes


class SteeringDataset(Dataset):
    def __init__(self, sessions, preprocessing: Preprocessing, cache=None, alignment=None, task="steering"):
        if task not in {'steering', 'driving'}:
            raise ValueError('unsupported learning task')
        self.task = task
        sessions = list(sessions)
        self.samples = [sample for session in sessions for sample in session.samples]
        if task == 'driving':
            for sample in self.samples:
                if any(v is None or not math.isfinite(v) or not 0 <= v <= 1
                       for v in (sample.throttle, sample.brake)):
                    raise ValueError('driving training requires finite human throttle/brake labels in [0,1]')
        self.preprocessing = preprocessing
        self.cache = cache
        self.cache_keys = []
        if cache is not None:
            self.cache_keys = cache_keys_for_sessions(sessions, preprocessing, alignment, task)

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
        label = sample.angle_deg / config.angle_scale_deg
        if self.task == 'driving':
            # A signed longitudinal action cannot press both virtual pedals.
            # Braking takes priority when the demonstrator overlaps pedals.
            label = [label, 0.0 if sample.brake > 0 else sample.throttle, sample.brake]
        result = (preprocess_rgb(pixels, config),
                torch.tensor(sample.speed_mps / config.speed_scale_mps, dtype=SAMPLE_DTYPE),
                torch.tensor(label, dtype=SAMPLE_DTYPE))
        if self.cache is not None:
            self.cache.put(self.cache_keys[index], result)
        return result
