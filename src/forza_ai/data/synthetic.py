"""Small deterministic software fixtures; these do not model real driving."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def generate(root: Path, sessions: int = 3, frames: int = 24, seed: int = 7):
    root = Path(root)
    if sessions < 2 or frames < 4:
        raise ValueError('synthetic fixture requires >=2 sessions and >=4 frames')
    if root.exists() and any(root.iterdir()):
        raise ValueError('synthetic destination must be empty')
    rng = np.random.default_rng(seed)
    for n in range(sessions):
        path = root / f'synthetic-{n:03d}'
        (path / 'images').mkdir(parents=True, exist_ok=True)
        (path / 'metadata.json').write_text(json.dumps({
            'schema_version': 1, 'session_id': path.name, 'clock': 'monotonic_ns',
            'wheel_rotation_deg': 900, 'image_stage': 'road_crop', 'completed': True,
            'synthetic': True, 'seed': seed,
        }, indent=2) + '\n')
        indexed, wheels, telemetry = [], [], []
        for i in range(frames):
            timestamp = 1_000_000_000 + i * 33_333_333
            angle = float(75 * np.sin(i / 5 + n))
            pixels = rng.integers(0, 25, size=(80, 240, 3), dtype=np.uint8)
            center = int(120 + angle)
            pixels[:, max(0, center - 3):min(240, center + 3), :] = [240, 180, 80]
            image_path = f'images/{i:06d}.png'
            Image.fromarray(pixels).save(path / image_path)
            indexed.append([i, image_path, timestamp])
            wheels.append([timestamp, angle, 0.4, 0.0, 'manual'])
            telemetry.append([timestamp, 15 + n, 1])
        for filename, columns, rows in [
            ('frames.csv', ['frame_id', 'image_path', 'capture_time_ns'], indexed),
            ('wheel.csv', ['timestamp_ns', 'angle_deg', 'throttle', 'brake', 'control_mode'], wheels),
            ('telemetry.csv', ['timestamp_ns', 'speed_mps', 'is_race_on'], telemetry),
        ]:
            with (path / filename).open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(rows)
    return root
