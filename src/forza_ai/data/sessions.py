"""Validate the v1 on-disk contract and align continuous expert segments."""
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image


@dataclass(frozen=True)
class Alignment:
    label_offset_ns: int = 0
    max_wheel_gap_ns: int = 50_000_000
    max_telemetry_age_ns: int = 100_000_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int:
                raise ValueError(f"{name} must be integer nanoseconds")
        if self.max_wheel_gap_ns <= 0 or self.max_telemetry_age_ns <= 0:
            raise ValueError("freshness limits must be positive")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    capture_time_ns: int
    angle_deg: float
    speed_mps: float
    control_mode: str


@dataclass
class Session:
    path: Path
    session_id: str
    samples: list[Sample]
    rejected: dict[str, int]
    fingerprint: str
    split_group: str | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def group(self):
        return self.split_group or self.session_id

    def summary(self):
        return {"session_id": self.session_id, "accepted": len(self.samples),
                "rejected": self.rejected, "fingerprint": self.fingerprint,
                "split_group": self.group, "provenance": self.provenance}


def _rows(path, required, timestamp):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required) <= set(reader.fieldnames):
            raise ValueError(f"{path}: required columns: {', '.join(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty stream")
    previous = -1
    for row in rows:
        value = int(row[timestamp])
        if value < 0 or value <= previous:
            raise ValueError(f"{path}: timestamps must be nonnegative and strictly increasing")
        row[timestamp] = value
        previous = value
    return rows


def _number(row, key, low, high):
    value = float(row[key])
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"invalid {key}: {row[key]}")
    row[key] = value


def load_session(path: Path, alignment: Alignment = Alignment()) -> Session:
    path = Path(path).resolve()
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get('schema_version') == 'record_py_aligned_v1':
        from forza_ai.data.recording import load_recording
        return load_recording(path, metadata, alignment)
    group = metadata.get('split_group')
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError('split_group must be a nonempty string')
    expected = {"schema_version": 1, "clock": "monotonic_ns", "image_stage": "road_crop",
                "wheel_rotation_deg": 900, "completed": True}
    for key, value in expected.items():
        if metadata.get(key) != value or (key == "completed" and metadata[key] is not True):
            raise ValueError(f"{path}: metadata {key} must be {value!r}")
    session_id = metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(f"{path}: nonempty session_id required")
    frames = _rows(path / "frames.csv", ["frame_id", "image_path", "capture_time_ns"], "capture_time_ns")
    wheel = _rows(path / "wheel.csv", ["timestamp_ns", "angle_deg", "throttle", "brake", "control_mode"], "timestamp_ns")
    telemetry = _rows(path / "telemetry.csv", ["timestamp_ns", "speed_mps", "is_race_on"], "timestamp_ns")
    if len({f['frame_id'] for f in frames}) != len(frames) or any(not f['frame_id'] for f in frames):
        raise ValueError(f"{path}: duplicate or empty frame_id")
    for row in wheel:
        _number(row, "angle_deg", -450, 450)
        _number(row, "throttle", 0, 1)
        _number(row, "brake", 0, 1)
        if row['control_mode'] not in {'manual', 'assist', 'takeover'}:
            raise ValueError(f"{path}: invalid control_mode")
    for row in telemetry:
        _number(row, "speed_mps", 0, float('inf'))
        if row['is_race_on'] not in {'0', '1'}:
            raise ValueError(f"{path}: is_race_on must be 0 or 1")
    digest = hashlib.sha256()
    for name in ['metadata.json', 'frames.csv', 'wheel.csv', 'telemetry.csv']:
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    wt = [r['timestamp_ns'] for r in wheel]
    tt = [r['timestamp_ns'] for r in telemetry]
    accepted, rejected = [], Counter()
    for frame in frames:
        relative = Path(frame['image_path'])
        image = (path / relative).resolve()
        if relative.is_absolute() or not image.is_relative_to(path):
            raise ValueError(f"{path}: image path escapes session")
        with Image.open(image) as decoded:
            if decoded.format not in {'PNG', 'JPEG'}:
                raise ValueError(f"{image}: only PNG/JPEG supported")
            decoded.convert('RGB').load()
        digest.update(frame['image_path'].encode())
        digest.update(image.read_bytes())
        captured = frame['capture_time_ns']
        target = captured + alignment.label_offset_ns
        start, end = min(captured, target), max(captured, target)
        # Include the source observation and entire offset interval, not just label endpoints.
        wi = bisect_right(wt, start) - 1
        wj = bisect_left(wt, end)
        if wi < 0 or wj >= len(wheel):
            rejected['wheel_coverage'] += 1
            continue
        segment = wheel[wi:wj + 1]
        mode = segment[0]['control_mode']
        if mode not in {'manual', 'takeover'} or any(r['control_mode'] != mode for r in segment):
            rejected['nonexpert_or_mode_boundary'] += 1
            continue
        if any(b['timestamp_ns'] - a['timestamp_ns'] > alignment.max_wheel_gap_ns
               for a, b in zip(segment, segment[1:])):
            rejected['wheel_gap'] += 1
            continue
        # Interpolation relies on BOTH supporting wheel rows, even when an
        # endpoint lies outside the capture-to-target interval. Race state and
        # freshness must hold over that full support, while input speed remains
        # causal at captured below. Exact labels add no extra support interval.
        support_start, support_end = wt[wi], wt[wj]
        ti = bisect_right(tt, support_start) - 1
        tj = bisect_right(tt, support_end) - 1
        if ti < 0 or support_end - tt[tj] > alignment.max_telemetry_age_ns:
            rejected['stale_telemetry'] += 1
            continue
        segment_t = telemetry[ti:tj + 1]
        if support_start - tt[ti] > alignment.max_telemetry_age_ns or any(
            b['timestamp_ns'] - a['timestamp_ns'] > alignment.max_telemetry_age_ns
            for a, b in zip(segment_t, segment_t[1:])
        ):
            rejected['stale_telemetry'] += 1
            continue
        if any(r['is_race_on'] != '1' for r in segment_t):
            rejected['race_off'] += 1
            continue
        left = bisect_right(wt, target) - 1
        right = bisect_left(wt, target)
        if left == right:
            angle = wheel[left]['angle_deg']
        else:
            weight = (target - wt[left]) / (wt[right] - wt[left])
            angle = wheel[left]['angle_deg'] * (1 - weight) + wheel[right]['angle_deg'] * weight
        speed = telemetry[bisect_right(tt, captured) - 1]['speed_mps']
        accepted.append(Sample(image, captured, angle, speed, mode))
    return Session(path, session_id, accepted, dict(rejected), digest.hexdigest(), group,
                   {'format': 'session_v1', 'clock': 'monotonic_ns'})


def load_sessions(root: Path, alignment: Alignment = Alignment()) -> list[Session]:
    root = Path(root)
    paths = [root] if (root / 'metadata.json').exists() else sorted(p.parent for p in root.glob('*/metadata.json'))
    if not paths:
        raise ValueError(f"no sessions found under {root}")
    sessions = [load_session(path, alignment) for path in paths]
    ids = [s.session_id for s in sessions]
    if len(set(ids)) != len(ids):
        raise ValueError("session_id must be unique across the dataset")
    return sessions


def split_sessions(sessions: list[Session], validation_fraction: float, seed: int):
    if any(session.provenance.get('diagnostic_only') for session in sessions):
        raise ValueError('diagnostic-only recordings cannot enter production train/validation splits')
    groups = sorted({session.group for session in sessions})
    if not 0 < validation_fraction < 1 or len(groups) < 2:
        raise ValueError("need at least two independent session groups and 0 < validation_fraction < 1")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * validation_fraction)))
    held_out = set(groups[:count])
    ordered = sorted(sessions, key=lambda s: s.session_id)
    validation = [session for session in ordered if session.group in held_out]
    train = [session for session in ordered if session.group not in held_out]
    if any(not s.samples for s in sessions):
        raise ValueError("every session must have accepted samples; inspect validation report")
    return train, validation
