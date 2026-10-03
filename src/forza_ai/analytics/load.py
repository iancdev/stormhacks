"""Convert existing CSV artifacts into hypertable rows. Nothing here touches the recorder or runtime."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

COLUMNS = ["time", "session", "source", "segment", "frame_id", "image_path", "mode", "steer_deg",
           "target_deg", "predicted_deg", "torque", "speed_mps", "gas", "brake", "race_on", "obs_age_ms"]
RECORDER_COLUMNS = ["frame", "segment", "t", "steer_raw", "steer_deg", "brake", "gas",
                    "wheel_age_ms", "speed_mps", "race_on", "tele_steer", "tele_age_ms"]
RUNTIME_REQUIRED = {"timestamp_ns", "mode", "target_angle_deg", "actual_angle_deg", "torque"}


def _float(value):
    return None if value in (None, "", "None") else float(value)


def _row(**values):
    row = dict.fromkeys(COLUMNS)
    row.update(values)
    return row


def recording_rows(path):
    """Rows from a record.py recording. Time is anchored to the session's YYYYMMDD_HHMMSS name (local clock)."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    session = str(meta["session"])
    try:
        start = datetime.strptime(session, "%Y%m%d_%H%M%S").astimezone()
    except ValueError:
        start = datetime.fromtimestamp((path / "meta.json").stat().st_mtime).astimezone()
    with (path / "labels.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RECORDER_COLUMNS:
            raise ValueError("labels.csv columns do not match record.py")
        rows = []
        for raw in reader:
            frame = int(raw["frame"])
            rows.append(_row(time=start + timedelta(seconds=float(raw["t"])), session=session, source="recorder",
                             segment=int(raw["segment"]), frame_id=frame, image_path=f"frames/{frame:06d}.jpg",
                             mode="manual", steer_deg=_float(raw["steer_deg"]), speed_mps=_float(raw["speed_mps"]),
                             gas=_float(raw["gas"]), brake=_float(raw["brake"]),
                             race_on=None if raw["race_on"] == "" else raw["race_on"] == "1",
                             obs_age_ms=_float(raw["wheel_age_ms"])))
    return session, rows


def runtime_rows(path, session=None, end_time=None):
    """Rows from `forza_ai.runtime --status-csv`. Monotonic timestamps are anchored so the last tick is `end_time`
    (default: the CSV's modification time)."""
    path = Path(path)
    session = session or path.stem
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not RUNTIME_REQUIRED <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected runtime status columns {sorted(RUNTIME_REQUIRED)}")
        raws = list(reader)
    if not raws:
        return session, []
    end_time = end_time or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    last_ns = int(raws[-1]["timestamp_ns"])
    rows = []
    for raw in raws:
        offset = timedelta(microseconds=(int(raw["timestamp_ns"]) - last_ns) / 1000)
        rows.append(_row(time=end_time + offset, session=session, source="runtime", mode=raw["mode"],
                         steer_deg=_float(raw["actual_angle_deg"]), target_deg=_float(raw["target_angle_deg"]),
                         predicted_deg=_float(raw.get("requested_angle_deg")), torque=_float(raw["torque"]),
                         speed_mps=_float(raw.get("speed_mps")), obs_age_ms=_float(raw.get("observation_age_ms"))))
    return session, rows


def copy_rows(conn, rows, replace=True):
    """Bulk load with COPY. By default an existing (session, source) is replaced so reloads are idempotent."""
    if not rows:
        return 0
    session, source = rows[0]["session"], rows[0]["source"]
    with conn.cursor() as cursor:
        if replace:
            cursor.execute("DELETE FROM wheel_samples WHERE session = %s AND source = %s", (session, source))
        with cursor.copy(f"COPY wheel_samples ({', '.join(COLUMNS)}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row([row[column] for column in COLUMNS])
    return len(rows)
