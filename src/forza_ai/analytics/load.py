"""Convert existing CSV artifacts into hypertable rows. Nothing here touches the recorder or runtime."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

COLUMNS = ["time", "session", "source", "segment", "frame_id", "image_path", "mode", "steer_deg",
           "target_deg", "predicted_deg", "torque", "speed_mps", "gas", "brake", "race_on", "obs_age_ms",
           "yaw_rate", "gear", "rpm", "game_ms"]
EVENT_COLUMNS = ["time", "session", "control_mode", "training_mode", "expert", "reason"]
REPORT_COLUMNS = ["session", "actuation", "duration_s", "ticks", "manual_s", "assist_s", "takeover_s", "fault_s",
                  "human_interventions", "interventions_per_assist_minute", "tracking_rmse_deg", "max_abs_torque",
                  "fault_entries", "routes_attempted", "routes_completed", "routes_aborted", "error", "report"]
RECORDER_COLUMNS = ["frame", "segment", "t", "steer_raw", "steer_deg", "brake", "gas",
                    "wheel_age_ms", "speed_mps", "race_on", "tele_steer", "tele_age_ms"]
RUNTIME_REQUIRED = {"timestamp_ns", "mode", "target_angle_deg", "actual_angle_deg", "torque"}
SESSION_FIELDS = {
    "wheel": ["timestamp_ns", "angle_deg", "throttle", "brake", "control_mode"],
    "telemetry": ["timestamp_ns", "speed_mps", "is_race_on", "game_timestamp_ms", "rpm", "steering_input"],
    "events": ["timestamp_ns", "control_mode", "training_mode", "expert", "reason"],
}


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
        # Newer recorder versions append diagnostic columns (race_time, yaw_rate, gear, car_*); the
        # original twelve must lead unchanged.
        if (reader.fieldnames or [])[:len(RECORDER_COLUMNS)] != RECORDER_COLUMNS:
            raise ValueError("labels.csv columns do not match record.py")
        rows = []
        for raw in reader:
            frame = int(raw["frame"])
            rows.append(_row(time=start + timedelta(seconds=float(raw["t"])), session=session, source="recorder",
                             segment=int(raw["segment"]), frame_id=frame, image_path=f"frames/{frame:06d}.jpg",
                             mode="manual", steer_deg=_float(raw["steer_deg"]), speed_mps=_float(raw["speed_mps"]),
                             gas=_float(raw["gas"]), brake=_float(raw["brake"]),
                             race_on=None if raw["race_on"] == "" else raw["race_on"] == "1",
                             obs_age_ms=_float(raw["wheel_age_ms"]), yaw_rate=_float(raw.get("yaw_rate")),
                             gear=None if raw.get("gear") in (None, "") else int(float(raw["gear"]))))
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
    at = _anchor(raws, end_time or _mtime(path))
    rows = [_row(time=at(raw["timestamp_ns"]), session=session, source="runtime", mode=raw["mode"],
                 steer_deg=_float(raw["actual_angle_deg"]), target_deg=_float(raw["target_angle_deg"]),
                 predicted_deg=_float(raw.get("requested_angle_deg")), torque=_float(raw["torque"]),
                 speed_mps=_float(raw.get("speed_mps")), obs_age_ms=_float(raw.get("observation_age_ms")))
            for raw in raws]
    return session, rows


def _mtime(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc)


def _anchor(raws, end_time):
    """Map monotonic nanoseconds onto wall clock so the last sample lands on end_time."""
    last_ns = int(raws[-1]["timestamp_ns"])
    return lambda ns: end_time + timedelta(microseconds=(int(ns) - last_ns) / 1000)


def _read_stream(path, name):
    with (path / f"{name}.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = SESSION_FIELDS[name]
        if (reader.fieldnames or [])[:len(required)] != required:
            raise ValueError(f"{name}.csv columns do not match the stream-v1 recorder")
        return list(reader)


def session_rows(path, session=None, end_time=None, allow_incomplete=False):
    """Rows from a stream-v1 session written by `forza_ai.runtime --record-session`.

    wheel.csv is the sample stream; its `control_mode` column is the training label (non-expert
    samples are written as `assist`), so the true mode comes from the latest preceding events.csv
    entry. Telemetry is joined causally: the latest sample at or before each wheel timestamp.
    Returns (session, wheel_rows, event_rows).
    """
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if not meta.get("completed") and not allow_incomplete:
        raise ValueError("session is not marked completed; pass --allow-incomplete to load it anyway")
    session = session or str(meta["session_id"])
    wheel = _read_stream(path, "wheel")
    telemetry = _read_stream(path, "telemetry")
    events = _read_stream(path, "events")
    if not wheel:
        return session, [], []
    at = _anchor(wheel, end_time or _mtime(path / "metadata.json"))
    rows, tele_i, event_i, current, mode = [], 0, 0, None, None
    for raw in wheel:
        ns = int(raw["timestamp_ns"])
        while tele_i < len(telemetry) and int(telemetry[tele_i]["timestamp_ns"]) <= ns:
            current = telemetry[tele_i]
            tele_i += 1
        while event_i < len(events) and int(events[event_i]["timestamp_ns"]) <= ns:
            mode = events[event_i]["control_mode"]
            event_i += 1
        rows.append(_row(time=at(ns), session=session, source="session", mode=mode or raw["control_mode"],
                         steer_deg=_float(raw["angle_deg"]), gas=_float(raw["throttle"]), brake=_float(raw["brake"]),
                         speed_mps=_float(current["speed_mps"]) if current else None,
                         race_on=None if current is None else current["is_race_on"] == "1",
                         rpm=_float(current["rpm"]) if current else None,
                         game_ms=int(current["game_timestamp_ms"]) if current else None))
    event_rows = [dict(time=at(raw["timestamp_ns"]), session=session, control_mode=raw["control_mode"],
                       training_mode=raw["training_mode"], expert=raw["expert"] == "1", reason=raw["reason"])
                  for raw in events]
    return session, rows, event_rows


def report_row(path, session):
    """Flatten a `--run-report` JSON (runtime summary with nested RunMetrics) into one run_reports row."""
    report = json.loads(Path(path).read_text())
    metrics = report.get("metrics", {})
    seconds = metrics.get("mode_seconds", {})
    routes = metrics.get("routes", {})
    return dict(session=session, actuation=report.get("actuation"), duration_s=metrics.get("duration_seconds"),
                ticks=metrics.get("ticks", report.get("ticks")), manual_s=seconds.get("manual"),
                assist_s=seconds.get("assist"), takeover_s=seconds.get("takeover"), fault_s=seconds.get("fault"),
                human_interventions=metrics.get("human_interventions"),
                interventions_per_assist_minute=metrics.get("interventions_per_assist_minute"),
                tracking_rmse_deg=metrics.get("tracking_rmse_deg"), max_abs_torque=metrics.get("max_abs_torque"),
                fault_entries=metrics.get("fault_entries"), routes_attempted=routes.get("attempts"),
                routes_completed=routes.get("completed"), routes_aborted=routes.get("aborted"),
                error=report.get("error"), report=json.dumps(report))


def _copy(cursor, table, columns, rows):
    with cursor.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row([row[column] for column in columns])


def copy_rows(conn, rows, replace=True, events=()):
    """Bulk load with COPY. By default an existing (session, source) is replaced so reloads are idempotent."""
    if not rows:
        return 0
    session, source = rows[0]["session"], rows[0]["source"]
    with conn.cursor() as cursor:
        if replace:
            cursor.execute("DELETE FROM wheel_samples WHERE session = %s AND source = %s", (session, source))
            cursor.execute("DELETE FROM control_events WHERE session = %s", (session,))
        _copy(cursor, "wheel_samples", COLUMNS, rows)
        if events:
            _copy(cursor, "control_events", EVENT_COLUMNS, events)
    return len(rows)


def upsert_report(conn, row):
    assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in REPORT_COLUMNS if column != "session")
    with conn.cursor() as cursor:
        cursor.execute(f"INSERT INTO run_reports ({', '.join(REPORT_COLUMNS)}, loaded_at) "
                       f"VALUES ({', '.join('%s' for _ in REPORT_COLUMNS)}, now()) "
                       f"ON CONFLICT (session) DO UPDATE SET {assignments}, loaded_at = now()",
                       [row[column] for column in REPORT_COLUMNS])
