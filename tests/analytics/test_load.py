"""CSV-to-row conversion and schema parsing without a database."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from forza_ai.analytics import db, load, report


def write_recording(root, session="20261003_161200", frames=3):
    path = Path(root) / session
    (path / "frames").mkdir(parents=True)
    with (path / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(load.RECORDER_COLUMNS)
        for index in range(frames):
            writer.writerow([index, 0, f"{index / 30:.4f}", -1204 + index, f"{(-1204 + index) / 73:.2f}",
                             "0.0000", "0.6210", "1.2", "24.310", "1", "-12", "3.4"])
    (path / "meta.json").write_text(json.dumps({"session": session, "frames": frames, "segments": 1}))
    return path


def write_status(root, rows):
    path = Path(root) / "shadow-run.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class RecordingRowsTests(unittest.TestCase):
    def test_rows_follow_recorder_labels_and_anchor_to_session_name(self):
        with tempfile.TemporaryDirectory() as root:
            session, rows = load.recording_rows(write_recording(root))
        self.assertEqual(session, "20261003_161200")
        self.assertEqual(len(rows), 3)
        first, last = rows[0], rows[-1]
        self.assertEqual(list(first), load.COLUMNS)
        self.assertEqual(first["time"], datetime(2026, 10, 3, 16, 12, 0).astimezone())
        self.assertAlmostEqual((last["time"] - first["time"]).total_seconds(), 2 / 30, places=4)
        self.assertEqual((first["source"], first["mode"], first["image_path"]), ("recorder", "manual", "frames/000000.jpg"))
        self.assertEqual(first["steer_deg"], -16.49)
        self.assertEqual((first["speed_mps"], first["gas"], first["brake"], first["race_on"]), (24.31, 0.621, 0.0, True))
        self.assertIsNone(first["target_deg"])

    def test_rejects_foreign_columns(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_recording(root)
            (path / "labels.csv").write_text("frame,steer\n0,1\n")
            with self.assertRaises(ValueError):
                load.recording_rows(path)


class RuntimeRowsTests(unittest.TestCase):
    def test_rows_anchor_last_tick_to_end_time(self):
        status = [{"mode": "assist", "reason": "ok", "target_angle_deg": "5.0", "actual_angle_deg": "4.2",
                   "torque": "0.05", "timestamp_ns": str(1_000_000_000 + i * 10_000_000), "speed_mps": "",
                   "requested_angle_deg": "5.5", "observation_age_ms": "12.5", "input_status": "ready"}
                  for i in range(3)]
        status[-1]["mode"] = "takeover"
        end = datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as root:
            session, rows = load.runtime_rows(write_status(root, status), end_time=end)
        self.assertEqual(session, "shadow-run")
        self.assertEqual(rows[-1]["time"], end)
        self.assertEqual(rows[0]["time"], end - timedelta(milliseconds=20))
        self.assertEqual((rows[0]["source"], rows[0]["mode"], rows[-1]["mode"]), ("runtime", "assist", "takeover"))
        self.assertEqual((rows[0]["steer_deg"], rows[0]["target_deg"], rows[0]["predicted_deg"]), (4.2, 5.0, 5.5))
        self.assertIsNone(rows[0]["speed_mps"])
        self.assertEqual(rows[0]["obs_age_ms"], 12.5)

    def test_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_status(root, [{"mode": "assist", "timestamp_ns": "1"}])
            with self.assertRaises(ValueError):
                load.runtime_rows(path)


class FakeCopy:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_row(self, row):
        self.sink.append(row)


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append(("execute", " ".join(sql.split()), params))

    def copy(self, sql):
        self.log.append(("copy", sql))
        self.copied = []
        self.log.append(("rows", self.copied))
        return FakeCopy(self.copied)


class FakeConnection:
    def __init__(self):
        self.log = []

    def cursor(self):
        return FakeCursor(self.log)


class CopyRowsTests(unittest.TestCase):
    def test_replace_deletes_session_then_copies_in_column_order(self):
        conn = FakeConnection()
        rows = [load._row(time=datetime(2026, 10, 3, tzinfo=timezone.utc), session="s", source="runtime", steer_deg=1.0)]
        self.assertEqual(load.copy_rows(conn, rows), 1)
        kinds = [entry[0] for entry in conn.log]
        self.assertEqual(kinds, ["execute", "copy", "rows"])
        self.assertEqual(conn.log[0][2], ("s", "runtime"))
        self.assertIn("COPY wheel_samples (time, session, source", conn.log[1][1])
        self.assertEqual(conn.log[2][1][0][load.COLUMNS.index("steer_deg")], 1.0)

    def test_empty_and_append(self):
        conn = FakeConnection()
        self.assertEqual(load.copy_rows(conn, []), 0)
        load.copy_rows(conn, [load._row(time=None, session="s", source="recorder")], replace=False)
        self.assertEqual([entry[0] for entry in conn.log], ["copy", "rows"])


class SchemaTests(unittest.TestCase):
    def test_schema_splits_into_statements_without_comments(self):
        parts = db.statements()
        self.assertTrue(parts[0].startswith("CREATE TABLE IF NOT EXISTS wheel_samples"))
        self.assertTrue(any("create_hypertable" in part for part in parts))
        self.assertTrue(any("timescaledb.continuous" in part for part in parts))
        self.assertTrue(any("add_continuous_aggregate_policy" in part for part in parts))
        self.assertFalse(any("--" in part for part in parts))
        self.assertIn("obs_age_ms", parts[0])  # an inline comment with ';' must not split the table

    def test_schema_columns_match_loader(self):
        table = db.statements()[0]
        for column in load.COLUMNS:
            self.assertIn(f"\n    {column} ", table + " ", column)

    def test_database_url_resolution(self):
        with tempfile.TemporaryDirectory() as root:
            env = Path(root) / ".env"
            env.write_text("OTHER=1\nTIGER_DATA_URL='postgres://from-env'\n")
            self.assertEqual(db.database_url(None, dotenv=env), "postgres://from-env")
            self.assertEqual(db.database_url("postgres://explicit", dotenv=env), "postgres://explicit")
            with self.assertRaises(SystemExit):
                db.database_url(None, dotenv=Path(root) / "missing")


class ReportFormatTests(unittest.TestCase):
    def test_format_table_pads_columns(self):
        text = report.format_table(["session", "n"], [("a", 1), ("longer", None)])
        self.assertEqual(text.splitlines()[0], "session  n")
        self.assertEqual(text.splitlines()[-1], "longer    ")


if __name__ == "__main__":
    unittest.main()
