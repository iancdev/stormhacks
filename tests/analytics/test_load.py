"""CSV-to-row conversion and schema parsing without a database."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from forza_ai.analytics import db, load, report


EXTENDED = ["race_time", "distance", "yaw_rate", "game_ms", "gear", "car_ordinal", "car_class", "car_pi"]


def write_recording(root, session="20261003_161200", frames=3, extended=False):
    path = Path(root) / session
    (path / "frames").mkdir(parents=True)
    with (path / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(load.RECORDER_COLUMNS + (EXTENDED if extended else []))
        for index in range(frames):
            writer.writerow([index, 0, f"{index / 30:.4f}", -1204 + index, f"{(-1204 + index) / 73:.2f}",
                             "0.0000", "0.6210", "1.2", "24.310", "1", "-12", "3.4"]
                            + (["12.5", "310.2", "-0.0421", "812345", "4", "2345", "6", "740"] if extended else []))
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
        self.assertIsNone(first["yaw_rate"])
        self.assertIsNone(first["gear"])

    def test_accepts_newer_recorder_diagnostic_columns(self):
        with tempfile.TemporaryDirectory() as root:
            _, rows = load.recording_rows(write_recording(root, extended=True))
        self.assertEqual((rows[0]["yaw_rate"], rows[0]["gear"], rows[0]["steer_deg"]), (-0.0421, 4, -16.49))

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


def write_session(root, completed=True):
    """Tiny stream-v1 session: 4 wheel samples at 10 ms, 2 telemetry samples, 2 control events."""
    path = Path(root) / "session-001"
    (path / "images").mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({"schema_version": 1, "session_id": "run-abc",
                                                     "clock": "monotonic_ns", "completed": completed}))
    streams = {
        # training label is 'assist' for non-expert samples; the true mode is in events.csv
        "wheel": [(1_000_000_000, "3.0", "0.5", "0.0", "assist"),
                  (1_010_000_000, "3.5", "0.5", "0.0", "assist"),
                  (1_020_000_000, "9.0", "0.2", "0.1", "takeover"),
                  (1_030_000_000, "12.0", "0.2", "0.1", "takeover")],
        "telemetry": [(1_005_000_000, "20.0", "1", "500", "3000.0", "10"),
                      (1_025_000_000, "21.0", "1", "520", "3100.0", "40")],
        "events": [(1_000_000_000, "assist", "assist", "0", "engaged"),
                   (1_020_000_000, "takeover", "takeover", "1", "takeover_button")],
    }
    for name, rows in streams.items():
        with (path / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(load.SESSION_FIELDS[name])
            writer.writerows(rows)
    (path / "frames.csv").write_text("frame_id,image_path,capture_time_ns\n")
    return path


class SessionRowsTests(unittest.TestCase):
    END = datetime(2026, 10, 4, 1, 0, tzinfo=timezone.utc)

    def test_wheel_rows_join_telemetry_causally_and_take_mode_from_events(self):
        with tempfile.TemporaryDirectory() as root:
            session, rows, events = load.session_rows(write_session(root), end_time=self.END)
        self.assertEqual(session, "run-abc")
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["source"] for row in rows], ["session"] * 4)
        self.assertEqual(rows[-1]["time"], self.END)
        self.assertEqual(rows[0]["time"], self.END - timedelta(milliseconds=30))
        # first wheel sample precedes all telemetry: no speed yet
        self.assertIsNone(rows[0]["speed_mps"])
        self.assertIsNone(rows[0]["race_on"])
        # 1.010 s sees telemetry @1.005; 1.020 s still sees @1.005; 1.030 s sees @1.025
        self.assertEqual([row["speed_mps"] for row in rows[1:]], [20.0, 20.0, 21.0])
        self.assertEqual((rows[1]["rpm"], rows[1]["game_ms"], rows[1]["race_on"]), (3000.0, 500, True))
        self.assertEqual([row["mode"] for row in rows], ["assist", "assist", "takeover", "takeover"])
        self.assertEqual((rows[2]["steer_deg"], rows[2]["gas"], rows[2]["brake"]), (9.0, 0.2, 0.1))
        self.assertIsNone(rows[0]["target_deg"])

    def test_events_rows(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, events = load.session_rows(write_session(root), end_time=self.END)
        self.assertEqual(len(events), 2)
        self.assertEqual(list(events[0]), load.EVENT_COLUMNS)
        self.assertEqual(events[1]["time"], self.END - timedelta(milliseconds=10))
        self.assertEqual((events[1]["control_mode"], events[1]["expert"], events[1]["reason"]),
                         ("takeover", True, "takeover_button"))
        self.assertFalse(events[0]["expert"])

    def test_true_mode_overrides_training_label(self):
        """A manual sample not declared expert is written as 'assist' in wheel.csv; events say 'manual'."""
        with tempfile.TemporaryDirectory() as root:
            path = write_session(root)
            (path / "events.csv").write_text("timestamp_ns,control_mode,training_mode,expert,reason\n"
                                             "1000000000,manual,assist,0,unmarked_manual\n")
            _, rows, _ = load.session_rows(path, end_time=self.END)
        self.assertEqual([row["mode"] for row in rows], ["manual"] * 4)

    def test_incomplete_session_refused_unless_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_session(root, completed=False)
            with self.assertRaises(ValueError):
                load.session_rows(path)
            session, rows, _ = load.session_rows(path, session="override", end_time=self.END, allow_incomplete=True)
        self.assertEqual((session, len(rows)), ("override", 4))

    def test_rejects_foreign_stream_columns(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_session(root)
            (path / "telemetry.csv").write_text("timestamp_ns,speed\n1,2\n")
            with self.assertRaises(ValueError):
                load.session_rows(path, end_time=self.END)


class ReportRowTests(unittest.TestCase):
    REPORT = {
        "ticks": 1200, "mode": "assist", "actuation": "direct_vjoy", "error": None, "hardware_verified": False,
        "metrics": {
            "schema_version": 1, "duration_seconds": 12.0, "ticks": 1200,
            "mode_seconds": {"manual": 2.0, "assist": 8.0, "takeover": 1.5, "fault": 0.5, "unknown": 0.0},
            "human_interventions": 2, "interventions_per_assist_minute": 15.0,
            "tracking_rmse_deg": 3.25, "tracking_samples": 800, "max_abs_torque": 0.12, "fault_entries": 1,
            "routes": {"attempts": 1, "completed": 1, "aborted": 0},
        },
    }

    def test_flattens_runtime_summary_and_nested_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "report.json"
            path.write_text(json.dumps(self.REPORT))
            row = load.report_row(path, "run-7")
        self.assertEqual(list(row), load.REPORT_COLUMNS)
        self.assertEqual((row["session"], row["actuation"], row["duration_s"], row["ticks"]), ("run-7", "direct_vjoy", 12.0, 1200))
        self.assertEqual((row["manual_s"], row["assist_s"], row["takeover_s"], row["fault_s"]), (2.0, 8.0, 1.5, 0.5))
        self.assertEqual((row["human_interventions"], row["interventions_per_assist_minute"]), (2, 15.0))
        self.assertEqual((row["tracking_rmse_deg"], row["max_abs_torque"], row["fault_entries"]), (3.25, 0.12, 1))
        self.assertEqual((row["routes_attempted"], row["routes_completed"], row["routes_aborted"]), (1, 1, 0))
        self.assertIsNone(row["error"])
        self.assertEqual(json.loads(row["report"]), self.REPORT)

    def test_missing_metrics_yield_nulls_not_errors(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "report.json"
            path.write_text(json.dumps({"ticks": 5, "error": "RuntimeError: wheel lost"}))
            row = load.report_row(path, "broken")
        self.assertEqual((row["ticks"], row["error"], row["actuation"]), (5, "RuntimeError: wheel lost", None))
        self.assertIsNone(row["interventions_per_assist_minute"])


class UpsertReportTests(unittest.TestCase):
    def test_upsert_binds_every_column_and_replaces_on_conflict(self):
        conn = FakeConnection()
        row = dict.fromkeys(load.REPORT_COLUMNS)
        row.update(session="run-1", report="{}")
        load.upsert_report(conn, row)
        kind, sql, params = conn.log[0]
        self.assertEqual(kind, "execute")
        self.assertIn("INSERT INTO run_reports (session, actuation", sql)
        self.assertIn("ON CONFLICT (session) DO UPDATE SET actuation = EXCLUDED.actuation", sql)
        self.assertNotIn("session = EXCLUDED.session", sql)
        self.assertEqual(len(params), len(load.REPORT_COLUMNS))
        self.assertEqual(params[0], "run-1")


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
        self.assertEqual(kinds, ["execute", "execute", "copy", "rows"])
        self.assertEqual(conn.log[0][2], ("s", "runtime"))
        self.assertIn("DELETE FROM control_events", conn.log[1][1])
        self.assertIn("COPY wheel_samples (time, session, source", conn.log[2][1])
        self.assertEqual(conn.log[3][1][0][load.COLUMNS.index("steer_deg")], 1.0)

    def test_events_are_copied_after_samples(self):
        conn = FakeConnection()
        rows = [load._row(time=None, session="s", source="session")]
        events = [dict(time=None, session="s", control_mode="takeover", training_mode="takeover", expert=True, reason="button")]
        load.copy_rows(conn, rows, events=events)
        copies = [entry[1] for entry in conn.log if entry[0] == "copy"]
        self.assertEqual(len(copies), 2)
        self.assertIn("COPY control_events (time, session, control_mode, training_mode, expert, reason)", copies[1])
        self.assertEqual(conn.log[-1][1][0][2], "takeover")

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
        parts = db.statements()
        table = parts[0]
        added = {part.split("ADD COLUMN IF NOT EXISTS ")[1].split()[0]
                 for part in parts if "ADD COLUMN IF NOT EXISTS" in part}
        for column in load.COLUMNS:
            self.assertTrue(f"\n    {column} " in table + " " or column in added, column)
        events_table = next(part for part in parts if part.startswith("CREATE TABLE IF NOT EXISTS control_events"))
        for column in load.EVENT_COLUMNS:
            self.assertIn(f"\n    {column} ", events_table + " ", column)
        reports_table = next(part for part in parts if part.startswith("CREATE TABLE IF NOT EXISTS run_reports"))
        for column in load.REPORT_COLUMNS:
            self.assertIn(f"\n    {column} ", reports_table + " ", column)
        self.assertEqual(sum("create_hypertable" in part for part in parts), 2)

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
