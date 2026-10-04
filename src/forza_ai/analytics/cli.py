"""forza-analytics: load recorder/runtime artifacts into Tiger Data and report driving metrics."""
import argparse
from datetime import datetime

from forza_ai.analytics import db, load, report


def _end_time(value):
    return datetime.fromisoformat(value).astimezone() if value else None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="forza-analytics", description=__doc__)
    parser.add_argument("--url", help=f"connection string (default: ${db.ENV_VAR} or .env)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="create tables and the continuous aggregate (idempotent)")

    recording = commands.add_parser("load-recording", help="load a standalone record.py recording directory")
    recording.add_argument("path")
    recording.add_argument("--append", action="store_true", help="keep existing rows for this session")

    session = commands.add_parser("load-session",
                                  help="load a stream-v1 session from `forza_ai.runtime --record-session`")
    session.add_argument("path")
    session.add_argument("--session", help="session name (default: metadata session_id)")
    session.add_argument("--end-time", help="ISO 8601 wall-clock time of the last sample (default: metadata mtime)")
    session.add_argument("--allow-incomplete", action="store_true", help="load a session not marked completed")
    session.add_argument("--append", action="store_true")

    run_report = commands.add_parser("load-report", help="load a `--run-report` JSON summary")
    run_report.add_argument("path")
    run_report.add_argument("--session", required=True, help="run name; reloading the same name replaces it")

    run = commands.add_parser("load-run", help="load a bounded test run's `--status-csv` (finite runs only)")
    run.add_argument("path")
    run.add_argument("--session", help="session name (default: CSV file stem)")
    run.add_argument("--end-time", help="ISO 8601 wall-clock time of the last tick (default: file mtime)")
    run.add_argument("--append", action="store_true")

    summary = commands.add_parser("report", help="print runs, per-session and per-minute metrics")
    summary.add_argument("--session")
    summary.add_argument("--events", action="store_true", help="also list control hand-offs and takeovers")
    args = parser.parse_args(argv)

    with db.connect(args.url) as conn:
        if args.command == "init":
            db.apply_schema(conn)
            print("schema ready: wheel_samples, control_events, run_reports, stability_1m")
        elif args.command == "load-recording":
            session, rows = load.recording_rows(args.path)
            count = load.copy_rows(conn, rows, replace=not args.append)
            db.refresh_aggregates(conn)
            print(f"loaded {count} recorder rows into session {session!r}")
        elif args.command == "load-session":
            session, rows, events = load.session_rows(args.path, args.session, _end_time(args.end_time),
                                                      args.allow_incomplete)
            count = load.copy_rows(conn, rows, replace=not args.append, events=events)
            db.refresh_aggregates(conn)
            print(f"loaded {count} wheel samples and {len(events)} control events into session {session!r}")
        elif args.command == "load-report":
            load.upsert_report(conn, load.report_row(args.path, args.session))
            print(f"stored run report for {args.session!r}")
        elif args.command == "load-run":
            session, rows = load.runtime_rows(args.path, args.session, _end_time(args.end_time))
            count = load.copy_rows(conn, rows, replace=not args.append)
            db.refresh_aggregates(conn)
            print(f"loaded {count} runtime rows into session {session!r}")
        elif args.command == "report":
            print("Runs (run_reports)\n" + report.format_table(*report.fetch(conn, report.RUNS)))
            print("\nSessions\n" + report.format_table(*report.fetch(conn, report.SESSION_SUMMARY)))
            print("\nPer minute (stability_1m)\n"
                  + report.format_table(*report.fetch(conn, report.PER_MINUTE, session=args.session)))
            if args.events:
                print("\nControl hand-offs (control_events)\n"
                      + report.format_table(*report.fetch(conn, report.CONTROL_EVENTS, session=args.session)))
                print("\nTakeover onsets in status-CSV runs\n"
                      + report.format_table(*report.fetch(conn, report.TAKEOVER_EVENTS, session=args.session)))


if __name__ == "__main__":
    main()
