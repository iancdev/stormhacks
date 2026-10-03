"""forza-analytics: load recorder/runtime CSV artifacts into Tiger Data and report driving metrics."""
import argparse
from datetime import datetime

from forza_ai.analytics import db, load, report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="forza-analytics", description=__doc__)
    parser.add_argument("--url", help=f"connection string (default: ${db.ENV_VAR} or .env)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="create the hypertable and continuous aggregate (idempotent)")
    recording = commands.add_parser("load-recording", help="load a record.py recording directory")
    recording.add_argument("path")
    recording.add_argument("--append", action="store_true", help="keep existing rows for this session")
    run = commands.add_parser("load-run", help="load a `forza_ai.runtime --status-csv` file")
    run.add_argument("path")
    run.add_argument("--session", help="session name (default: CSV file stem)")
    run.add_argument("--end-time", help="ISO 8601 wall-clock time of the last tick (default: file mtime)")
    run.add_argument("--append", action="store_true")
    summary = commands.add_parser("report", help="print per-session and per-minute metrics")
    summary.add_argument("--session")
    summary.add_argument("--events", action="store_true", help="also list takeover events")
    args = parser.parse_args(argv)

    with db.connect(args.url) as conn:
        if args.command == "init":
            db.apply_schema(conn)
            print("schema ready: wheel_samples (hypertable), stability_1m (continuous aggregate)")
        elif args.command == "load-recording":
            session, rows = load.recording_rows(args.path)
            count = load.copy_rows(conn, rows, replace=not args.append)
            db.refresh_aggregates(conn)
            print(f"loaded {count} recorder rows into session {session!r}")
        elif args.command == "load-run":
            end = datetime.fromisoformat(args.end_time).astimezone() if args.end_time else None
            session, rows = load.runtime_rows(args.path, args.session, end)
            count = load.copy_rows(conn, rows, replace=not args.append)
            db.refresh_aggregates(conn)
            print(f"loaded {count} runtime rows into session {session!r}")
        elif args.command == "report":
            print("Sessions\n" + report.format_table(*report.fetch(conn, report.SESSION_SUMMARY)))
            print("\nPer minute (stability_1m)\n"
                  + report.format_table(*report.fetch(conn, report.PER_MINUTE, session=args.session)))
            if args.events:
                print("\nTakeover events\n"
                      + report.format_table(*report.fetch(conn, report.TAKEOVER_EVENTS, session=args.session)))


if __name__ == "__main__":
    main()
