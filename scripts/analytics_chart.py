"""Render the Tiger Data analytics demo chart (read-only; needs the `analytics` extra).

Top: one AI session's measured wheel angle against the AI's predicted angle, with human
takeover intervals shaded from control_events. Bottom: stability_1m takeover seconds and
steering jitter per minute for every loaded session, human laps beside AI runs.

    python scripts/analytics_chart.py --ai-session sim-ai-run-001 --out docs/assets/tiger-analytics.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forza_ai.analytics import db, report

TRACE = """
SELECT time, steer_deg, predicted_deg
FROM wheel_samples
WHERE session = %(session)s AND source IN ('session', 'runtime')
ORDER BY time
"""

INTERVALS = """
SELECT time, control_mode,
       lead(time) OVER (ORDER BY time) AS until
FROM control_events
WHERE session = %(session)s
ORDER BY time
"""

MINUTES = """
SELECT session, source, bucket, takeover_samples, steer_jitter_deg, samples
FROM stability_1m
ORDER BY session, bucket
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ai-session", required=True, help="session name for the top trace panel")
    parser.add_argument("--out", default="docs/assets/tiger-analytics.png")
    parser.add_argument("--url")
    args = parser.parse_args(argv)

    with db.connect(args.url) as conn:
        _, trace = report.fetch(conn, TRACE, session=args.ai_session)
        _, intervals = report.fetch(conn, INTERVALS, session=args.ai_session)
        _, minutes = report.fetch(conn, MINUTES)
    if not trace:
        raise SystemExit(f"no AI samples for session {args.ai_session!r}")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [3, 2]})
    start = trace[0][0]
    seconds = [(row[0] - start).total_seconds() for row in trace]
    top.plot(seconds, [row[1] for row in trace], color="#1f77b4", linewidth=0.9, label="measured wheel angle (human or AI)")
    top.plot(seconds, [row[2] for row in trace], color="#ff7f0e", linewidth=0.9, alpha=0.85, label="AI predicted angle")
    shaded = False
    for at, mode, until in intervals:
        if mode == "takeover":
            end = (until or trace[-1][0]) - start
            top.axvspan((at - start).total_seconds(), end.total_seconds(), color="#d62728", alpha=0.18,
                        label=None if shaded else "human takeover")
            shaded = True
    top.set_title(f"Session {args.ai_session}: what the AI wanted vs what the wheel did (Tiger Data wheel_samples + control_events)")
    top.set_xlabel("seconds into run")
    top.set_ylabel("degrees (right positive)")
    top.legend(loc="upper right", fontsize=9)
    top.grid(alpha=0.3)

    sessions = sorted({(row[0], row[1]) for row in minutes})
    width = 0.8 / max(len(sessions), 1)
    jitter_axis = bottom.twinx()
    for index, (session, source) in enumerate(sessions):
        rows = [row for row in minutes if row[0] == session and row[1] == source]
        xs = [i + index * width for i in range(len(rows))]
        takeover_s = [row[3] / max(row[5] / 60.0, 1e-9) for row in rows]  # takeover samples -> seconds per minute
        bottom.bar(xs, takeover_s, width=width, label=f"{session} ({source}) takeover s/min", alpha=0.8)
        jitter_axis.plot([x + width / 2 for x in xs], [row[4] for row in rows], marker="o", linewidth=1,
                         label=f"{session} jitter")
    bottom.set_title("stability_1m continuous aggregate: human takeover seconds per minute (bars) and steering jitter (lines)")
    bottom.set_xlabel("minute of session")
    bottom.set_ylabel("takeover seconds / minute")
    jitter_axis.set_ylabel("steering jitter (deg, stddev)")
    bottom.legend(loc="upper center", bbox_to_anchor=(0.28, -0.22), ncol=1, fontsize=8, frameon=False)
    jitter_axis.legend(loc="upper center", bbox_to_anchor=(0.75, -0.22), ncol=1, fontsize=8, frameon=False)
    bottom.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out} ({len(trace)} trace samples, {len(sessions)} sessions in the per-minute panel)")


if __name__ == "__main__":
    main()
