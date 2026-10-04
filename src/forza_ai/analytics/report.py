"""Canned SQL for the metrics named in README milestone 6."""

SESSION_SUMMARY = """
SELECT session,
       source,
       min(time)                                                       AS started,
       extract(epoch FROM max(time) - min(time))                       AS duration_s,
       count(*)                                                        AS samples,
       round(avg(abs(steer_deg))::numeric, 2)                          AS mean_abs_steer_deg,
       round(stddev_samp(steer_deg)::numeric, 2)                       AS steer_jitter_deg,
       round(avg(abs(target_deg - steer_deg))::numeric, 2)             AS tracking_err_deg,
       round((100.0 * count(*) FILTER (WHERE mode = 'takeover') / count(*))::numeric, 1)
                                                                       AS takeover_pct,
       round((avg(speed_mps) * 3.6)::numeric, 1)                       AS avg_kmh
FROM wheel_samples
GROUP BY session, source
ORDER BY started
"""

PER_MINUTE = """
SELECT session, source, bucket, samples,
       round(mean_abs_steer_deg::numeric, 2) AS mean_abs_steer_deg,
       round(steer_jitter_deg::numeric, 2)   AS steer_jitter_deg,
       round(tracking_err_deg::numeric, 2)   AS tracking_err_deg,
       takeover_samples,
       round(avg_kmh::numeric, 1)            AS avg_kmh
FROM stability_1m
WHERE (%(session)s::text IS NULL OR session = %(session)s)
ORDER BY session, bucket
"""

RUNS = """
SELECT session, actuation, loaded_at,
       round(duration_s::numeric, 1)                      AS duration_s,
       round(assist_s::numeric, 1)                        AS assist_s,
       human_interventions                                AS interventions,
       round(interventions_per_assist_minute::numeric, 2) AS per_assist_min,
       round(tracking_rmse_deg::numeric, 2)               AS tracking_rmse_deg,
       routes_completed || '/' || routes_attempted        AS routes,
       fault_entries,
       error
FROM run_reports
ORDER BY loaded_at
"""

CONTROL_EVENTS = """
SELECT e.session, e.time, e.control_mode, e.expert, e.reason,
       round(s.steer_deg::numeric, 1) AS steer_deg,
       round((s.speed_mps * 3.6)::numeric, 1) AS kmh
FROM control_events e
LEFT JOIN LATERAL (
    SELECT steer_deg, speed_mps FROM wheel_samples w
    WHERE w.session = e.session AND w.time <= e.time
    ORDER BY w.time DESC LIMIT 1
) s ON true
WHERE (%(session)s::text IS NULL OR e.session = %(session)s)
ORDER BY e.session, e.time
"""

TAKEOVER_EVENTS = """
SELECT session, time, steer_deg, target_deg, speed_mps * 3.6 AS kmh
FROM (
    SELECT *, lag(mode) OVER (PARTITION BY session ORDER BY time) AS previous_mode
    FROM wheel_samples
    WHERE source IN ('runtime', 'session')
) ticks
WHERE mode = 'takeover' AND previous_mode IS DISTINCT FROM 'takeover'
  AND (%(session)s::text IS NULL OR session = %(session)s)
ORDER BY session, time
"""


def fetch(conn, sql, **params):
    with conn.cursor() as cursor:
        cursor.execute(sql, params or None)
        return [description.name for description in cursor.description], cursor.fetchall()


def format_table(columns, rows):
    cells = [[str(value) if value is not None else "" for value in row] for row in rows]
    widths = [max(len(column), *(len(row[i]) for row in cells)) if cells else len(column)
              for i, column in enumerate(columns)]
    line = lambda values: "  ".join(value.ljust(width) for value, width in zip(values, widths))
    return "\n".join([line(columns), line(["-" * width for width in widths]), *map(line, cells)])
