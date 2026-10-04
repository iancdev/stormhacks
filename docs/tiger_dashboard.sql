-- Tiger Console "Data view" dashboard panels. Each query is one saved query / one
-- chart. Paste into Data view (toggle at the top right of Tiger Console), run,
-- click "Chart", then "Add to dashboard". Column named `time`/`bucket` is the x
-- axis; every other numeric column is a series. Replace 'shadow-v1' with a
-- Data view query variable ({{session}}) to make the whole dashboard switch runs.
-- Schedules (clock icon on the dashboard) refresh every panel automatically.

-- 1. Wheel trace: human steering vs AI prediction, 200 ms buckets. Line chart.
--    Series: wheel_deg (measured), ai_deg (predicted), takeover_deg (the wheel only while a
--    human had it; null elsewhere, so it draws as separate segments in its own color).
--    disagreement_deg is |wheel - ai| and is ~0 except during takeovers: in a Combo chart put
--    wheel_deg/ai_deg as lines on the left axis and disagreement_deg as bars on the right axis.
--    x is seconds into the run (t_s) so the axis is readable. Turn OFF Auto limit: 301 rows.
SELECT b.bucket                                                           AS time,
       round(extract(epoch FROM b.bucket - min(b.bucket) OVER ())::numeric, 1) AS t_s,
       b.wheel_deg, b.ai_deg, b.takeover_deg, b.disagreement_deg
FROM (
    SELECT time_bucket('200 milliseconds', time)                       AS bucket,
           round(avg(steer_deg)::numeric, 2)                          AS wheel_deg,
           round(avg(predicted_deg)::numeric, 2)                      AS ai_deg,
           round(avg(steer_deg) FILTER (WHERE mode = 'takeover')::numeric, 2) AS takeover_deg,
           round(abs(avg(steer_deg) - avg(predicted_deg))::numeric, 2) AS disagreement_deg
    FROM wheel_samples
    WHERE session = 'shadow-v1'
    GROUP BY 1
) b
ORDER BY t_s;

-- 2. Speed and pedals over the run. Line chart (secondary axis for km/h).
SELECT time_bucket('500 milliseconds', time)                       AS time,
       round((avg(speed_mps) * 3.6)::numeric, 1)                  AS kmh,
       round(avg(gas)::numeric, 2)                                AS gas,
       round(avg(brake)::numeric, 2)                              AS brake
FROM wheel_samples
WHERE session = 'shadow-v1' AND speed_mps IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- 3. Per-minute stability from the continuous aggregate (pre-computed; this is the
--    "instant dashboard" panel: it reads stability_1m, not the raw samples).
--    Bar or line chart, series = session.
SELECT bucket, session,
       round(steer_jitter_deg::numeric, 2)                        AS steer_jitter_deg,
       round(tracking_err_deg::numeric, 2)                        AS tracking_err_deg,
       round((100.0 * takeover_samples / samples)::numeric, 1)    AS takeover_pct,
       round(avg_kmh::numeric, 1)                                 AS avg_kmh
FROM stability_1m
ORDER BY bucket, session;

-- 4. Human vs AI by corner side during takeovers (H1). Bar chart.
--    human_over_ai_ratio > 1: the model understeers on that side.
SELECT session || ' / ' || CASE WHEN steer_deg < 0 THEN 'left' ELSE 'right' END AS corner,
       count(*)                                                   AS takeover_samples,
       round(avg(abs(steer_deg))::numeric, 1)                     AS human_abs_deg,
       round(avg(abs(predicted_deg))::numeric, 1)                 AS ai_abs_deg,
       round((avg(abs(steer_deg)) / nullif(avg(abs(predicted_deg)), 0))::numeric, 2) AS human_over_ai_ratio
FROM wheel_samples
WHERE source IN ('session', 'runtime') AND mode = 'takeover' AND predicted_deg IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- 5. Takeover share and disagreement per 20 km/h band (H2). Bar chart, x = kmh_from.
SELECT (width_bucket(speed_mps * 3.6, 0, 200, 10) - 1) * 20      AS kmh_from,
       count(*)                                                   AS samples,
       round((100.0 * count(*) FILTER (WHERE mode = 'takeover') / count(*))::numeric, 1) AS takeover_pct,
       round(avg(abs(predicted_deg - steer_deg))::numeric, 2)     AS disagreement_deg
FROM wheel_samples
WHERE session = 'shadow-v1' AND speed_mps IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- 6. What triggered each takeover (H6). Pie chart.
SELECT reason, count(*) AS takeovers
FROM control_events
WHERE control_mode = 'takeover'
GROUP BY reason
ORDER BY takeovers DESC;

-- 7. Run leaderboard: relational run summaries next to the sample stream. Table.
SELECT r.session, r.actuation,
       round(r.duration_s::numeric, 1)                            AS duration_s,
       r.human_interventions,
       round(r.interventions_per_assist_minute::numeric, 2)       AS per_assist_min,
       round(r.tracking_rmse_deg::numeric, 2)                     AS tracking_rmse_deg,
       s.samples,
       round(s.takeover_pct::numeric, 1)                          AS takeover_pct
FROM run_reports r
LEFT JOIN (
    SELECT session, count(*) AS samples,
           100.0 * count(*) FILTER (WHERE mode = 'takeover') / count(*) AS takeover_pct
    FROM wheel_samples GROUP BY session
) s USING (session)
ORDER BY r.interventions_per_assist_minute NULLS LAST, r.tracking_rmse_deg;

-- 8. Takeover time per run via the state_agg hyperfunction (TimescaleDB Toolkit).
--    One call over `mode` replaces a hand-written state machine. Bar chart or table.
--    Pair with the understeer ratio and its spread from stats_agg (tight = consistent).
WITH modes AS (
    SELECT session, state_agg(time, mode) AS agg
    FROM wheel_samples WHERE source IN ('session', 'runtime') GROUP BY session
),
ratio AS (
    SELECT session,
           average(stats_agg(abs(steer_deg)) FILTER (WHERE steer_deg < 0))
             / average(stats_agg(abs(predicted_deg)) FILTER (WHERE steer_deg < 0))       AS left_ratio,
           average(stats_agg(abs(steer_deg)) FILTER (WHERE steer_deg > 0))
             / average(stats_agg(abs(predicted_deg)) FILTER (WHERE steer_deg > 0))       AS right_ratio,
           stddev(stats_agg(abs(steer_deg) / nullif(abs(predicted_deg), 0)) FILTER (WHERE steer_deg < 0)) AS left_spread
    FROM wheel_samples
    WHERE mode = 'takeover' AND predicted_deg IS NOT NULL GROUP BY session
)
SELECT m.session,
       round(extract(epoch FROM duration_in(m.agg, 'takeover'))::numeric, 1) AS takeover_s,
       round(extract(epoch FROM duration_in(m.agg, 'assist'))::numeric, 1)   AS assist_s,
       (SELECT count(*) FROM control_events e WHERE e.session = m.session AND e.control_mode = 'takeover') AS grabs,
       round(r.left_ratio::numeric, 2)                                       AS left_human_over_ai,
       round(r.right_ratio::numeric, 2)                                      AS right_human_over_ai,
       round(r.left_spread::numeric, 2)                                      AS left_spread
FROM modes m LEFT JOIN ratio r USING (session)
ORDER BY m.session;

-- 9. Storage: rows and compression on the hypertable. Single-value / table panel.
SELECT (SELECT count(*) FROM wheel_samples)                       AS rows,
       pg_size_pretty(before_compression_total_bytes)             AS uncompressed,
       pg_size_pretty(after_compression_total_bytes)              AS compressed,
       round(100 * (1 - after_compression_total_bytes::numeric / before_compression_total_bytes), 1) AS saved_pct
FROM hypertable_compression_stats('wheel_samples');
