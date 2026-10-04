"""Hypothesis queries: where and why the AI needs a human, and which knob to turn.

Each entry is (title, hint, sql). Every query takes a %(session)s parameter (NULL = all AI sessions)
and only reads AI-driven sources ('session', 'runtime'); H5 also reads human 'recorder' laps.
"""

_AI = "source IN ('session', 'runtime')"
_SESSION = "(%(session)s::text IS NULL OR session = %(session)s)"

H1_UNDERSTEER = f"""
SELECT session,
       CASE WHEN steer_deg < 0 THEN 'left' ELSE 'right' END                 AS corner,
       count(*)                                                             AS takeover_samples,
       round(avg(abs(steer_deg))::numeric, 1)                               AS human_abs_deg,
       round(avg(abs(predicted_deg))::numeric, 1)                           AS ai_abs_deg,
       round((avg(abs(steer_deg)) / nullif(avg(abs(predicted_deg)), 0))::numeric, 2)
                                                                            AS human_over_ai_ratio
FROM wheel_samples
WHERE {_AI} AND mode = 'takeover' AND predicted_deg IS NOT NULL AND {_SESSION}
GROUP BY session, corner
ORDER BY session, corner
"""

H2_SPEED = f"""
SELECT session,
       (width_bucket(speed_mps * 3.6, 0, 200, 10) - 1) * 20                AS kmh_from,
       count(*)                                                             AS samples,
       round((100.0 * count(*) FILTER (WHERE mode = 'takeover') / count(*))::numeric, 1)
                                                                            AS takeover_pct,
       round(avg(abs(predicted_deg - steer_deg))::numeric, 2)               AS disagreement_deg
FROM wheel_samples
WHERE {_AI} AND speed_mps IS NOT NULL AND {_SESSION}
GROUP BY session, kmh_from
ORDER BY session, kmh_from
"""

H3_DIRECTION = f"""
SELECT e.session,
       CASE WHEN s.steer_deg < -3 THEN 'left' WHEN s.steer_deg > 3 THEN 'right' ELSE 'straight' END AS corner,
       count(*)                                                             AS takeovers,
       round(avg(s.speed_mps * 3.6)::numeric, 1)                            AS avg_kmh
FROM control_events e
LEFT JOIN LATERAL (
    SELECT steer_deg, speed_mps FROM wheel_samples w
    WHERE w.session = e.session AND w.time <= e.time ORDER BY w.time DESC LIMIT 1
) s ON true
WHERE e.control_mode = 'takeover' AND (%(session)s::text IS NULL OR e.session = %(session)s)
GROUP BY e.session, corner
ORDER BY e.session, takeovers DESC
"""

# Correlate the AI prediction with the human's angle shifted by k samples in human-driven rows.
# The lag with the smallest error is how late (positive) or early (negative) the AI reacts.
H4_LAG = f"""
WITH human AS (
    SELECT session, time, steer_deg, predicted_deg,
           row_number() OVER (PARTITION BY session ORDER BY time) AS n
    FROM wheel_samples
    WHERE {_AI} AND mode IN ('manual', 'takeover') AND predicted_deg IS NOT NULL AND {_SESSION}
),
lags AS (SELECT generate_series(-30, 30, 5) AS k)
SELECT a.session, lags.k * 10 AS ai_lag_ms,
       count(*)                                                             AS pairs,
       round(avg(abs(a.predicted_deg - b.steer_deg))::numeric, 2)           AS mean_abs_error_deg
FROM human a
CROSS JOIN lags
JOIN human b ON b.session = a.session AND b.n = a.n - lags.k
GROUP BY a.session, lags.k
ORDER BY a.session, mean_abs_error_deg
"""

H5_COVERAGE = f"""
WITH cells AS (
    SELECT session, source, mode,
           width_bucket(abs(steer_deg), 0, 90, 6) AS steer_band,
           width_bucket(speed_mps * 3.6, 0, 200, 5) AS speed_band
    FROM wheel_samples
    WHERE speed_mps IS NOT NULL AND (source = 'recorder' OR ({_AI} AND {_SESSION}))
),
train AS (SELECT steer_band, speed_band, count(*) AS training_samples FROM cells WHERE source = 'recorder' GROUP BY 1, 2),
fail  AS (SELECT steer_band, speed_band, count(*) AS takeover_samples FROM cells WHERE mode = 'takeover' AND source <> 'recorder' GROUP BY 1, 2)
SELECT (fail.steer_band - 1) * 15 AS abs_steer_from_deg,
       (fail.speed_band - 1) * 40 AS kmh_from,
       fail.takeover_samples,
       coalesce(train.training_samples, 0) AS training_samples,
       round((fail.takeover_samples::numeric / greatest(coalesce(train.training_samples, 0), 1)), 3) AS takeovers_per_training_sample
FROM fail LEFT JOIN train USING (steer_band, speed_band)
ORDER BY takeovers_per_training_sample DESC, fail.takeover_samples DESC
LIMIT 10
"""

H6_TRIGGERS = """
SELECT session, reason, count(*) AS takeovers
FROM control_events
WHERE control_mode = 'takeover' AND (%(session)s::text IS NULL OR session = %(session)s)
GROUP BY session, reason
ORDER BY session, takeovers DESC
"""

HYPOTHESES = [
    ("H1 Understeer: human vs AI angle during takeovers, by corner side",
     "ratio > 1 means the model understeers; the ratio is a candidate --steering-gain", H1_UNDERSTEER),
    ("H2 Speed: takeover share and disagreement per 20 km/h band",
     "a cliff suggests --max-speed-kmh / --corner-speed-kmh / --brake-gain", H2_SPEED),
    ("H3 Direction: takeovers by corner side at the moment of hand-off",
     "asymmetry suggests DAgger-recording that side and retraining", H3_DIRECTION),
    ("H4 Lag: AI prediction vs human angle shifted in time (best row first)",
     "positive ai_lag_ms means the AI reacts late; consider --label-offset-ms or faster target slew", H4_LAG),
    ("H5 Coverage: takeover cells with the least human training data",
     "record more laps in these steer/speed cells", H5_COVERAGE),
    ("H6 Triggers: what caused each takeover",
     "grab-dominated means controller/motor limits; button-dominated means model errors", H6_TRIGGERS),
]
