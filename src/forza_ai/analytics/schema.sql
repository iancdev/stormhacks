-- Tiger Data (TimescaleDB) analytics schema. Analytics only: never read by the
-- control loop, training, or inference. Each statement is executed separately
-- in autocommit mode; continuous aggregates cannot be created in a transaction.

CREATE TABLE IF NOT EXISTS wheel_samples (
    time          TIMESTAMPTZ      NOT NULL,
    session       TEXT             NOT NULL,
    source        TEXT             NOT NULL,  -- 'recorder' (human laps) or 'runtime' (controller/AI runs)
    segment       INTEGER,
    frame_id      BIGINT,
    image_path    TEXT,                       -- pointer to the JPEG on disk; pixels are never stored here
    mode          TEXT,                       -- manual | assist | takeover
    steer_deg     DOUBLE PRECISION,           -- measured physical wheel angle, right positive
    target_deg    DOUBLE PRECISION,           -- limited controller target (runtime only)
    predicted_deg DOUBLE PRECISION,           -- raw policy output (runtime only)
    torque        DOUBLE PRECISION,           -- normalized [-1, 1] (runtime only)
    speed_mps     DOUBLE PRECISION,
    gas           DOUBLE PRECISION,
    brake         DOUBLE PRECISION,
    race_on       BOOLEAN,
    obs_age_ms    DOUBLE PRECISION,           -- wheel sample age (recorder) or observation age (runtime)
    yaw_rate      DOUBLE PRECISION,           -- game yaw rate (newer recorder versions only)
    gear          INTEGER                     -- in-game gear (newer recorder versions only)
);

ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS yaw_rate DOUBLE PRECISION;
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS gear INTEGER;

SELECT create_hypertable('wheel_samples', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS wheel_samples_session_time ON wheel_samples (session, time DESC);

ALTER TABLE wheel_samples SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'session, source',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('wheel_samples', INTERVAL '1 day', if_not_exists => TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS stability_1m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT session,
       source,
       time_bucket(INTERVAL '1 minute', time)                        AS bucket,
       count(*)                                                      AS samples,
       avg(abs(steer_deg))                                           AS mean_abs_steer_deg,
       stddev_samp(steer_deg)                                        AS steer_jitter_deg,
       avg(abs(target_deg - steer_deg))                              AS tracking_err_deg,
       count(*) FILTER (WHERE mode = 'takeover')                     AS takeover_samples,
       count(*) FILTER (WHERE mode = 'assist')                       AS assist_samples,
       avg(speed_mps) * 3.6                                          AS avg_kmh,
       max(speed_mps) * 3.6                                          AS max_kmh
FROM wheel_samples
GROUP BY session, source, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('stability_1m',
    start_offset => NULL,
    end_offset => INTERVAL '10 seconds',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE);
