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
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS rpm DOUBLE PRECISION;
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS game_ms BIGINT;

-- Control hand-offs from stream-v1 events.csv: the authoritative takeover record.
CREATE TABLE IF NOT EXISTS control_events (
    time          TIMESTAMPTZ NOT NULL,
    session       TEXT        NOT NULL,
    control_mode  TEXT        NOT NULL,       -- true runtime mode: manual | assist | takeover | fault
    training_mode TEXT,                       -- what the recorder labelled for training (assist unless expert)
    expert        BOOLEAN,
    reason        TEXT
);

SELECT create_hypertable('control_events', 'time', if_not_exists => TRUE);

-- One row per runtime run from `--run-report` JSON; the full document is kept for anything not flattened.
CREATE TABLE IF NOT EXISTS run_reports (
    session                         TEXT PRIMARY KEY,
    loaded_at                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    actuation                       TEXT,     -- motor | direct_vjoy; never compare across these
    duration_s                      DOUBLE PRECISION,
    ticks                           BIGINT,
    manual_s                        DOUBLE PRECISION,
    assist_s                        DOUBLE PRECISION,
    takeover_s                      DOUBLE PRECISION,
    fault_s                         DOUBLE PRECISION,
    human_interventions             INTEGER,
    interventions_per_assist_minute DOUBLE PRECISION,
    tracking_rmse_deg               DOUBLE PRECISION,
    max_abs_torque                  DOUBLE PRECISION,
    fault_entries                   INTEGER,
    routes_attempted                INTEGER,
    routes_completed                INTEGER,
    routes_aborted                  INTEGER,
    error                           TEXT,
    report                          JSONB NOT NULL
);

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
