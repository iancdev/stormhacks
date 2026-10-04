-- Tiger Data (TimescaleDB) analytics schema. Analytics only: never read by the
-- control loop, training, or inference. Each statement is executed separately
-- in autocommit mode; continuous aggregates cannot be created in a transaction.
-- Every statement is idempotent so `forza-analytics init` can be rerun after upgrades.

-- Sample stream from every source: 'recorder' (record.py human laps), 'session'
-- (runtime --record-session, stream-v1) and 'runtime' (--status-csv finite test runs).
CREATE TABLE IF NOT EXISTS wheel_samples (
    time          TIMESTAMPTZ      NOT NULL,
    session       TEXT             NOT NULL,
    source        TEXT             NOT NULL,  -- recorder | session | runtime
    segment       INTEGER,                    -- recorder pause segment
    frame_id      BIGINT,
    image_path    TEXT,                       -- pointer to the JPEG on disk; pixels are never stored here
    mode          TEXT,                       -- manual | assist | takeover | fault
    steer_deg     DOUBLE PRECISION,           -- measured physical wheel angle, right positive
    target_deg    DOUBLE PRECISION,           -- limited controller target (status CSV or predictions.csv)
    predicted_deg DOUBLE PRECISION,           -- raw policy output (status CSV or predictions.csv)
    torque        DOUBLE PRECISION,           -- normalized [-1, 1] (status CSV only)
    speed_mps     DOUBLE PRECISION,
    gas           DOUBLE PRECISION,
    brake         DOUBLE PRECISION,
    race_on       BOOLEAN,
    obs_age_ms    DOUBLE PRECISION,           -- wheel sample age (recorder) or observation age (status CSV)
    yaw_rate      DOUBLE PRECISION,           -- newer record.py versions only
    gear          INTEGER,                    -- newer record.py versions only
    rpm           DOUBLE PRECISION,           -- stream-v1 telemetry
    game_ms       BIGINT,                     -- stream-v1 game clock
    predicted_gas DOUBLE PRECISION,           -- AI pedal output (stream-v1 predictions.csv)
    predicted_brake DOUBLE PRECISION
);

-- Columns added after the first deployment; no-ops on a fresh install.
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS yaw_rate DOUBLE PRECISION;
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS gear INTEGER;
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS rpm DOUBLE PRECISION;
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS game_ms BIGINT;
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS predicted_gas DOUBLE PRECISION;
ALTER TABLE wheel_samples ADD COLUMN IF NOT EXISTS predicted_brake DOUBLE PRECISION;

SELECT create_hypertable('wheel_samples', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS wheel_samples_session_time ON wheel_samples (session, time DESC);

ALTER TABLE wheel_samples SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'session, source',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('wheel_samples', INTERVAL '1 day', if_not_exists => TRUE);

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

-- Per-minute rollup read by reports and dashboards.
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
