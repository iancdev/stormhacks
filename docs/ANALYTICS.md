# Driving analytics on Tiger Data

Tiger Cloud is hosted PostgreSQL with the TimescaleDB extension. The analytics
package loads the CSV files the recorder and runtime already write into one
hypertable and rolls them into a continuous aggregate, so the evaluation metrics
from the README (interventions, steering stability, tracking error) are plain
SQL instead of ad hoc scripts. It is analytics only: the control loop, training,
and inference never read from or write to the database, and no image pixels
are stored (rows keep an `image_path` pointer to the JPEG on disk).

## Setup

1. Create a Free Plan service in [Tiger Console](https://console.cloud.tigerdata.com/)
   (2 services per account, 750 MB each, read-only when full; an hour of 60 Hz
   rows is roughly 20 MB). Download the config when prompted; the password is
   shown once.
2. Copy `.env.example` to `.env` and set `TIGER_DATA_URL` to the service URL.
   `.env` is gitignored. `--url` or the environment variable also work.
3. Install and create the schema:

```sh
python -m pip install -e '.[analytics]'
forza-analytics init
```

`init` is idempotent. It creates:

- `wheel_samples`: hypertable keyed by `time`, with `session`, `source`
  (`recorder` or `runtime`), `mode`, measured `steer_deg`, controller
  `target_deg`, policy `predicted_deg`, `torque`, `speed_mps`, pedals,
  `race_on`, and `obs_age_ms`. Compression policy after one day.
- `stability_1m`: continuous aggregate per session, source, and one-minute
  bucket with mean absolute steering, steering jitter, tracking error,
  takeover/assist sample counts, and speed.

## Load data

```sh
# Human lap from record.py (meta.json, labels.csv, frames/)
forza-analytics load-recording data/recordings/20261003_161200

# Controller or AI run from `python -m forza_ai.runtime ... --status-csv runs/shadow.csv`
forza-analytics load-run runs/shadow.csv --session shadow-v1
```

Reloading a session replaces its rows, so repeated loads are safe; `--append`
keeps existing rows. Recorder rows are anchored to the session's
`YYYYMMDD_HHMMSS` directory name in local time; runtime rows are anchored so the
last tick lands on the CSV modification time (or `--end-time`). Both sources use
relative or monotonic clocks, so these wall-clock times are for charting and
grouping, not cross-machine comparison.

## Report

```sh
forza-analytics report            # per-session summary and per-minute rollup
forza-analytics report --events   # also list each takeover onset in AI runs
```

The SQL behind the report lives in `src/forza_ai/analytics/report.py` and can be
pasted into the Tiger Console SQL editor or any Postgres client. Takeover
percentage and `takeover_samples` per minute are the intervention metric; a
lower `tracking_err_deg` means the PD controller follows the model's target
more closely; comparing `steer_jitter_deg` between a `recorder` human lap and a
`runtime` AI run on the same route shows steering stability.

Tests in `tests/analytics` cover CSV conversion and schema parsing with fakes
and need no database.
