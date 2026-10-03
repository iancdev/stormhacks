"""Connection and schema management for the Tiger Data analytics sink."""
import os
from pathlib import Path

import psycopg

ENV_VAR = "TIGER_DATA_URL"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def database_url(explicit=None, dotenv=Path(".env")):
    """Resolve the connection string: argument, environment, then a local .env file."""
    if explicit:
        return explicit
    if os.environ.get(ENV_VAR):
        return os.environ[ENV_VAR]
    if dotenv is not None and Path(dotenv).is_file():
        for line in Path(dotenv).read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key.strip() == ENV_VAR and value.strip():
                return value.strip().strip("'\"")
    raise SystemExit(f"{ENV_VAR} is not set. Put the Tiger Cloud service URL in the environment or in .env")


def connect(url=None):
    """Autocommit connection; TimescaleDB DDL for continuous aggregates must not run in a transaction."""
    return psycopg.connect(database_url(url), autocommit=True)


def statements(text=None):
    text = SCHEMA_PATH.read_text() if text is None else text
    body = "\n".join(line.split("--", 1)[0].rstrip() for line in text.splitlines())
    return [part.strip() for part in body.split(";") if part.strip()]


def apply_schema(conn):
    """Idempotent: safe to rerun against an existing service."""
    with conn.cursor() as cursor:
        for statement in statements():
            cursor.execute(statement)


def refresh_aggregates(conn):
    with conn.cursor() as cursor:
        cursor.execute("CALL refresh_continuous_aggregate('stability_1m', NULL, NULL)")
