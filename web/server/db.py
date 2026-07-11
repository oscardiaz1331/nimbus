"""SQLite storage: schema, pragmas, and every query the API runs.

Timestamps are stored as integer epoch seconds UTC (``ts_utc``) so bucket
arithmetic for the series endpoint is plain integer math; the API layer
converts to/from ISO 8601. All cloud metrics are [0, 1] fractions.

One uvicorn worker + WAL means concurrent readers and the single writer
(observatory's ingest POSTs) never block each other. Connections are
cheap; the API opens one per request.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from .config import StationConfig

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    latitude    REAL,
    longitude   REAL,
    elevation_m REAL
);

CREATE TABLE IF NOT EXISTS observations (
    id           INTEGER PRIMARY KEY,
    station_id   TEXT NOT NULL REFERENCES stations(id),
    ts_utc       INTEGER NOT NULL,
    cloud_cover  REAL NOT NULL,
    sky_fraction REAL,
    classes_json TEXT,
    image_path   TEXT,
    mask_path    TEXT,
    model_json   TEXT,
    inference_ms REAL,
    UNIQUE (station_id, ts_utc)
);

CREATE TABLE IF NOT EXISTS provider_cache (
    provider   TEXT NOT NULL,
    resource   TEXT NOT NULL,
    fetched_at INTEGER NOT NULL,
    payload    TEXT NOT NULL,
    PRIMARY KEY (provider, resource)
);

CREATE TABLE IF NOT EXISTS node_telemetry (
    id          INTEGER PRIMARY KEY,
    node_id     TEXT NOT NULL,
    ts_utc      INTEGER NOT NULL,
    rssi_dbm    REAL,
    snr_db      REAL,
    battery_pct REAL,
    battery_v   REAL,
    extra_json  TEXT,
    UNIQUE (node_id, ts_utc)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI may run the yielding dependency and
    # the endpoint on different threadpool threads; each request still uses
    # its own connection, never shared across requests.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()


def seed_stations(conn: sqlite3.Connection, stations: Iterable[StationConfig]) -> None:
    """Upsert the config-declared stations; config.yaml wins over the DB."""
    conn.executemany(
        """
        INSERT INTO stations (id, name, latitude, longitude, elevation_m)
        VALUES (:id, :name, :latitude, :longitude, :elevation_m)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name, latitude=excluded.latitude,
            longitude=excluded.longitude, elevation_m=excluded.elevation_m
        """,
        [
            {
                "id": s.id,
                "name": s.name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "elevation_m": s.elevation_m,
            }
            for s in stations
        ],
    )
    conn.commit()


# --- observations ---------------------------------------------------------


def upsert_observation(
    conn: sqlite3.Connection,
    *,
    station_id: str,
    ts_utc: int,
    cloud_cover: float,
    sky_fraction: float | None = None,
    classes_json: str | None = None,
    image_path: str | None = None,
    mask_path: str | None = None,
    model_json: str | None = None,
    inference_ms: float | None = None,
) -> int:
    """Insert one observation; retries with the same (station, timestamp)
    overwrite instead of duplicating, so flaky-network re-sends are safe."""
    row = conn.execute(
        """
        INSERT INTO observations (station_id, ts_utc, cloud_cover, sky_fraction,
                                  classes_json, image_path, mask_path, model_json,
                                  inference_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_id, ts_utc) DO UPDATE SET
            cloud_cover=excluded.cloud_cover, sky_fraction=excluded.sky_fraction,
            classes_json=excluded.classes_json, image_path=excluded.image_path,
            mask_path=excluded.mask_path, model_json=excluded.model_json,
            inference_ms=excluded.inference_ms
        RETURNING id
        """,
        (station_id, ts_utc, cloud_cover, sky_fraction, classes_json,
         image_path, mask_path, model_json, inference_ms),
    ).fetchone()
    conn.commit()
    return row["id"]


def latest_observation(conn: sqlite3.Connection, station_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM observations WHERE station_id=? ORDER BY ts_utc DESC LIMIT 1",
        (station_id,),
    ).fetchone()


def list_observations(
    conn: sqlite3.Connection, station_id: str, start: int, end: int, limit: int
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM observations
        WHERE station_id=? AND ts_utc BETWEEN ? AND ?
        ORDER BY ts_utc DESC LIMIT ?
        """,
        (station_id, start, end, limit),
    ).fetchall()


def iter_observations(
    conn: sqlite3.Connection, station_id: str, start: int, end: int
):
    """Unbounded chronological cursor for CSV export — rows stream out
    without materializing the whole range in memory."""
    return conn.execute(
        """
        SELECT * FROM observations
        WHERE station_id=? AND ts_utc BETWEEN ? AND ?
        ORDER BY ts_utc ASC
        """,
        (station_id, start, end),
    )


def observation_series(
    conn: sqlite3.Connection, station_id: str, start: int, end: int, bucket_seconds: int
) -> dict[str, list]:
    """Columnar bucketed series (aligned arrays — uPlot's native format).

    On-the-fly GROUP BY over the (station_id, ts_utc) unique index; at
    ~1 obs/min this stays in the tens of milliseconds on a Pi 5 even for
    a year-long range, so there is no pre-aggregation table.
    """
    rows = conn.execute(
        """
        SELECT (ts_utc / :bucket) * :bucket AS bucket_ts,
               AVG(cloud_cover) AS avg, MIN(cloud_cover) AS min,
               MAX(cloud_cover) AS max, COUNT(*) AS count
        FROM observations
        WHERE station_id = :station_id AND ts_utc BETWEEN :start AND :end
        GROUP BY bucket_ts ORDER BY bucket_ts
        """,
        {"bucket": bucket_seconds, "station_id": station_id, "start": start, "end": end},
    ).fetchall()
    return {
        "ts": [r["bucket_ts"] for r in rows],
        "avg": [r["avg"] for r in rows],
        "min": [r["min"] for r in rows],
        "max": [r["max"] for r in rows],
        "count": [r["count"] for r in rows],
    }


# --- stations --------------------------------------------------------------


def stations_with_last_seen(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT s.*, MAX(o.ts_utc) AS last_seen_utc
        FROM stations s LEFT JOIN observations o ON o.station_id = s.id
        GROUP BY s.id ORDER BY s.id
        """
    ).fetchall()


def station_by_id(conn: sqlite3.Connection, station_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT s.*, MAX(o.ts_utc) AS last_seen_utc
        FROM stations s LEFT JOIN observations o ON o.station_id = s.id
        WHERE s.id = ? GROUP BY s.id
        """,
        (station_id,),
    ).fetchone()


# --- LoRa node telemetry ----------------------------------------------------
# Nodes are NOT config-declared (unlike stations): IoT sensors come and go,
# so they auto-register on their first telemetry POST.


def upsert_node_telemetry(
    conn: sqlite3.Connection,
    *,
    node_id: str,
    ts_utc: int,
    rssi_dbm: float | None = None,
    snr_db: float | None = None,
    battery_pct: float | None = None,
    battery_v: float | None = None,
    extra_json: str | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO node_telemetry (node_id, ts_utc, rssi_dbm, snr_db,
                                    battery_pct, battery_v, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id, ts_utc) DO UPDATE SET
            rssi_dbm=excluded.rssi_dbm, snr_db=excluded.snr_db,
            battery_pct=excluded.battery_pct, battery_v=excluded.battery_v,
            extra_json=excluded.extra_json
        RETURNING id
        """,
        (node_id, ts_utc, rssi_dbm, snr_db, battery_pct, battery_v, extra_json),
    ).fetchone()
    conn.commit()
    return row["id"]


def nodes_latest(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Most recent telemetry row per node."""
    return conn.execute(
        """
        SELECT t.* FROM node_telemetry t
        JOIN (SELECT node_id, MAX(ts_utc) AS m FROM node_telemetry GROUP BY node_id) x
          ON x.node_id = t.node_id AND x.m = t.ts_utc
        ORDER BY t.node_id
        """
    ).fetchall()


def node_telemetry_history(
    conn: sqlite3.Connection, node_id: str, limit: int
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM node_telemetry WHERE node_id=? ORDER BY ts_utc DESC LIMIT ?",
        (node_id, limit),
    ).fetchall()


# --- provider cache ---------------------------------------------------------


def cache_get(conn: sqlite3.Connection, provider: str, resource: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT fetched_at, payload FROM provider_cache WHERE provider=? AND resource=?",
        (provider, resource),
    ).fetchone()


def cache_put(
    conn: sqlite3.Connection,
    provider: str,
    resource: str,
    payload: str,
    fetched_at: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO provider_cache (provider, resource, fetched_at, payload)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(provider, resource) DO UPDATE SET
            fetched_at=excluded.fetched_at, payload=excluded.payload
        """,
        (provider, resource, fetched_at if fetched_at is not None else int(time.time()), payload),
    )
    conn.commit()


def cache_ages(conn: sqlite3.Connection, now: int | None = None) -> list[dict]:
    now = now if now is not None else int(time.time())
    rows = conn.execute(
        "SELECT provider, resource, fetched_at FROM provider_cache ORDER BY provider, resource"
    ).fetchall()
    return [
        {
            "provider": r["provider"],
            "resource": r["resource"],
            "age_seconds": now - r["fetched_at"],
        }
        for r in rows
    ]
