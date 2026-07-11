"""Ingest (POST) and query (latest / series / raw / CSV export) endpoints.

POST is the single write path of the whole system — a future SSE stream
would hook into it without touching anything else.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timedelta, timezone
from sqlite3 import Connection

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from .. import db
from ..config import Config
from ..models import ObservationIn, ObservationOut, SeriesOut
from .deps import get_config, get_db

router = APIRouter(prefix="/observations", tags=["observations"])

_BUCKET_RE = re.compile(r"^(\d+)([smhd]?)$")
_BUCKET_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_bucket(bucket: str) -> int:
    m = _BUCKET_RE.match(bucket)
    if not m or int(m.group(1)) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"invalid bucket '{bucket}' — use e.g. '30s', '10m', '1h', '1d'",
        )
    return int(m.group(1)) * _BUCKET_UNITS[m.group(2)]


def _resolve_station(config: Config, station_id: str | None) -> str:
    if station_id is None:
        return config.stations[0].id
    if config.station(station_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown station '{station_id}' — stations are declared in config.yaml",
        )
    return station_id


def _epoch(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # naive query params are taken as UTC
    return int(dt.timestamp())


@router.post("", status_code=201)
def ingest(
    obs: ObservationIn,
    conn: Connection = Depends(get_db),
    config: Config = Depends(get_config),
) -> dict:
    _resolve_station(config, obs.station_id)
    obs_id = db.upsert_observation(
        conn,
        station_id=obs.station_id,
        ts_utc=obs.ts_utc,
        cloud_cover=obs.cloud_cover,
        sky_fraction=obs.sky_fraction,
        classes_json=json.dumps(obs.classes) if obs.classes else None,
        image_path=obs.image.filename if obs.image else None,
        mask_path=obs.image.mask_filename if obs.image else None,
        model_json=json.dumps(obs.model.model_dump(exclude_none=True)) if obs.model else None,
        inference_ms=obs.inference_ms,
    )
    return {"id": obs_id}


@router.get("/latest")
def latest(
    station_id: str | None = None,
    conn: Connection = Depends(get_db),
    config: Config = Depends(get_config),
) -> ObservationOut:
    station = _resolve_station(config, station_id)
    row = db.latest_observation(conn, station)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no observations yet for '{station}'")
    return ObservationOut.from_row(row)


@router.get("/series")
def series(
    station_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    bucket: str = "10m",
    conn: Connection = Depends(get_db),
    config: Config = Depends(get_config),
) -> SeriesOut:
    station = _resolve_station(config, station_id)
    bucket_seconds = _parse_bucket(bucket)
    end_dt = end or datetime.now(timezone.utc)
    start_dt = start or end_dt - timedelta(hours=24)
    data = db.observation_series(conn, station, _epoch(start_dt), _epoch(end_dt), bucket_seconds)
    return SeriesOut(station_id=station, bucket_seconds=bucket_seconds, **data)


_CSV_COLUMNS = (
    "timestamp", "station_id", "cloud_cover", "sky_fraction",
    "inference_ms", "image_path", "mask_path", "classes_json", "model_json",
)


@router.get("/export.csv")
def export_csv(
    station_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    conn: Connection = Depends(get_db),
    config: Config = Depends(get_config),
) -> StreamingResponse:
    """Stream the raw rows of a time range as CSV (default: last 24 h)."""
    station = _resolve_station(config, station_id)
    end_dt = end or datetime.now(timezone.utc)
    start_dt = start or end_dt - timedelta(hours=24)

    def rows():
        buf = io.StringIO()
        writer = csv.writer(buf)

        def flush() -> str:
            value = buf.getvalue()
            buf.seek(0)
            buf.truncate(0)
            return value

        writer.writerow(_CSV_COLUMNS)
        yield flush()
        for r in db.iter_observations(conn, station, _epoch(start_dt), _epoch(end_dt)):
            writer.writerow(
                [
                    datetime.fromtimestamp(r["ts_utc"], tz=timezone.utc).isoformat(),
                    r["station_id"], r["cloud_cover"], r["sky_fraction"],
                    r["inference_ms"], r["image_path"], r["mask_path"],
                    r["classes_json"], r["model_json"],
                ]
            )
            yield flush()

    filename = f"{station}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv"
    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("")
def raw(
    station_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(100, ge=1, le=1000),
    conn: Connection = Depends(get_db),
    config: Config = Depends(get_config),
) -> list[ObservationOut]:
    station = _resolve_station(config, station_id)
    end_dt = end or datetime.now(timezone.utc)
    start_dt = start or end_dt - timedelta(hours=24)
    rows = db.list_observations(conn, station, _epoch(start_dt), _epoch(end_dt), limit)
    return [ObservationOut.from_row(r) for r in rows]
