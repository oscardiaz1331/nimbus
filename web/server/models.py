"""Pydantic request/response schemas for the /api/v1 endpoints.

``ObservationIn`` is the v1 ingest contract observatory POSTs against.
The required core is deliberately minimal (station, timestamp, cloud
cover); everything observatory has not finalized yet (per-class
breakdown, image refs, model info) is optional so the contract can grow
without a v2. JSON timestamps are ISO 8601 UTC; cloud metrics are [0, 1]
fractions everywhere — the UI formats percentages.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ImageRef(BaseModel):
    """Filenames relative to ``storage.images_dir`` (observatory and web
    share the Pi's filesystem; no bytes travel through the API)."""

    filename: str
    mask_filename: str | None = None


class ModelInfo(BaseModel):
    name: str | None = None
    variant: str | None = None
    version: str | None = None


class ObservationIn(BaseModel):
    schema_version: Literal[1]
    station_id: str
    timestamp: datetime
    cloud_cover: float = Field(ge=0.0, le=1.0)
    sky_fraction: float | None = Field(None, ge=0.0, le=1.0)
    classes: dict[str, float] | None = None
    image: ImageRef | None = None
    model: ModelInfo | None = None
    inference_ms: float | None = Field(None, ge=0.0)

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware ISO 8601 (e.g. ...Z)")
        return v.astimezone(timezone.utc)

    @property
    def ts_utc(self) -> int:
        return int(self.timestamp.timestamp())


def _iso(ts_utc: int) -> str:
    return datetime.fromtimestamp(ts_utc, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class ObservationOut(BaseModel):
    id: int
    station_id: str
    timestamp: str
    cloud_cover: float
    sky_fraction: float | None = None
    classes: dict[str, float] | None = None
    image_url: str | None = None
    mask_url: str | None = None
    model: ModelInfo | None = None
    inference_ms: float | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ObservationOut":
        return cls(
            id=row["id"],
            station_id=row["station_id"],
            timestamp=_iso(row["ts_utc"]),
            cloud_cover=row["cloud_cover"],
            sky_fraction=row["sky_fraction"],
            classes=json.loads(row["classes_json"]) if row["classes_json"] else None,
            image_url=f"/images/{row['image_path']}" if row["image_path"] else None,
            mask_url=f"/images/{row['mask_path']}" if row["mask_path"] else None,
            model=ModelInfo(**json.loads(row["model_json"])) if row["model_json"] else None,
            inference_ms=row["inference_ms"],
        )


class SeriesOut(BaseModel):
    """Columnar (aligned arrays): smaller on the wire and uPlot-native."""

    station_id: str
    bucket_seconds: int
    ts: list[int]
    avg: list[float]
    min: list[float]
    max: list[float]
    count: list[int]


class NodeTelemetryIn(BaseModel):
    """v1 telemetry contract for LoRa/IoT nodes (contract proposal for
    observatory's communication/ module, like ObservationIn is for its
    observations). Nodes auto-register on first POST — sensors come and go,
    unlike config-declared stations."""

    schema_version: Literal[1]
    node_id: str = Field(min_length=1)
    timestamp: datetime
    rssi_dbm: float | None = None
    snr_db: float | None = None
    battery_pct: float | None = Field(None, ge=0.0, le=100.0)
    battery_v: float | None = Field(None, ge=0.0)
    extra: dict | None = None  # free-form sensor payload (temperature, uptime, ...)

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware ISO 8601 (e.g. ...Z)")
        return v.astimezone(timezone.utc)

    @property
    def ts_utc(self) -> int:
        return int(self.timestamp.timestamp())


class NodeOut(BaseModel):
    node_id: str
    last_seen: str
    rssi_dbm: float | None = None
    snr_db: float | None = None
    battery_pct: float | None = None
    battery_v: float | None = None
    extra: dict | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "NodeOut":
        return cls(
            node_id=row["node_id"],
            last_seen=_iso(row["ts_utc"]),
            rssi_dbm=row["rssi_dbm"],
            snr_db=row["snr_db"],
            battery_pct=row["battery_pct"],
            battery_v=row["battery_v"],
            extra=json.loads(row["extra_json"]) if row["extra_json"] else None,
        )


class StationOut(BaseModel):
    id: str
    name: str
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None
    last_seen: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StationOut":
        return cls(
            id=row["id"],
            name=row["name"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            elevation_m=row["elevation_m"],
            last_seen=_iso(row["last_seen_utc"]) if row["last_seen_utc"] is not None else None,
        )
