"""Station metadata (declared in config.yaml, seeded into the DB) + last_seen."""

from __future__ import annotations

from sqlite3 import Connection

from fastapi import APIRouter, Depends, HTTPException

from .. import db
from ..models import StationOut
from .deps import get_db

router = APIRouter(prefix="/stations", tags=["stations"])


@router.get("")
def list_stations(conn: Connection = Depends(get_db)) -> list[StationOut]:
    return [StationOut.from_row(r) for r in db.stations_with_last_seen(conn)]


@router.get("/{station_id}")
def get_station(station_id: str, conn: Connection = Depends(get_db)) -> StationOut:
    row = db.station_by_id(conn, station_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown station '{station_id}'")
    return StationOut.from_row(row)
