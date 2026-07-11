"""LoRa/IoT node telemetry: ingest + health queries.

Nodes auto-register on their first POST (unlike stations, which are
config-declared): field sensors appear and disappear, and requiring a
config edit per node would defeat the point. The widget derives
online/stale/offline from ``last_seen`` age client-side.
"""

from __future__ import annotations

import json
from sqlite3 import Connection

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import db
from ..models import NodeOut, NodeTelemetryIn
from .deps import get_db

router = APIRouter(prefix="/nodes", tags=["nodes"])


@router.post("/telemetry", status_code=201)
def ingest(telemetry: NodeTelemetryIn, conn: Connection = Depends(get_db)) -> dict:
    row_id = db.upsert_node_telemetry(
        conn,
        node_id=telemetry.node_id,
        ts_utc=telemetry.ts_utc,
        rssi_dbm=telemetry.rssi_dbm,
        snr_db=telemetry.snr_db,
        battery_pct=telemetry.battery_pct,
        battery_v=telemetry.battery_v,
        extra_json=json.dumps(telemetry.extra) if telemetry.extra else None,
    )
    return {"id": row_id}


@router.get("")
def list_nodes(conn: Connection = Depends(get_db)) -> list[NodeOut]:
    return [NodeOut.from_row(r) for r in db.nodes_latest(conn)]


@router.get("/{node_id}/telemetry")
def history(
    node_id: str,
    limit: int = Query(100, ge=1, le=1000),
    conn: Connection = Depends(get_db),
) -> list[NodeOut]:
    rows = db.node_telemetry_history(conn, node_id, limit)
    if not rows:
        raise HTTPException(status_code=404, detail=f"no telemetry for node '{node_id}'")
    return [NodeOut.from_row(r) for r in rows]
