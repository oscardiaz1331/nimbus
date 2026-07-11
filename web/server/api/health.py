"""Liveness/diagnostics endpoint (also handy for e-paper status screens).

The ``system`` block reads /proc and /sys directly (no psutil dependency);
any probe that fails on the host — e.g. no thermal zone under WSL — just
reports null.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from sqlite3 import Connection

from fastapi import APIRouter, Depends

from .. import __version__, db
from ..config import Config
from .deps import get_config, get_db

router = APIRouter(tags=["health"])


def _read_first_line(path: str) -> str:
    return Path(path).read_text(encoding="ascii").splitlines()[0]


def _system_health() -> dict:
    out = {
        "cpu_count": os.cpu_count(),
        "load_1m": None,
        "mem_total_mb": None,
        "mem_available_mb": None,
        "cpu_temp_c": None,
        "uptime_hours": None,
    }
    try:
        out["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            meminfo[key] = int(rest.split()[0])  # kB
        out["mem_total_mb"] = meminfo["MemTotal"] // 1024
        out["mem_available_mb"] = meminfo["MemAvailable"] // 1024
    except (OSError, KeyError, ValueError, IndexError):
        pass
    try:
        out["cpu_temp_c"] = round(
            int(_read_first_line("/sys/class/thermal/thermal_zone0/temp")) / 1000, 1
        )
    except (OSError, ValueError, IndexError):
        pass
    try:
        out["uptime_hours"] = round(float(_read_first_line("/proc/uptime").split()[0]) / 3600, 1)
    except (OSError, ValueError, IndexError):
        pass
    return out


@router.get("/health")
def health(
    conn: Connection = Depends(get_db),
    config: Config = Depends(get_config),
) -> dict:
    db_ok = conn.execute("SELECT 1").fetchone() is not None
    disk = shutil.disk_usage(config.storage.db_path.parent)
    return {
        "status": "ok" if db_ok else "degraded",
        "version": __version__,
        "db_ok": db_ok,
        "disk_free_mb": disk.free // (1024 * 1024),
        "system": _system_health(),
        "provider_cache": db.cache_ages(conn),
    }
