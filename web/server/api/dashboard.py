"""Dashboard layout endpoint — the frontend renders exactly what
config.yaml declares, so rearranging widgets is a config edit, not code."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends

from ..config import Config
from .deps import get_config

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard")
def dashboard(config: Config = Depends(get_config)) -> dict:
    return {
        "title": config.dashboard.title,
        "refresh_seconds": config.dashboard.refresh_seconds,
        "ambient": config.dashboard.ambient,
        "sections": [
            {
                "id": s.id,
                "title": s.title,
                "icon": s.icon,
                "layout": [dataclasses.asdict(w) for w in s.layout if w.enabled],
            }
            for s in config.dashboard.sections
        ],
    }
