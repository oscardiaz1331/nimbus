"""Generic cache-first routes for external data providers.

This module only knows the Provider ABC and the registry — never a
concrete provider (see server/providers/base.py).
"""

from __future__ import annotations

from sqlite3 import Connection

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import db
from ..providers import registry
from ..providers.base import Provider
from .deps import get_db

router = APIRouter(prefix="/providers", tags=["providers"])


def _get_provider(request: Request, name: str) -> Provider:
    provider = request.app.state.providers.get(name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"unknown or disabled provider '{name}'")
    return provider


@router.get("")
def list_providers(request: Request, conn: Connection = Depends(get_db)) -> list[dict]:
    ages = {(c["provider"], c["resource"]): c["age_seconds"] for c in db.cache_ages(conn)}
    return [
        {
            "name": p.name,
            "resources": list(p.resources),
            "ttl_seconds": p.ttl_seconds,
            "cache_age_seconds": {r: ages.get((p.name, r)) for r in p.resources},
        }
        for p in request.app.state.providers.values()
    ]


@router.get("/{name}/{resource}")
def provider_data(
    name: str,
    resource: str,
    request: Request,
    conn: Connection = Depends(get_db),
) -> dict:
    provider = _get_provider(request, name)
    if resource not in provider.resources:
        raise HTTPException(
            status_code=404,
            detail=f"provider '{name}' has no resource '{resource}' "
            f"(available: {list(provider.resources)})",
        )
    try:
        result = registry.get_data(conn, provider, resource)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"provider '{name}/{resource}' fetch failed and no cached data exists: {exc}",
        ) from exc
    return {"provider": name, "resource": resource, **result}
