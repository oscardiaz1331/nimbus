"""Explicit provider registry + the cache-first read path.

Read-through semantics (WeatherNode's cache-first idea, one code path):
fresh cache -> serve it; expired -> fetch, store, serve; fetch failed but
a stale payload exists -> serve stale (flagged) rather than a broken
widget; nothing at all -> let the error propagate (routes turn it into
502). The cache lives in SQLite so it survives Pi reboots.
"""

from __future__ import annotations

import json
import logging
import time
from sqlite3 import Connection

from .. import db
from ..config import ProviderConfig
from .astronomy import AstronomyProvider
from .base import Provider
from .open_meteo import OpenMeteoProvider
from .rain_viewer import RainViewerProvider

logger = logging.getLogger(__name__)

# Explicit map, no decorator magic (academy convention). Adding a provider
# is one import + one line here + a config.yaml block.
PROVIDERS: dict[str, type[Provider]] = {
    OpenMeteoProvider.name: OpenMeteoProvider,
    AstronomyProvider.name: AstronomyProvider,
    RainViewerProvider.name: RainViewerProvider,
}


def build_providers(configs: list[ProviderConfig]) -> dict[str, Provider]:
    """Instantiate the enabled providers; unknown names fail at startup."""
    built: dict[str, Provider] = {}
    for cfg in configs:
        if not cfg.enabled:
            continue
        if cfg.name not in PROVIDERS:
            raise ValueError(
                f"unknown provider '{cfg.name}' in config.yaml — "
                f"known providers: {sorted(PROVIDERS)}"
            )
        built[cfg.name] = PROVIDERS[cfg.name](cfg.settings, cfg.ttl_seconds)
    return built


def get_data(
    conn: Connection, provider: Provider, resource: str, now: int | None = None
) -> dict:
    """Cache-first read. Returns ``{"data", "fetched_at", "stale"}``."""
    now = now if now is not None else int(time.time())
    cached = db.cache_get(conn, provider.name, resource)

    if cached and now - cached["fetched_at"] < provider.ttl_seconds:
        return {
            "data": json.loads(cached["payload"]),
            "fetched_at": cached["fetched_at"],
            "stale": False,
        }

    try:
        payload = provider.fetch(resource)
    except Exception:
        if cached:
            logger.warning(
                "provider '%s/%s' fetch failed; serving stale cache",
                provider.name, resource, exc_info=True,
            )
            return {
                "data": json.loads(cached["payload"]),
                "fetched_at": cached["fetched_at"],
                "stale": True,
            }
        raise

    db.cache_put(conn, provider.name, resource, json.dumps(payload), fetched_at=now)
    return {"data": payload, "fetched_at": now, "stale": False}
