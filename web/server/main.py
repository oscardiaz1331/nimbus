"""App factory + uvicorn entry point.

Run from web/:  uvicorn server.main:app --reload --port 8080
Config path comes from $NIMBUS_WEB_CONFIG, defaulting to web/config.yaml.
Tests build their own app via ``create_app(Config...)``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__, db
from .api import router as api_router
from .config import Config
from .providers import registry


def create_app(config: Config) -> FastAPI:
    conn = db.connect(config.storage.db_path)
    db.init_db(conn)
    db.seed_stations(conn, config.stations)
    conn.close()
    config.storage.images_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Nimbus Web", version=__version__)
    app.state.config = config
    app.state.providers = registry.build_providers(config.providers)

    app.include_router(api_router)
    app.mount("/images", StaticFiles(directory=config.storage.images_dir), name="images")
    # Built SPA, when present; without it the server runs API-only (dev mode).
    # Single page, no client router, so html=True is all the fallback needed.
    if config.server.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=config.server.static_dir, html=True), name="spa")
    return app


def _default_config_path() -> Path:
    return Path(
        os.environ.get(
            "NIMBUS_WEB_CONFIG", Path(__file__).resolve().parent.parent / "config.yaml"
        )
    )


def __getattr__(name: str) -> FastAPI:
    # PEP 562 lazy attribute: building `app` only when uvicorn asks for it
    # keeps `from server.main import create_app` (tests) side-effect free.
    if name == "app":
        return create_app(Config.from_yaml(_default_config_path()))
    raise AttributeError(name)
