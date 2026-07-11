"""Shared FastAPI dependencies: config access and a per-request DB connection."""

from __future__ import annotations

from collections.abc import Iterator
from sqlite3 import Connection

from fastapi import Request

from .. import db
from ..config import Config


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_db(request: Request) -> Iterator[Connection]:
    conn = db.connect(request.app.state.config.storage.db_path)
    try:
        yield conn
    finally:
        conn.close()
