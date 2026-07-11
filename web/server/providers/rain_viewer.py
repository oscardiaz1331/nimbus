"""RainViewer provider (https://www.rainviewer.com — free, no API key).

Serves the frame *index* only; the map widget builds tile URLs from it and
the browser loads tiles straight from RainViewer's CDN (tiles are not
cacheable through provider_cache and never touch the Pi). Tile URL shape:
``{host}{path}/256/{z}/{x}/{y}/2/1_1.png``.
"""

from __future__ import annotations

import httpx

from .base import Provider

API_URL = "https://api.rainviewer.com/public/weather-maps.json"


class RainViewerProvider(Provider):
    name = "rain-viewer"
    resources = ("frames",)

    def fetch(self, resource: str) -> dict:
        response = httpx.get(API_URL, timeout=15)
        response.raise_for_status()
        return self.parse(response.json())

    @staticmethod
    def parse(raw: dict) -> dict:
        radar = raw.get("radar") or {}
        frame = lambda f: {"time": f["time"], "path": f["path"]}  # noqa: E731
        return {
            "host": raw.get("host", "https://tilecache.rainviewer.com"),
            "past": [frame(f) for f in radar.get("past") or []],
            "nowcast": [frame(f) for f in radar.get("nowcast") or []],
        }
