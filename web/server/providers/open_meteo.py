"""Open-Meteo forecast provider (https://open-meteo.com — free, no API key).

Chosen as the first provider because its hourly low/mid/high cloud-cover
layers line up with the domain. Cloud-cover values are percent 0-100 as
delivered by the API.
"""

from __future__ import annotations

import httpx

from .base import Provider

API_URL = "https://api.open-meteo.com/v1/forecast"

_HOURLY_FIELDS = (
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
    "precipitation_probability",
)

# `current` feeds the ambient-effects layer (rain/snow/sun/clouds) and any
# "conditions now" fallback when the station itself is offline.
_CURRENT_FIELDS = (
    "temperature_2m",
    "cloud_cover",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "is_day",
)


class OpenMeteoProvider(Provider):
    name = "open-meteo"
    resources = ("forecast",)

    def fetch(self, resource: str) -> dict:
        response = httpx.get(
            API_URL,
            params={
                "latitude": self.settings["latitude"],
                "longitude": self.settings["longitude"],
                "hourly": ",".join(_HOURLY_FIELDS),
                "current": ",".join(_CURRENT_FIELDS),
                "forecast_days": self.settings.get("forecast_days", 2),
                "timezone": "UTC",
            },
            timeout=15,
        )
        response.raise_for_status()
        return self.parse(response.json())

    @staticmethod
    def parse(raw: dict) -> dict:
        """Trim the Open-Meteo response to the fields the dashboard uses."""
        hourly = raw.get("hourly") or {}
        current = raw.get("current") or {}
        return {
            "latitude": raw.get("latitude"),
            "longitude": raw.get("longitude"),
            "hourly_units": raw.get("hourly_units") or {},
            "hourly": {
                "time": hourly.get("time") or [],
                **{f: hourly.get(f) or [] for f in _HOURLY_FIELDS},
            },
            "current": {f: current.get(f) for f in _CURRENT_FIELDS},
        }
