"""Astronomy provider — sun/moon ephemeris computed locally with `astral`.

A provider does not have to be a remote API: this one computes everything
on the Pi (no network), and the registry's TTL cache simply spares the
recomputation. The astronomical-darkness window is the observing window
for an all-sky camera, which is why it gets first-class fields.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

from astral import Observer, moon, sun

from .base import Provider

_PHASES = [
    ("New Moon", "\U0001f311"),
    ("Waxing Crescent", "\U0001f312"),
    ("First Quarter", "\U0001f313"),
    ("Waxing Gibbous", "\U0001f314"),
    ("Full Moon", "\U0001f315"),
    ("Waning Gibbous", "\U0001f316"),
    ("Last Quarter", "\U0001f317"),
    ("Waning Crescent", "\U0001f318"),
]


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _try(fn, *args, **kwargs):
    """astral raises ValueError when an event doesn't happen (e.g. no
    astronomical darkness at high latitudes in summer) — map that to None."""
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return None


class AstronomyProvider(Provider):
    name = "astronomy"
    resources = ("ephemeris",)

    def fetch(self, resource: str) -> dict:
        observer = Observer(
            latitude=self.settings["latitude"],
            longitude=self.settings["longitude"],
            elevation=self.settings.get("elevation_m", 0.0),
        )
        now = datetime.now(timezone.utc)
        today: date = now.date()
        tomorrow = today + timedelta(days=1)

        sunrise = _try(sun.sunrise, observer, today, tzinfo=timezone.utc)
        sunset = _try(sun.sunset, observer, today, tzinfo=timezone.utc)
        noon = _try(sun.noon, observer, today, tzinfo=timezone.utc)
        elevation_now = sun.elevation(observer, now)
        azimuth_now = sun.azimuth(observer, now)
        # Haurwitz clear-sky model: theoretical GHI under a cloudless sky.
        # The gap between this and a measured/derived irradiance is cloud.
        if elevation_now > 0:
            sin_el = math.sin(math.radians(elevation_now))
            clear_sky_wm2 = round(1098 * sin_el * math.exp(-0.057 / sin_el))
        else:
            clear_sky_wm2 = 0

        # Tonight's observing window: astronomical dusk today -> dawn tomorrow.
        astro_dusk = _try(sun.dusk, observer, today, tzinfo=timezone.utc, depression=18.0)
        astro_dawn = _try(sun.dawn, observer, tomorrow, tzinfo=timezone.utc, depression=18.0)
        darkness_hours = (
            round((astro_dawn - astro_dusk).total_seconds() / 3600, 1)
            if astro_dusk and astro_dawn
            else None
        )

        phase_days = moon.phase(today)  # 0 = new, 14 = full, 0..27.99
        # Illuminated fraction from the phase angle (cosine approximation).
        illumination = round((1 - math.cos(math.pi * phase_days / 14)) / 2, 2)
        phase_name, phase_icon = _PHASES[int((phase_days + 1.75) / 3.5) % 8]

        return {
            "date": today.isoformat(),
            "latitude": self.settings["latitude"],
            "longitude": self.settings["longitude"],
            "sun": {
                "sunrise": _iso(sunrise),
                "sunset": _iso(sunset),
                "solar_noon": _iso(noon),
                "elevation_now": round(elevation_now, 1),
                "azimuth_now": round(azimuth_now, 1),
                "clear_sky_irradiance_wm2": clear_sky_wm2,
                "is_up": elevation_now > 0,
            },
            "darkness": {
                "astronomical_dusk": _iso(astro_dusk),
                "astronomical_dawn": _iso(astro_dawn),
                "hours": darkness_hours,
            },
            "moon": {
                # astral's domain is [0, 28); rounding 27.96 to 28.0 would
                # leak out of it, so wrap the rounded value back (28.0 ≡ 0.0,
                # both "new moon").
                "phase_days": round(phase_days, 1) % 28.0,
                "illumination": illumination,
                "phase_name": phase_name,
                "icon": phase_icon,
                "moonrise": _iso(_try(moon.moonrise, observer, today, tzinfo=timezone.utc)),
                "moonset": _iso(_try(moon.moonset, observer, today, tzinfo=timezone.utc)),
            },
        }
