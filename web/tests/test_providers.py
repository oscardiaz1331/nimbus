import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from server import db
from server.config import ProviderConfig
from server.providers import registry
from server.providers.astronomy import AstronomyProvider
from server.providers.base import Provider
from server.providers.open_meteo import OpenMeteoProvider
from server.providers.rain_viewer import RainViewerProvider


class FakeProvider(Provider):
    name = "fake"
    resources = ("data",)

    def __init__(self, settings=None, ttl_seconds=600):
        super().__init__(settings or {}, ttl_seconds)
        self.fetch_count = 0
        self.fail = False

    def fetch(self, resource: str) -> dict:
        self.fetch_count += 1
        if self.fail:
            raise RuntimeError("external API down")
        return {"value": self.fetch_count}


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = db.connect(Path(self.tmp.name) / "test.db")
        self.addCleanup(self.conn.close)
        db.init_db(self.conn)
        self.provider = FakeProvider()

    def test_fresh_cache_is_not_refetched(self):
        first = registry.get_data(self.conn, self.provider, "data", now=1000)
        second = registry.get_data(self.conn, self.provider, "data", now=1000 + 599)
        self.assertEqual(self.provider.fetch_count, 1)
        self.assertEqual(second["data"], first["data"])
        self.assertFalse(second["stale"])

    def test_expired_cache_refetches(self):
        registry.get_data(self.conn, self.provider, "data", now=1000)
        result = registry.get_data(self.conn, self.provider, "data", now=1000 + 600)
        self.assertEqual(self.provider.fetch_count, 2)
        self.assertEqual(result["data"], {"value": 2})
        self.assertEqual(result["fetched_at"], 1600)

    def test_failed_fetch_serves_stale_cache(self):
        registry.get_data(self.conn, self.provider, "data", now=1000)
        self.provider.fail = True
        result = registry.get_data(self.conn, self.provider, "data", now=1000 + 9999)
        self.assertTrue(result["stale"])
        self.assertEqual(result["data"], {"value": 1})
        self.assertEqual(result["fetched_at"], 1000)

    def test_failed_fetch_without_cache_raises(self):
        self.provider.fail = True
        with self.assertRaises(RuntimeError):
            registry.get_data(self.conn, self.provider, "data", now=1000)


class TestBuildProviders(unittest.TestCase):
    def test_unknown_provider_name_fails_at_startup(self):
        with self.assertRaises(ValueError):
            registry.build_providers([ProviderConfig(name="nope")])

    def test_disabled_provider_not_instantiated(self):
        built = registry.build_providers(
            [ProviderConfig(name="open-meteo", enabled=False, settings={})]
        )
        self.assertEqual(built, {})

    def test_enabled_provider_gets_settings_and_ttl(self):
        built = registry.build_providers(
            [
                ProviderConfig(
                    name="open-meteo",
                    ttl_seconds=123,
                    settings={"latitude": 1.0, "longitude": 2.0},
                )
            ]
        )
        provider = built["open-meteo"]
        self.assertIsInstance(provider, OpenMeteoProvider)
        self.assertEqual(provider.ttl_seconds, 123)
        self.assertEqual(provider.settings["latitude"], 1.0)


class TestOpenMeteoParse(unittest.TestCase):
    # Trimmed real response shape from api.open-meteo.com/v1/forecast
    RAW = {
        "latitude": 40.4168,
        "longitude": -3.7038,
        "generationtime_ms": 0.2,
        "hourly_units": {"cloud_cover": "%", "temperature_2m": "°C"},
        "hourly": {
            "time": ["2026-07-10T00:00", "2026-07-10T01:00"],
            "cloud_cover": [25, 80],
            "cloud_cover_low": [10, 60],
            "cloud_cover_mid": [15, 20],
            "cloud_cover_high": [0, 0],
            "temperature_2m": [21.4, 20.9],
            "precipitation_probability": [0, 5],
        },
        "current": {
            "temperature_2m": 22.1,
            "cloud_cover": 40,
            "precipitation": 0.0,
            "rain": 0.0,
            "snowfall": 0.0,
            "weather_code": 2,
            "is_day": 1,
        },
        "unwanted_key": {"huge": "blob"},
    }

    def test_parse_keeps_dashboard_fields_only(self):
        parsed = OpenMeteoProvider.parse(self.RAW)
        self.assertEqual(parsed["latitude"], 40.4168)
        self.assertEqual(parsed["hourly"]["cloud_cover"], [25, 80])
        self.assertEqual(parsed["hourly"]["time"], ["2026-07-10T00:00", "2026-07-10T01:00"])
        self.assertEqual(parsed["hourly_units"]["cloud_cover"], "%")
        self.assertEqual(parsed["current"]["weather_code"], 2)
        self.assertEqual(parsed["current"]["is_day"], 1)
        self.assertNotIn("unwanted_key", parsed)

    def test_parse_tolerates_missing_fields(self):
        parsed = OpenMeteoProvider.parse({"latitude": 1.0})
        self.assertEqual(parsed["hourly"]["time"], [])
        self.assertEqual(parsed["hourly"]["cloud_cover"], [])
        self.assertIsNone(parsed["current"]["weather_code"])


class TestRainViewerParse(unittest.TestCase):
    # Trimmed real response shape from api.rainviewer.com/public/weather-maps.json
    RAW = {
        "version": "2.0",
        "generated": 1783600000,
        "host": "https://tilecache.rainviewer.com",
        "radar": {
            "past": [
                {"time": 1783596400, "path": "/v2/radar/1783596400"},
                {"time": 1783597000, "path": "/v2/radar/1783597000"},
            ],
            "nowcast": [{"time": 1783600600, "path": "/v2/radar/nowcast_abc"}],
        },
        "satellite": {"infrared": [{"time": 1, "path": "/x"}]},
    }

    def test_parse_keeps_radar_frames(self):
        parsed = RainViewerProvider.parse(self.RAW)
        self.assertEqual(parsed["host"], "https://tilecache.rainviewer.com")
        self.assertEqual(len(parsed["past"]), 2)
        self.assertEqual(parsed["past"][1]["path"], "/v2/radar/1783597000")
        self.assertEqual(len(parsed["nowcast"]), 1)
        self.assertNotIn("satellite", parsed)

    def test_parse_tolerates_missing_radar(self):
        parsed = RainViewerProvider.parse({})
        self.assertEqual(parsed["past"], [])
        self.assertEqual(parsed["nowcast"], [])


class TestAstronomyProvider(unittest.TestCase):
    def test_ephemeris_structure_and_sanity(self):
        provider = AstronomyProvider(
            {"latitude": 40.4168, "longitude": -3.7038, "elevation_m": 667},
            ttl_seconds=900,
        )
        data = provider.fetch("ephemeris")

        sunrise = datetime.fromisoformat(data["sun"]["sunrise"])
        sunset = datetime.fromisoformat(data["sun"]["sunset"])
        self.assertLess(sunrise, sunset)
        self.assertIsInstance(data["sun"]["is_up"], bool)
        self.assertGreaterEqual(data["sun"]["azimuth_now"], 0)
        self.assertLess(data["sun"]["azimuth_now"], 360)
        self.assertGreaterEqual(data["sun"]["clear_sky_irradiance_wm2"], 0)
        self.assertLess(data["sun"]["clear_sky_irradiance_wm2"], 1200)
        if not data["sun"]["is_up"]:
            self.assertEqual(data["sun"]["clear_sky_irradiance_wm2"], 0)

        # Madrid always has astronomical darkness; dusk today < dawn tomorrow.
        dusk = datetime.fromisoformat(data["darkness"]["astronomical_dusk"])
        dawn = datetime.fromisoformat(data["darkness"]["astronomical_dawn"])
        self.assertLess(dusk, dawn)
        self.assertGreater(data["darkness"]["hours"], 0)

        moon = data["moon"]
        self.assertGreaterEqual(moon["phase_days"], 0)
        self.assertLess(moon["phase_days"], 28)
        self.assertGreaterEqual(moon["illumination"], 0)
        self.assertLessEqual(moon["illumination"], 1)
        self.assertIn(moon["phase_name"], [
            "New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
            "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent",
        ])

    def test_no_network_is_used(self):
        # The provider must stay computable offline: it imports astral only.
        import inspect

        from server.providers import astronomy

        self.assertNotIn("httpx", inspect.getsource(astronomy))


if __name__ == "__main__":
    unittest.main()
