import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from server.config import Config
from server.main import create_app

CONFIG_TEMPLATE = """
storage:
  db_path: "data/nimbus.db"
  images_dir: "data/images"
stations:
  - id: "st-1"
    name: "Test station"
    latitude: 1.0
    longitude: 2.0
dashboard:
  title: "Test"
  refresh_seconds: 10
  layout:
    - {{ type: "current-conditions", title: "Now", span: 1 }}
    - {{ type: "history-chart", title: "History", span: 2, props: {{ hours: 24 }} }}
{extra}
"""

# 2026-01-01T00:00:00Z — divisible by 600, so 10m buckets align on :00.
T0 = 1767225600


def make_observation(ts: str = "2026-01-01T00:00:00Z", **overrides) -> dict:
    body = {
        "schema_version": 1,
        "station_id": "st-1",
        "timestamp": ts,
        "cloud_cover": 0.62,
        "sky_fraction": 0.87,
        "classes": {"cloud": 0.62},
        "image": {"filename": "a.jpg", "mask_filename": "a_mask.png"},
        "model": {"name": "yolo26s-seg", "variant": "int8"},
        "inference_ms": 148.2,
    }
    body.update(overrides)
    return body


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        config_path = Path(self.tmp.name) / "config.yaml"
        config_path.write_text(CONFIG_TEMPLATE.format(extra=""), encoding="utf-8")
        self.app = create_app(Config.from_yaml(config_path))
        self.client = TestClient(self.app)


class TestIngestAndQuery(ApiTestCase):
    def test_ingest_then_latest_round_trip(self):
        created = self.client.post("/api/v1/observations", json=make_observation())
        self.assertEqual(created.status_code, 201)
        self.assertIn("id", created.json())

        latest = self.client.get("/api/v1/observations/latest")
        self.assertEqual(latest.status_code, 200)
        body = latest.json()
        self.assertEqual(body["station_id"], "st-1")
        self.assertEqual(body["timestamp"], "2026-01-01T00:00:00Z")
        self.assertAlmostEqual(body["cloud_cover"], 0.62)
        self.assertEqual(body["image_url"], "/images/a.jpg")
        self.assertEqual(body["mask_url"], "/images/a_mask.png")
        self.assertEqual(body["model"]["name"], "yolo26s-seg")

    def test_reingest_same_timestamp_upserts(self):
        self.client.post("/api/v1/observations", json=make_observation(cloud_cover=0.1))
        self.client.post("/api/v1/observations", json=make_observation(cloud_cover=0.9))
        rows = self.client.get(
            "/api/v1/observations",
            params={"start": "2025-12-31T00:00:00Z", "end": "2026-01-02T00:00:00Z"},
        ).json()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["cloud_cover"], 0.9)

    def test_unknown_station_404(self):
        r = self.client.post(
            "/api/v1/observations", json=make_observation(station_id="nope")
        )
        self.assertEqual(r.status_code, 404)

    def test_naive_timestamp_rejected(self):
        r = self.client.post(
            "/api/v1/observations", json=make_observation(ts="2026-01-01T00:00:00")
        )
        self.assertEqual(r.status_code, 422)

    def test_cloud_cover_out_of_range_rejected(self):
        r = self.client.post("/api/v1/observations", json=make_observation(cloud_cover=1.5))
        self.assertEqual(r.status_code, 422)

    def test_latest_without_observations_404(self):
        r = self.client.get("/api/v1/observations/latest")
        self.assertEqual(r.status_code, 404)


class TestSeries(ApiTestCase):
    def plant(self, offset_seconds: int, cloud_cover: float):
        from datetime import datetime, timezone

        ts = datetime.fromtimestamp(T0 + offset_seconds, tz=timezone.utc)
        r = self.client.post(
            "/api/v1/observations",
            json=make_observation(ts=ts.isoformat(), cloud_cover=cloud_cover),
        )
        self.assertEqual(r.status_code, 201)

    def test_bucket_boundaries_and_averages(self):
        self.plant(0, 0.2)      # bucket T0
        self.plant(300, 0.4)    # bucket T0 (00:05)
        self.plant(900, 0.6)    # bucket T0+600 (00:15)
        series = self.client.get(
            "/api/v1/observations/series",
            params={
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-01T01:00:00Z",
                "bucket": "10m",
            },
        ).json()
        self.assertEqual(series["bucket_seconds"], 600)
        self.assertEqual(series["ts"], [T0, T0 + 600])
        self.assertAlmostEqual(series["avg"][0], 0.3)
        self.assertAlmostEqual(series["avg"][1], 0.6)
        self.assertEqual(series["min"][0], 0.2)
        self.assertEqual(series["max"][0], 0.4)
        self.assertEqual(series["count"], [2, 1])

    def test_invalid_bucket_422(self):
        r = self.client.get("/api/v1/observations/series", params={"bucket": "diez"})
        self.assertEqual(r.status_code, 422)


class TestStationsDashboardHealth(ApiTestCase):
    def test_stations_last_seen(self):
        before = self.client.get("/api/v1/stations").json()
        self.assertEqual(before[0]["id"], "st-1")
        self.assertIsNone(before[0]["last_seen"])

        self.client.post("/api/v1/observations", json=make_observation())
        after = self.client.get("/api/v1/stations/st-1").json()
        self.assertEqual(after["last_seen"], "2026-01-01T00:00:00Z")

    def test_unknown_station_404(self):
        self.assertEqual(self.client.get("/api/v1/stations/nope").status_code, 404)

    def test_dashboard_layout_matches_config(self):
        body = self.client.get("/api/v1/dashboard").json()
        self.assertEqual(body["title"], "Test")
        self.assertEqual(body["refresh_seconds"], 10)
        self.assertTrue(body["ambient"])
        # bare layout arrives as one implicit section
        self.assertEqual(len(body["sections"]), 1)
        layout = body["sections"][0]["layout"]
        self.assertEqual([w["type"] for w in layout], ["current-conditions", "history-chart"])
        self.assertEqual(layout[1]["props"], {"hours": 24})

    def test_dashboard_filters_disabled_widgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                CONFIG_TEMPLATE.format(
                    extra='    - { type: "raw-observations", title: "Raw", enabled: false }\n'
                ),
                encoding="utf-8",
            )
            client = TestClient(create_app(Config.from_yaml(config_path)))
            layout = client.get("/api/v1/dashboard").json()["sections"][0]["layout"]
            types = [w["type"] for w in layout]
        self.assertNotIn("raw-observations", types)
        self.assertEqual(len(types), 2)

    def test_export_csv(self):
        self.client.post("/api/v1/observations", json=make_observation())
        self.client.post(
            "/api/v1/observations",
            json=make_observation(ts="2026-01-01T00:10:00Z", cloud_cover=0.4),
        )
        r = self.client.get(
            "/api/v1/observations/export.csv",
            params={"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/csv"))
        self.assertIn("attachment", r.headers["content-disposition"])
        lines = r.text.strip().splitlines()
        self.assertEqual(len(lines), 3)  # header + 2 rows, chronological
        self.assertTrue(lines[0].startswith("timestamp,station_id,cloud_cover"))
        self.assertIn("2026-01-01T00:00:00+00:00", lines[1])
        self.assertIn("0.4", lines[2])

    def test_health_ok(self):
        body = self.client.get("/api/v1/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["db_ok"])
        self.assertGreater(body["disk_free_mb"], 0)
        # system block always present; individual probes may be null on
        # exotic hosts, but on Linux these two must resolve
        system = body["system"]
        self.assertGreater(system["cpu_count"], 0)
        self.assertIsNotNone(system["mem_total_mb"])
        self.assertIn("cpu_temp_c", system)
        self.assertIn("uptime_hours", system)


if __name__ == "__main__":
    unittest.main()
