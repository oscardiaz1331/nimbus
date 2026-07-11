import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from server.config import Config
from server.main import create_app

CONFIG = """
storage:
  db_path: "data/nimbus.db"
  images_dir: "data/images"
stations:
  - id: "st-1"
    name: "Test station"
dashboard:
  layout: []
"""


def make_telemetry(ts: str = "2026-01-01T00:00:00Z", **overrides) -> dict:
    body = {
        "schema_version": 1,
        "node_id": "lora-rain-01",
        "timestamp": ts,
        "rssi_dbm": -97.5,
        "snr_db": 8.2,
        "battery_pct": 84.0,
        "battery_v": 3.9,
        "extra": {"uptime_s": 3600},
    }
    body.update(overrides)
    return body


class NodesTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        config_path = Path(self.tmp.name) / "config.yaml"
        config_path.write_text(CONFIG, encoding="utf-8")
        self.client = TestClient(create_app(Config.from_yaml(config_path)))

    def test_ingest_then_list_round_trip(self):
        created = self.client.post("/api/v1/nodes/telemetry", json=make_telemetry())
        self.assertEqual(created.status_code, 201)

        nodes = self.client.get("/api/v1/nodes").json()
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node["node_id"], "lora-rain-01")
        self.assertEqual(node["last_seen"], "2026-01-01T00:00:00Z")
        self.assertAlmostEqual(node["rssi_dbm"], -97.5)
        self.assertEqual(node["extra"], {"uptime_s": 3600})

    def test_list_returns_latest_row_per_node(self):
        self.client.post(
            "/api/v1/nodes/telemetry",
            json=make_telemetry("2026-01-01T00:00:00Z", battery_pct=90),
        )
        self.client.post(
            "/api/v1/nodes/telemetry",
            json=make_telemetry("2026-01-01T00:10:00Z", battery_pct=89),
        )
        self.client.post(
            "/api/v1/nodes/telemetry",
            json=make_telemetry(node_id="lora-wind-02", rssi_dbm=-110),
        )
        nodes = self.client.get("/api/v1/nodes").json()
        self.assertEqual([n["node_id"] for n in nodes], ["lora-rain-01", "lora-wind-02"])
        rain = nodes[0]
        self.assertEqual(rain["last_seen"], "2026-01-01T00:10:00Z")
        self.assertEqual(rain["battery_pct"], 89)

    def test_history_and_unknown_node_404(self):
        self.client.post("/api/v1/nodes/telemetry", json=make_telemetry())
        self.client.post(
            "/api/v1/nodes/telemetry", json=make_telemetry("2026-01-01T00:10:00Z")
        )
        history = self.client.get("/api/v1/nodes/lora-rain-01/telemetry").json()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["last_seen"], "2026-01-01T00:10:00Z")  # newest first

        self.assertEqual(
            self.client.get("/api/v1/nodes/nope/telemetry").status_code, 404
        )

    def test_reingest_same_timestamp_upserts(self):
        self.client.post("/api/v1/nodes/telemetry", json=make_telemetry(battery_pct=50))
        self.client.post("/api/v1/nodes/telemetry", json=make_telemetry(battery_pct=49))
        history = self.client.get("/api/v1/nodes/lora-rain-01/telemetry").json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["battery_pct"], 49)

    def test_naive_timestamp_rejected(self):
        r = self.client.post(
            "/api/v1/nodes/telemetry", json=make_telemetry(timestamp="2026-01-01T00:00:00")
        )
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
