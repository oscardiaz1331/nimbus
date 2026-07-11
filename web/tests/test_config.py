import tempfile
import unittest
from pathlib import Path

from server.config import Config

MINIMAL_YAML = """
storage:
  db_path: "data/test.db"
  images_dir: "data/images"
stations:
  - id: "st-1"
    name: "Test station"
providers:
  open-meteo:
    enabled: true
    ttl_seconds: 600
    latitude: 1.0
    longitude: 2.0
dashboard:
  refresh_seconds: 15
  layout:
    - { type: "current-conditions", title: "Now", span: 1 }
"""


def write_config(tmpdir: str, text: str) -> Path:
    path = Path(tmpdir) / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestConfigLoad(unittest.TestCase):
    def test_minimal_config_loads_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config.from_yaml(write_config(tmp, MINIMAL_YAML))
            self.assertEqual(cfg.server.port, 8080)  # default
            self.assertTrue(cfg.storage.db_path.is_absolute())
            self.assertEqual(cfg.storage.db_path, (Path(tmp) / "data/test.db").resolve())
            self.assertEqual(cfg.stations[0].id, "st-1")
            self.assertEqual(cfg.dashboard.refresh_seconds, 15)
            # a bare layout becomes one implicit section
            self.assertEqual(len(cfg.dashboard.sections), 1)
            self.assertEqual(cfg.dashboard.sections[0].id, "dashboard")
            self.assertEqual(cfg.dashboard.sections[0].layout[0].type, "current-conditions")

    def test_explicit_sections(self):
        yaml_text = MINIMAL_YAML.replace(
            """dashboard:
  refresh_seconds: 15
  layout:
    - { type: "current-conditions", title: "Now", span: 1 }
""",
            """dashboard:
  refresh_seconds: 15
  sections:
    - id: "overview"
      title: "Dashboard"
      icon: "H"
      layout:
        - { type: "current-conditions", title: "Now" }
    - id: "sky"
      title: "Sky"
      layout:
        - { type: "allsky-image", title: "Image" }
""",
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config.from_yaml(write_config(tmp, yaml_text))
        self.assertEqual([s.id for s in cfg.dashboard.sections], ["overview", "sky"])
        self.assertEqual(cfg.dashboard.sections[0].icon, "H")
        self.assertEqual(cfg.dashboard.sections[1].layout[0].type, "allsky-image")

    def test_duplicate_section_ids_rejected(self):
        yaml_text = MINIMAL_YAML.replace(
            "  layout:\n    - { type: \"current-conditions\", title: \"Now\", span: 1 }\n",
            "  sections:\n"
            "    - { id: \"a\", title: \"A\", layout: [] }\n"
            "    - { id: \"a\", title: \"B\", layout: [] }\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                Config.from_yaml(write_config(tmp, yaml_text))

    def test_provider_settings_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config.from_yaml(write_config(tmp, MINIMAL_YAML))
            provider = cfg.providers[0]
            self.assertEqual(provider.name, "open-meteo")
            self.assertEqual(provider.ttl_seconds, 600)
            # enabled/ttl_seconds are popped out; the rest stays in settings
            self.assertEqual(provider.settings, {"latitude": 1.0, "longitude": 2.0})

    def test_station_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config.from_yaml(write_config(tmp, MINIMAL_YAML))
            self.assertIsNotNone(cfg.station("st-1"))
            self.assertIsNone(cfg.station("nope"))


class TestConfigValidation(unittest.TestCase):
    def load(self, text: str) -> Config:
        with tempfile.TemporaryDirectory() as tmp:
            return Config.from_yaml(write_config(tmp, text))

    def test_bad_port_rejected(self):
        bad = MINIMAL_YAML + "\nserver:\n  port: 99999\n"
        with self.assertRaises(ValueError):
            self.load(bad)

    def test_nonpositive_ttl_rejected(self):
        bad = MINIMAL_YAML.replace("ttl_seconds: 600", "ttl_seconds: 0")
        with self.assertRaises(ValueError):
            self.load(bad)

    def test_no_stations_rejected(self):
        bad = MINIMAL_YAML.replace('  - id: "st-1"\n    name: "Test station"\n', "")
        with self.assertRaises(ValueError):
            self.load(bad)

    def test_widget_without_type_rejected(self):
        bad = MINIMAL_YAML.replace('type: "current-conditions"', 'type: ""')
        with self.assertRaises(ValueError):
            self.load(bad)

    def test_duplicate_station_ids_rejected(self):
        bad = MINIMAL_YAML.replace(
            'stations:\n  - id: "st-1"\n    name: "Test station"\n',
            'stations:\n  - id: "st-1"\n    name: "A"\n  - id: "st-1"\n    name: "B"\n',
        )
        with self.assertRaises(ValueError):
            self.load(bad)


if __name__ == "__main__":
    unittest.main()
