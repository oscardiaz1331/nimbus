"""Typed configuration objects loaded from ``config.yaml``.

Single source of truth for the API server, storage paths, external data
providers, and the dashboard layout. Mirrors ``academy/utils/config.py``:
plain dataclasses, validation in ``__post_init__``, fail loudly at load
time. Relative paths are resolved against the config file's directory so
the same loader works for the real ``web/config.yaml`` and for the
temporary configs the tests generate.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    static_dir: Path = Path("frontend/dist")

    def __post_init__(self) -> None:
        if not (0 < self.port < 65536):
            raise ValueError(f"server.port must be in (0, 65536), got {self.port}")


@dataclasses.dataclass
class StorageConfig:
    db_path: Path
    images_dir: Path


@dataclasses.dataclass
class StationConfig:
    id: str
    name: str
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("station.id must be a non-empty string")


@dataclasses.dataclass
class ProviderConfig:
    """One entry under ``providers:``.

    Attributes:
        name: Registry key and URL slug (e.g. "open-meteo").
        enabled: Disabled providers are never instantiated.
        ttl_seconds: Cache freshness window; the external API is hit at
            most once per window per resource.
        settings: Provider-specific options (lat/lon, API keys, ...),
            passed through untouched to the Provider implementation.
    """

    name: str
    enabled: bool = True
    ttl_seconds: int = 900
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("provider entries need a non-empty name")
        if self.ttl_seconds <= 0:
            raise ValueError(f"provider '{self.name}': ttl_seconds must be > 0")


@dataclasses.dataclass
class WidgetConfig:
    """One entry under ``dashboard.layout``: which widget, where, with what props.

    ``enabled: false`` hides the widget without losing its entry/props —
    the dashboard endpoint filters it out.
    """

    type: str
    title: str = ""
    span: int = 1
    enabled: bool = True
    props: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type or not isinstance(self.type, str):
            raise ValueError("dashboard.layout entries need a non-empty 'type'")
        if self.span < 1:
            raise ValueError(f"widget '{self.type}': span must be >= 1")


@dataclasses.dataclass
class SectionConfig:
    """One page of the dashboard (observatory-style navigation): its own
    widget layout under a nav entry. A bare ``dashboard.layout`` in the
    YAML is parsed as a single implicit section for backward compatibility."""

    id: str
    title: str
    icon: str = ""
    layout: list[WidgetConfig] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("dashboard sections need a non-empty 'id'")


@dataclasses.dataclass
class DashboardConfig:
    title: str = "Nimbus"
    refresh_seconds: int = 30
    ambient: bool = True  # full-page weather effects layer (rain/sun/clouds/stars)
    sections: list[SectionConfig] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.refresh_seconds <= 0:
            raise ValueError("dashboard.refresh_seconds must be > 0")
        ids = [s.id for s in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate dashboard section ids: {ids}")


@dataclasses.dataclass
class Config:
    server: ServerConfig
    storage: StorageConfig
    stations: list[StationConfig]
    providers: list[ProviderConfig]
    dashboard: DashboardConfig

    def __post_init__(self) -> None:
        if not self.stations:
            raise ValueError("at least one station must be declared under 'stations'")
        ids = [s.id for s in self.stations]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate station ids in config: {ids}")

    def station(self, station_id: str) -> StationConfig | None:
        return next((s for s in self.stations if s.id == station_id), None)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load, parse and validate a :class:`Config` from a YAML file."""
        path = Path(path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        base_dir = path.parent

        def resolve(p: str | Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else (base_dir / p).resolve()

        server_raw = raw.get("server", {})
        server = ServerConfig(
            host=server_raw.get("host", "0.0.0.0"),
            port=server_raw.get("port", 8080),
            static_dir=resolve(server_raw.get("static_dir", "frontend/dist")),
        )

        storage_raw = raw["storage"]
        storage = StorageConfig(
            db_path=resolve(storage_raw["db_path"]),
            images_dir=resolve(storage_raw["images_dir"]),
        )

        stations = [StationConfig(**s) for s in raw.get("stations") or []]

        providers = []
        for name, p in (raw.get("providers") or {}).items():
            p = dict(p)
            providers.append(
                ProviderConfig(
                    name=name,
                    enabled=p.pop("enabled", True),
                    ttl_seconds=p.pop("ttl_seconds", 900),
                    settings=p,
                )
            )

        dashboard_raw = dict(raw.get("dashboard", {}))
        sections_raw = dashboard_raw.pop("sections", None)
        layout_raw = dashboard_raw.pop("layout", None)
        if sections_raw:
            sections = []
            for s in sections_raw:
                s = dict(s)
                widgets = [WidgetConfig(**w) for w in s.pop("layout", []) or []]
                sections.append(SectionConfig(layout=widgets, **s))
        else:
            # Single-page config: wrap the bare layout in one implicit section.
            widgets = [WidgetConfig(**w) for w in layout_raw or []]
            sections = [SectionConfig(id="dashboard", title="Dashboard", layout=widgets)]
        dashboard = DashboardConfig(sections=sections, **dashboard_raw)

        return cls(
            server=server,
            storage=storage,
            stations=stations,
            providers=providers,
            dashboard=dashboard,
        )
