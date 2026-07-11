"""Provider ABC — the extensibility contract for external data sources.

Routes never import a concrete provider, only this ABC plus the registry
(the ModelAdapter rule from academy applied here). A provider knows how
to fetch; freshness, caching and stale fallback live in the registry so
every provider gets them for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Provider(ABC):
    """One external data source (weather agency, astronomy, satellite...).

    Attributes:
        name: Registry key and URL slug (``/api/v1/providers/{name}/...``).
        resources: The resource names this provider can fetch; each one is
            served at ``/api/v1/providers/{name}/{resource}``.
    """

    name: str
    resources: tuple[str, ...] = ("data",)

    def __init__(self, settings: dict[str, Any], ttl_seconds: int) -> None:
        self.settings = settings
        self.ttl_seconds = ttl_seconds

    @abstractmethod
    def fetch(self, resource: str) -> dict:
        """Hit the external API and return a JSON-serializable payload.

        Raise on any failure — the registry decides whether a stale cached
        payload can cover for it. Payloads keep the external source's
        native units (the [0, 1] fraction convention applies to nimbus's
        own observations, not to third-party data).
        """
