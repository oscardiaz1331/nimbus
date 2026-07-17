"""Typed loader for the list of checkpoints ``benchmark_checkpoints.py`` compares.

Deliberately not part of ``utils/config.py``: a trial is just enough to
override ``Config.framework``/``Config.model`` for one run, not a full
training/inference configuration in its own right.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from utils.config import Framework


@dataclasses.dataclass
class Trial:
    """One checkpoint to benchmark: enough to override ``Config.model``/``framework``.

    Attributes:
        name: Display name for the results table (e.g. "YOLO11-N-seg").
        framework: "yolo" | "rfdetr" — selects the adapter and which of
            ``variant``'s sibling fields (``model.yolo``/``model.rfdetr``) it sets.
        checkpoint: Path to the trained weights, resolved relative to the
            current working directory (run the script from the repo root).
        variant: Base architecture variant (e.g. "yolo11n-seg", "nano") —
            still required even with an explicit checkpoint, since each
            adapter needs it to pick the right model class before loading weights.
    """

    name: str
    framework: Framework
    checkpoint: str
    variant: str

    def __post_init__(self) -> None:
        try:
            self.framework = Framework(self.framework)
        except ValueError:
            raise ValueError(
                f"trial '{self.name}': framework must be one of {[f.value for f in Framework]}, got '{self.framework}'"
            ) from None


def load_trials(path: str | Path) -> list[Trial]:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return [Trial(**t) for t in raw["trials"]]
