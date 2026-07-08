"""Framework-agnostic logic for the 3-stage progressive training schedule.

Stage 1 (warm-up) and Stage 2 (intermediate) use a plateau detector to
decide when to *advance* to the next stage early. Stage 3 (full
fine-tune) is left to each framework's own native early-stopping
(Ultralytics ``patience=``, RF-DETR ``early_stopping_patience``) since
both already implement that correctly — re-deriving it here would be
duplicated effort for no benefit.
"""

from __future__ import annotations

import dataclasses
import enum


class StageAction(enum.Enum):
    CONTINUE = "continue"
    ADVANCE = "advance"  # plateaued mid-stage -> move to the next stage now


@dataclasses.dataclass
class PlateauDetector:
    """Tracks whether a monitored value has stopped improving.

    Lower is assumed better (e.g. validation loss). ``patience`` epochs
    without an improvement of at least ``min_delta`` triggers a plateau.
    """

    patience: int
    min_delta: float = 1e-3
    best: float | None = dataclasses.field(default=None, init=False)
    wait: int = dataclasses.field(default=0, init=False)

    def update(self, value: float) -> bool:
        """Feed in one new epoch's value; returns True if plateaued."""
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def reset(self) -> None:
        self.best = None
        self.wait = 0


class StageController:
    """Decides CONTINUE vs ADVANCE for the current non-final stage.

    Only used for stages before the last one in the schedule — the
    caller is responsible for not consulting it once the final stage
    (full fine-tune) has been reached, since that stage exits via the
    backend's native early stopping instead.
    """

    def __init__(self, patience: int, min_delta: float = 1e-3):
        self._detector = PlateauDetector(patience=patience, min_delta=min_delta)

    def step(self, val_metric: float) -> StageAction:
        return (
            StageAction.ADVANCE
            if self._detector.update(val_metric)
            else StageAction.CONTINUE
        )

    def reset(self) -> None:
        self._detector.reset()
