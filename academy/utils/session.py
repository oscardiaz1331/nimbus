"""One resolved set of artifact paths per pipeline run.

Computed once from Config and passed around, instead of every consumer
(benchmark/, reports/, optimize.py) re-deriving the same paths.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from utils import paths
from utils.config import Config


@dataclasses.dataclass
class Session:
    cfg: Config
    onnx: Path
    fp16: Path
    int8: Path
    report_dir: Path
    plots_dir: Path

    @classmethod
    def from_config(cls, cfg: Config) -> "Session":
        return cls(
            cfg=cfg,
            onnx=paths.onnx_path(cfg),
            fp16=paths.fp16_path(cfg),
            int8=paths.int8_path(cfg),
            report_dir=paths.report_dir(cfg),
            plots_dir=paths.plots_dir(cfg),
        )