"""Canonical artifact locations for one training run's ``output_dir``.

Single source of truth so exporters/, benchmark/, reports/, and optimize.py
never re-derive — and risk disagreeing on — where a given artifact lives.
"""
from __future__ import annotations

from pathlib import Path

from utils.config import Config


def variant_dir(cfg: Config) -> Path:
    """Base dir for all artifacts of a given model variant (YOLO or RF-DETR)."""
    return Path(cfg.output_dir) / cfg.framework / cfg.variant

def optimize_dir(cfg: Config) -> Path:
    d = variant_dir(cfg) / "optimize"
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_dir(cfg: Config) -> Path:
    d =  variant_dir(cfg) / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_checkpoint(cfg: Config, candidate_names: tuple[str, ...]) -> Path:
    """Explicit cfg.model.checkpoint wins; otherwise search weights_dir() for the first match."""
    if cfg.model.checkpoint:
        path = Path(cfg.model.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"configured checkpoint not found: {path}")
        return path
    wd = weights_dir(cfg)
    for name in candidate_names:
        candidate = wd / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no checkpoint configured and none of {candidate_names} found in {wd}")


def onnx_path(cfg: Config) -> Path:
    return optimize_dir(cfg) / "model.onnx"


def fp16_path(cfg: Config) -> Path:
    return optimize_dir(cfg) / "model_fp16.onnx"


def int8_path(cfg: Config) -> Path:
    return optimize_dir(cfg) / "model_int8.onnx"


def report_dir(cfg: Config) -> Path:
    d = optimize_dir(cfg) / "optimization_report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plots_dir(cfg: Config) -> Path:
    d = report_dir(cfg) / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d