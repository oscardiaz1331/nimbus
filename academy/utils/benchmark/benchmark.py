"""Assembles one comparable row per model variant: size, latency/fps/memory
(via profiler.py), and accuracy (via an injected callable — see evaluator.py).

Accuracy is injected rather than imported directly so this file doesn't need
to know how a given variant's accuracy is computed; it just needs a
zero-arg callable that returns a metric-name -> value dict.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from utils.benchmark.profiler import profile


@dataclasses.dataclass
class BenchmarkRow:
    name: str
    size_mb: float
    fps: float
    latency_ms: float
    latency_std_ms: float
    ram_mb: float
    vram_mb: float | None
    accuracy: dict[str, float]


def benchmark_variant(
    name: str,
    model_path: Path,
    run_once: Callable[[], None],
    evaluate: Callable[[], dict[str, float]],
    warmup: int = 10,
    iters: int = 50,
    vram_baseline_mb: float | None = None,
) -> BenchmarkRow:
    result = profile(run_once, warmup=warmup, iters=iters, vram_baseline_mb=vram_baseline_mb)
    return BenchmarkRow(
        name=name,
        size_mb=model_path.stat().st_size / (1024 ** 2),
        fps=result.fps,
        latency_ms=result.mean_latency_ms,
        latency_std_ms=result.std_latency_ms,
        ram_mb=result.ram_mb,
        vram_mb=result.vram_mb,
        accuracy=evaluate(),
    )