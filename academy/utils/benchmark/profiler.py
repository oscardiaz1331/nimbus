"""Timing/throughput/memory profiling for a single model variant.

Framework-agnostic: takes any zero-arg callable that runs one inference
call. The caller wraps onnxruntime, ultralytics, or rfdetr into that shape,
so the same profiler works for the PyTorch baseline and every ONNX variant
without duplicating per-framework inference code here.
"""
from __future__ import annotations

import dataclasses
import statistics
import time
from typing import Callable

import psutil


@dataclasses.dataclass
class ProfileResult:
    mean_latency_ms: float
    std_latency_ms: float
    fps: float
    ram_mb: float
    vram_mb: float | None


def profile(run_once: Callable[[], None], warmup: int, iters: int) -> ProfileResult:
    """Time ``run_once`` after warming up, and sample process RAM/VRAM after warmup."""
    for _ in range(warmup):
        run_once()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)

    mean = statistics.mean(times)
    std = statistics.pstdev(times)

    return ProfileResult(
        mean_latency_ms=mean * 1000,
        std_latency_ms=std * 1000,
        fps=1 / mean if mean > 0 else float("inf"),
        ram_mb=_process_ram_mb(),
        vram_mb=_process_vram_mb(),
    )


def _process_ram_mb() -> float:
    """Resident set size of the current process, read straight from /proc."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
        return float("nan")
    except FileNotFoundError:
        # fallback for non-Linux platforms (e.g. Windows)
        return psutil.Process().memory_info().rss / (1024 ** 2)


def _process_vram_mb() -> float | None:
    """Current process' CUDA memory usage in MB, or None if no CUDA device."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 ** 2)