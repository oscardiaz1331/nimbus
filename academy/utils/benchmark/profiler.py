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


def profile(
    run_once: Callable[[], None], warmup: int, iters: int, vram_baseline_mb: float | None = None
) -> ProfileResult:
    """Time ``run_once`` after warming up, and sample process RAM after warmup.

    ``vram_baseline_mb`` is a ``device_vram_used_mb()`` reading the caller took
    *before* creating whatever this profiles (e.g. an onnxruntime session) —
    ``vram_mb`` is then the device-level growth from that baseline to the
    post-warmup sample, i.e. the footprint attributable to the profiled model.
    Without a baseline ``vram_mb`` is None; callers that can measure their own
    allocator directly (torch peak stats in ``infer.benchmark_fps``) should.
    """
    for _ in range(warmup):
        run_once()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)

    mean = statistics.mean(times)
    std = statistics.pstdev(times)

    vram_mb = None
    if vram_baseline_mb is not None:
        used = device_vram_used_mb()
        if used is not None:
            vram_mb = max(used - vram_baseline_mb, 0.0)

    return ProfileResult(
        mean_latency_ms=mean * 1000,
        std_latency_ms=std * 1000,
        fps=1 / mean if mean > 0 else float("inf"),
        ram_mb=_process_ram_mb(),
        vram_mb=vram_mb,
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


def device_vram_used_mb() -> float | None:
    """Total used VRAM across all NVIDIA devices as the driver reports it
    (NVML), or None when no driver/GPU is available.

    Deliberately NOT ``torch.cuda.memory_allocated()``: that only sees torch's
    own caching allocator, and onnxruntime's CUDA provider allocates outside
    it — which is how every ONNX row in SUMMARY.md (CPU rows included) used to
    report the same stale few MB of leftover torch memory. Per-process NVML
    accounting would be nicer than device-wide, but Windows' WDDM driver model
    can't attribute VRAM per process, so callers instead measure a
    before/after delta around whatever they want attributed (see ``profile``'s
    ``vram_baseline_mb``). Measurement must never kill a benchmark run, hence
    the broad excepts.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        total_bytes = sum(
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used
            for i in range(pynvml.nvmlDeviceGetCount())
        )
        return total_bytes / (1024 ** 2)
    except Exception:
        return None
    finally:
        pynvml.nvmlShutdown()