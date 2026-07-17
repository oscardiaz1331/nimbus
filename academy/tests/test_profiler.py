"""Tests for profiler.py's VRAM delta logic. device_vram_used_mb is stubbed —
the point is the baseline arithmetic, not NVML itself (which is a driver call
this suite can't rely on having).
"""
from __future__ import annotations

import unittest

from utils.benchmark import profiler


class TestProfileVram(unittest.TestCase):
    def setUp(self):
        self._real_reader = profiler.device_vram_used_mb

    def tearDown(self):
        profiler.device_vram_used_mb = self._real_reader

    def test_no_baseline_means_no_vram_reading(self):
        profiler.device_vram_used_mb = lambda: 9999.0  # must not even be consulted
        result = profiler.profile(lambda: None, warmup=1, iters=3)
        self.assertIsNone(result.vram_mb)
        self.assertGreater(result.fps, 0)

    def test_vram_is_growth_over_baseline(self):
        profiler.device_vram_used_mb = lambda: 532.0
        result = profiler.profile(lambda: None, warmup=1, iters=3, vram_baseline_mb=500.0)
        self.assertAlmostEqual(result.vram_mb, 32.0)

    def test_negative_growth_clamps_to_zero(self):
        # another process freeing memory mid-benchmark must not go negative
        profiler.device_vram_used_mb = lambda: 490.0
        result = profiler.profile(lambda: None, warmup=1, iters=3, vram_baseline_mb=500.0)
        self.assertEqual(result.vram_mb, 0.0)

    def test_unreadable_device_yields_none(self):
        profiler.device_vram_used_mb = lambda: None
        result = profiler.profile(lambda: None, warmup=1, iters=3, vram_baseline_mb=500.0)
        self.assertIsNone(result.vram_mb)


class TestDeviceVramUsedMb(unittest.TestCase):
    def test_returns_none_or_nonnegative_float(self):
        used = profiler.device_vram_used_mb()
        if used is not None:
            self.assertGreaterEqual(used, 0.0)


if __name__ == "__main__":
    unittest.main()
