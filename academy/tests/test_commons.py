"""Tests for utils/commons.py — pure-Python/numpy logic, no GPU needed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.commons import (
    FPSBenchmark,
    compute_classification_metrics,
    compute_segmentation_metrics,
    rasterize_polygon_mask,
)


class TestSegmentationMetrics(unittest.TestCase):
    def test_perfect_overlap(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        m = compute_segmentation_metrics(mask, mask)
        self.assertAlmostEqual(m["iou"], 1.0)
        self.assertAlmostEqual(m["dice"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)

    def test_no_overlap(self):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0:3, 0:3] = 1
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[7:10, 7:10] = 1
        m = compute_segmentation_metrics(gt, pred)
        self.assertAlmostEqual(m["iou"], 0.0)
        self.assertAlmostEqual(m["precision"], 0.0)
        self.assertAlmostEqual(m["recall"], 0.0)

    def test_both_empty_is_perfect_not_nan(self):
        empty = np.zeros((10, 10), dtype=np.uint8)
        m = compute_segmentation_metrics(empty, empty)
        for v in m.values():
            self.assertEqual(v, 1.0)


class TestRasterizePolygonMask(unittest.TestCase):
    def test_square_polygon_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "img.txt"
            # A unit-square polygon covering the central half of a 10x10 image.
            label_path.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
            mask = rasterize_polygon_mask(label_path, img_h=10, img_w=10, num_classes=1)
            self.assertEqual(mask.shape, (1, 10, 10))
            self.assertGreater(mask[0].sum(), 0)
            self.assertEqual(mask[0, 0, 0], 0)  # corner outside the polygon stays empty
            self.assertEqual(mask[0, 5, 5], 1)  # center inside the polygon is filled

    def test_missing_label_file_returns_zero_mask(self):
        mask = rasterize_polygon_mask(
            Path("/nonexistent/label.txt"), 10, 10, num_classes=2
        )
        self.assertEqual(mask.sum(), 0)


class TestClassificationMetrics(unittest.TestCase):
    def test_all_correct(self):
        y = np.array([0, 1, 2, 1, 0])
        m = compute_classification_metrics(y, y, num_classes=3)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_all_wrong_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_classification_metrics(y_true, y_pred, num_classes=2)
        self.assertAlmostEqual(m["accuracy"], 0.0)


class TestFPSBenchmark(unittest.TestCase):
    def test_calls_warmup_plus_iters_times(self):
        calls = {"n": 0}

        def fake_predict():
            calls["n"] += 1

        bench = FPSBenchmark(warmup=3, iters=7)
        fps = bench.run(fake_predict)
        self.assertEqual(calls["n"], 10)
        self.assertGreater(fps, 0)


if __name__ == "__main__":
    unittest.main()
