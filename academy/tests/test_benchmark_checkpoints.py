"""Tests for benchmark_checkpoints.py's pure row-building/recommendation
logic (no GPU, model, or ONNX runtime needed)."""

from __future__ import annotations

import unittest

from benchmark_checkpoints import _row_from_benchmark, recommend, render_summary_md
from utils.benchmark.benchmark import BenchmarkRow


def _bench_row(name: str, fps: float, iou: float, dice: float = None, accuracy: dict = None, vram_mb=100.0) -> BenchmarkRow:
    if accuracy is None:
        accuracy = {"mIoU": iou, "dice": dice if dice is not None else iou + 1}
    return BenchmarkRow(
        name=name,
        size_mb=10.0,
        fps=fps,
        latency_ms=1000.0 / fps,
        latency_std_ms=0.1,
        ram_mb=50.0,
        vram_mb=vram_mb,
        accuracy=accuracy,
    )


class TestRowFromBenchmark(unittest.TestCase):
    def test_maps_all_five_pixel_metrics(self):
        bench_row = _bench_row(
            "onnx-cpu", fps=50.0, iou=90.0,
            accuracy={"mIoU": 90.0, "dice": 91.0, "precision": 92.0, "recall": 88.0, "accuracy": 93.0},
        )
        row = _row_from_benchmark("YOLO11-N-seg", bench_row)
        self.assertEqual(row["model"], "YOLO11-N-seg")
        self.assertEqual(row["stage"], "onnx-cpu")
        self.assertEqual(row["iou"], 90.0)
        self.assertEqual(row["dice"], 91.0)
        self.assertEqual(row["precision"], 92.0)
        self.assertEqual(row["recall"], 88.0)
        self.assertEqual(row["accuracy"], 93.0)
        self.assertEqual(row["fps"], 50.0)

    def test_missing_keys_fall_back_to_none(self):
        """cfg.task != "segmentation": optimize.py's `evaluate` callable returns {}."""
        row = _row_from_benchmark("YOLO11-N-seg", _bench_row("onnx-cpu", fps=50.0, iou=90.0, accuracy={}))
        self.assertIsNone(row["iou"])
        self.assertIsNone(row["precision"])
        self.assertIsNone(row["recall"])
        self.assertIsNone(row["accuracy"])


class TestRecommend(unittest.TestCase):
    def test_picks_fastest_onnx_variant_within_budget(self):
        rows = [
            {"model": "A", "stage": "pytorch", "iou": 90.0, "fps": 40.0},
            {"model": "A", "stage": "onnx-cpu", "iou": 89.5, "fps": 60.0},
            {"model": "A", "stage": "int8-cpu", "iou": 80.0, "fps": 120.0},
        ]
        result = recommend(rows, max_accuracy_drop_pts=2.0)
        self.assertIn("onnx-cpu", result)
        self.assertNotIn("int8-cpu", result)

    def test_never_recommends_a_pytorch_row(self):
        rows = [{"model": "A", "stage": "pytorch", "iou": 90.0, "fps": 999.0}]
        result = recommend(rows, max_accuracy_drop_pts=2.0)
        self.assertIn("no onnx variants", result.lower())

    def test_falls_back_to_fastest_when_nothing_meets_budget(self):
        rows = [
            {"model": "A", "stage": "pytorch", "iou": 90.0, "fps": 40.0},
            {"model": "A", "stage": "int8-cpu", "iou": 50.0, "fps": 120.0},
        ]
        result = recommend(rows, max_accuracy_drop_pts=2.0)
        self.assertIn("int8-cpu", result)
        self.assertIn("budget", result.lower())

    def test_no_data_returns_placeholder(self):
        self.assertEqual(recommend([]), "Not enough data to make a recommendation.")


class TestRenderSummaryMd(unittest.TestCase):
    def test_renders_header_stages_and_missing_values(self):
        rows = [
            {
                "model": "A", "stage": "pytorch", "iou": 90.0, "dice": 92.0, "precision": 95.0,
                "recall": 91.0, "accuracy": 93.0, "fps": 40.0, "latency_ms": 25.0,
                "size_mb": 6.0, "vram_mb": 100.0,
            },
            {
                "model": "A", "stage": "onnx-cpu", "iou": 89.0, "dice": 91.0, "precision": None,
                "recall": None, "accuracy": None, "fps": 60.0, "latency_ms": 16.0,
                "size_mb": 6.0, "vram_mb": None,
            },
        ]
        md = render_summary_md(rows, max_accuracy_drop_pts=2.0)
        self.assertIn("| Model | Stage |", md)
        self.assertIn("pytorch", md)
        self.assertIn("onnx-cpu", md)
        self.assertIn("—", md)  # em dash for the missing onnx precision/recall/accuracy/vram
        self.assertIn("## Recommendation", md)


if __name__ == "__main__":
    unittest.main()
