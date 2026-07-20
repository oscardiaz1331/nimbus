"""Shared, framework-agnostic utilities reused by both the segmentation
and classification pipelines: dataset I/O, metric computation, overlay
rendering, and inference-time helpers (FPS benchmarking, device choice).

Keeping this in one module is what lets ``train.py``/``infer.py`` and the
YOLO/RF-DETR adapters stay free of duplicated metric or plotting code.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import yaml


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return ``"cuda"`` if a GPU is available, else ``"cpu"``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def checkpoint_candidate_names(cfg) -> tuple[str, ...]:
    """Default checkpoint filenames to search for when ``cfg.model.checkpoint``
    isn't set explicitly, per framework's native save format."""
    return ("best.pt", "last.pt") if cfg.framework == "yolo" else ("best.pth", "last.pth")


def load_class_names(dataset_yaml: Path) -> dict[int, str]:
    """Read a YOLO-format dataset descriptor and return ``{class_id: name}``."""
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    return (
        {i: n for i, n in enumerate(names)}
        if isinstance(names, list)
        else {int(k): v for k, v in names.items()}
    )


# ---------------------------------------------------------------------
# Segmentation: ground truth + metrics
# ---------------------------------------------------------------------


def rasterize_polygon_mask(
    label_path: Path, img_h: int, img_w: int, num_classes: int
) -> np.ndarray:
    """Rasterize YOLO-polygon labels into a ``(num_classes, H, W)`` binary mask.

    Args:
        label_path: Path to a YOLO-format label file (``class_id x1 y1 x2 y2 ...``,
            normalized coordinates).
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        num_classes: Number of segmentation classes.

    Returns:
        A ``uint8`` mask of shape ``(num_classes, H, W)``. If the label file
        is missing or empty, an all-zero mask is returned — that represents
        "no annotated instance of any class in this image", not an error.
    """
    mask = np.zeros((num_classes, img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask

    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        if class_id >= num_classes:
            continue
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        cv2.fillPoly(mask[class_id], [coords.astype(np.int32)], 1)
    return mask


def compute_segmentation_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Pixel-level IoU / Dice / Precision / Recall / Accuracy for one class.

    Args:
        gt: Ground-truth binary mask, shape ``(H, W)``.
        pred: Predicted binary mask, same shape as ``gt``.

    Returns:
        A dict of float metrics. An image with no foreground pixels in
        either mask scores a perfect 1.0 for every metric rather than
        NaN — that is a true negative, not an undefined comparison.
    """
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    intersection = np.logical_and(gt_b, pred_b).sum()
    union = np.logical_or(gt_b, pred_b).sum()
    gt_sum, pred_sum = gt_b.sum(), pred_b.sum()

    if union == 0:
        return {
            "iou": 1.0,
            "dice": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "accuracy": 1.0,
        }

    return {
        "iou": float(intersection / union),
        "dice": (
            float(2 * intersection / (gt_sum + pred_sum))
            if (gt_sum + pred_sum) > 0
            else 1.0
        ),
        "precision": float(intersection / pred_sum) if pred_sum > 0 else 0.0,
        "recall": float(intersection / gt_sum) if gt_sum > 0 else 0.0,
        "accuracy": float((gt_b == pred_b).mean()),
    }


def save_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Save a 3-panel comparison figure: original | ground truth | TP/FN/FP.

    Color coding for the rightmost panel:
        Green  — True Positive  (GT and prediction agree on foreground)
        Red    — False Negative (missed by the model)
        Yellow — False Positive (hallucinated by the model)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_b, pred_b = gt_mask.astype(bool), pred_mask.astype(bool)
    tp = np.logical_and(gt_b, pred_b)
    fn = np.logical_and(gt_b, ~pred_b)
    fp = np.logical_and(~gt_b, pred_b)

    comparison = img_rgb.copy()
    comparison[tp] = (0.4 * comparison[tp] + 0.6 * np.array([0, 255, 0])).astype(
        np.uint8
    )
    comparison[fn] = (0.4 * comparison[fn] + 0.6 * np.array([255, 0, 0])).astype(
        np.uint8
    )
    comparison[fp] = (0.4 * comparison[fp] + 0.6 * np.array([255, 255, 0])).astype(
        np.uint8
    )

    gt_overlay = img_rgb.copy()
    gt_overlay[gt_b] = (0.5 * gt_overlay[gt_b] + 0.5 * np.array([0, 255, 0])).astype(
        np.uint8
    )
    pred_overlay = img_rgb.copy()
    pred_overlay[pred_b] = (
        0.5 * pred_overlay[pred_b] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, panel, panel_title in zip(
        axes,
        (gt_overlay, pred_overlay, comparison),
        ("Ground Truth", "Prediction", "TP green / FN red / FP yellow"),
    ):
        ax.imshow(panel)
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict[str, float]:
    """Macro-averaged accuracy / precision / recall / F1 from label arrays.

    # Ponytail: O(num_classes) confusion-matrix scan via boolean masks —
    # fine up to the hundreds of classes typical of these pipelines. For
    # very large label spaces, swap in sklearn.metrics for a vectorized
    # confusion-matrix-based version instead.
    """
    per_class_p, per_class_r, per_class_f1 = [], [], []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_p.append(p)
        per_class_r.append(r)
        per_class_f1.append(f1)

    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "precision": float(np.mean(per_class_p)) if per_class_p else 0.0,
        "recall": float(np.mean(per_class_r)) if per_class_r else 0.0,
        "f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def save_metrics_table(
    df: pd.DataFrame, out_dir: Path, group_col: str, metric_cols: list[str]
) -> pd.DataFrame:
    """Persist per-sample metrics and a mean +- std summary grouped by class.

    Returns the summary DataFrame so callers can also print or plot it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_sample_metrics.csv", index=False)
    summary = df.groupby(group_col)[metric_cols].agg(["mean", "std"])
    summary.to_csv(out_dir / "metrics_summary.csv")
    return summary


# ---------------------------------------------------------------------
# Inference benchmarking
# ---------------------------------------------------------------------


class FPSBenchmark:
    """Measures steady-state inference throughput, excluding warm-up cost.

    # Ponytail: single-process wall-clock timing. For multi-stream
    # serving throughput, replace with a real load test (e.g. locust) —
    # this is a single-model sanity benchmark only.
    """

    def __init__(self, warmup: int = 10, iters: int = 50):
        self.warmup = warmup
        self.iters = iters

    def run(self, predict_fn: Callable[[], None]) -> float:
        """Run ``predict_fn`` ``warmup + iters`` times; return frames/sec."""
        try:
            import torch

            sync = (
                torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
            )
        except ImportError:
            sync = lambda: None

        for _ in range(self.warmup):
            predict_fn()
        sync()

        start = time.perf_counter()
        for _ in range(self.iters):
            predict_fn()
        sync()
        elapsed = time.perf_counter() - start

        return self.iters / elapsed if elapsed > 0 else float("inf")
