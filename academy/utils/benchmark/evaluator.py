"""IoU/Dice between a model's predicted binary mask and ground truth.

Deliberately semantic-level, not instance-level: the caller flattens every
predicted instance mask into one binary foreground mask (an OR across
instances) and this file compares it to a similarly-flattened ground-truth
mask. That sidesteps NMS/instance-matching entirely — overlapping
detections don't hurt a union — while still measuring real segmentation
quality per variant, which is all a quantization-degradation comparison
needs.

Framework-specific mask *decoding* (raw ONNX output -> that binary
foreground mask) is NOT this file's job — it's injected by the caller,
same pattern as benchmark.py's `evaluate` and profiler.py's `run_once`, so
this file stays framework-agnostic.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU between two same-shape masks (any truthy dtype)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union else 1.0  # both empty: define as a perfect match


def mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient between two same-shape masks (any truthy dtype)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * intersection / denom) if denom else 1.0


def mask_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """Precision (intersection / predicted-positive) between two same-shape masks.

    Mirrors ``utils.commons.compute_segmentation_metrics``'s precision exactly
    (both-empty -> 1.0, predicted-empty-but-gt-not -> 0.0) so pytorch and ONNX
    rows in ``benchmark_checkpoints.py`` stay directly comparable.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    pred_sum = pred.sum()
    if pred_sum == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(np.logical_and(pred, gt).sum() / pred_sum)


def mask_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Recall (intersection / ground-truth-positive) between two same-shape masks."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    gt_sum = gt.sum()
    if gt_sum == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(np.logical_and(pred, gt).sum() / gt_sum)


def mask_pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of matching pixels between two same-shape masks (any truthy dtype)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float((pred == gt).mean())


def evaluate(samples: Iterable[tuple], predict_mask: Callable[[object], np.ndarray]) -> dict[str, float]:
    """Average IoU/Dice/Precision/Recall/Accuracy (as 0-100) over ``samples``
    of (input, ground_truth_mask) pairs — the same five pixel metrics
    ``infer.py::run_segmentation_eval`` computes for the pytorch baseline, so
    ONNX rows are directly comparable to it in ``benchmark_checkpoints.py``'s
    combined table instead of only sharing IoU/Dice.

    ``predict_mask`` takes one sample's input and returns its predicted
    binary foreground mask — framework-specific decoding lives there, not here.
    """
    totals = {"mIoU": [], "dice": [], "precision": [], "recall": [], "accuracy": []}
    for sample_input, gt_mask in samples:
        pred_mask = predict_mask(sample_input)
        totals["mIoU"].append(mask_iou(pred_mask, gt_mask))
        totals["dice"].append(mask_dice(pred_mask, gt_mask))
        totals["precision"].append(mask_precision(pred_mask, gt_mask))
        totals["recall"].append(mask_recall(pred_mask, gt_mask))
        totals["accuracy"].append(mask_pixel_accuracy(pred_mask, gt_mask))
    if not totals["mIoU"]:
        raise ValueError("evaluate() got no samples")
    return {k: 100 * sum(v) / len(v) for k, v in totals.items()}