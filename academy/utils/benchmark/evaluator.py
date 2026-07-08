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


def evaluate(samples: Iterable[tuple], predict_mask: Callable[[object], np.ndarray]) -> dict[str, float]:
    """Average mIoU/Dice (as 0-100) over ``samples`` of (input, ground_truth_mask) pairs.

    ``predict_mask`` takes one sample's input and returns its predicted
    binary foreground mask — framework-specific decoding lives there, not here.
    """
    ious, dices = [], []
    for sample_input, gt_mask in samples:
        pred_mask = predict_mask(sample_input)
        ious.append(mask_iou(pred_mask, gt_mask))
        dices.append(mask_dice(pred_mask, gt_mask))
    if not ious:
        raise ValueError("evaluate() got no samples")
    return {"mIoU": 100 * sum(ious) / len(ious), "dice": 100 * sum(dices) / len(dices)}