"""Self-check for benchmark/evaluator.py. Run directly: python test_evaluator.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.benchmark.evaluator import evaluate, mask_dice, mask_iou


def test_identical_masks_score_perfect() -> None:
    mask = np.array([[1, 1, 0], [0, 1, 0]])
    assert mask_iou(mask, mask) == 1.0
    assert mask_dice(mask, mask) == 1.0


def test_disjoint_masks_score_zero() -> None:
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[0, 1], [0, 0]])
    assert mask_iou(a, b) == 0.0
    assert mask_dice(a, b) == 0.0


def test_both_empty_defined_as_perfect() -> None:
    empty = np.zeros((2, 2))
    assert mask_iou(empty, empty) == 1.0
    assert mask_dice(empty, empty) == 1.0


def test_known_partial_overlap() -> None:
    # pred covers {0,1,2}, gt covers {1,2,3} out of 4 -> IoU = 2/4, Dice = 2*2/(3+3)
    pred = np.array([1, 1, 1, 0])
    gt = np.array([0, 1, 1, 1])
    assert mask_iou(pred, gt) == 0.5
    assert abs(mask_dice(pred, gt) - (4 / 6)) < 1e-9


def test_evaluate_averages_over_samples() -> None:
    samples = [("img1", np.array([1, 1, 0, 0])), ("img2", np.array([1, 1, 1, 1]))]
    lookup = {"img1": np.array([1, 1, 0, 0]), "img2": np.array([1, 1, 1, 1])}
    result = evaluate(samples, predict_mask=lambda s: lookup[s])
    assert result == {"mIoU": 100.0, "dice": 100.0}


if __name__ == "__main__":
    test_identical_masks_score_perfect()
    test_disjoint_masks_score_zero()
    test_both_empty_defined_as_perfect()
    test_known_partial_overlap()
    test_evaluate_averages_over_samples()
    print("ok")