"""Tests for utils/benchmark/onnx_decode.py's RF-DETR decoding against both
export contracts (native rfdetr `.export()` and RFDETRSegONNXWrapper), using a
fake onnxruntime session — no model or onnxruntime execution needed.
"""
from __future__ import annotations

import unittest

import numpy as np

from utils.benchmark.onnx_decode import make_predict_mask, make_rfdetr_predict_mask


class _FakeIO:
    def __init__(self, name: str):
        self.name = name


class _FakeSession:
    """Quacks like ort.InferenceSession for the parts the decoder touches."""

    def __init__(self, output_names: list[str], outputs: list[np.ndarray]):
        self._output_names = output_names
        self._outputs = outputs

    def get_inputs(self):
        return [_FakeIO("input")]

    def get_outputs(self):
        return [_FakeIO(n) for n in self._output_names]

    def run(self, _output_names, _feeds):
        return self._outputs


def _image(h: int = 8, w: int = 8) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _half_plane_logits(size: int, value: float = 4.0) -> np.ndarray:
    """Left half +value, right half -value."""
    mask = np.full((size, size), -value, dtype=np.float32)
    mask[:, : size // 2] = value
    return mask


class TestNativeContract(unittest.TestCase):
    """Native export: outputs (dets, labels, masks) — labels/masks are raw
    logits, masks at the segmentation head's low resolution."""

    def _session(self) -> _FakeSession:
        # Query 0: confident (logit +2 -> 0.88), left-half-positive 4x4 mask.
        # Query 1: background (logit -3 -> 0.05), all-positive mask that must
        # NOT leak into the union.
        dets = np.array([[[0.9, 0.5, 0.2, 0.2], [0.9, 0.5, 0.2, 0.2]]], dtype=np.float32)
        labels = np.array([[[2.0, -5.0], [-3.0, -5.0]]], dtype=np.float32)
        masks = np.stack([_half_plane_logits(4), np.full((4, 4), 4.0, dtype=np.float32)])[None]
        return _FakeSession(["dets", "labels", "masks"], [dets, labels, masks])

    def test_decodes_only_confident_queries(self):
        predict = make_rfdetr_predict_mask(self._session(), imgsz=8, conf_threshold=0.25, num_classes=1)
        pred = predict(_image())
        self.assertEqual(pred.shape, (1, 8, 8))
        # left half of the confident query's mask, bilinear-resized from 4x4
        np.testing.assert_array_equal(pred[0][:, :4], 1)
        # right half must stay empty: query 1's all-positive mask is excluded.
        # Before the contract fix, `dets` (cx=0.9 > 0.25) was read as scores,
        # which pulled query 1 in and painted the whole image.
        np.testing.assert_array_equal(pred[0][:, 4:], 0)

    def test_nothing_above_threshold_returns_empty(self):
        dets = np.array([[[0.9, 0.5, 0.2, 0.2]]], dtype=np.float32)
        labels = np.array([[[-4.0, -5.0]]], dtype=np.float32)  # sigmoid ~0.018
        masks = np.full((1, 1, 4, 4), 4.0, dtype=np.float32)
        session = _FakeSession(["dets", "labels", "masks"], [dets, labels, masks])
        predict = make_rfdetr_predict_mask(session, imgsz=8, conf_threshold=0.25, num_classes=1)
        np.testing.assert_array_equal(predict(_image()), 0)


class TestWrapperContract(unittest.TestCase):
    """RFDETRSegONNXWrapper: outputs (scores, boxes, masks) — already
    sigmoided, masks already upsampled to the input resolution."""

    def _session(self, mask_dtype=np.float32) -> _FakeSession:
        scores = np.array([[[0.9, 0.01], [0.05, 0.01]]], dtype=np.float32)
        boxes = np.zeros((1, 2, 4), dtype=np.float32)
        probs = 1.0 / (1.0 + np.exp(-_half_plane_logits(8)))
        masks = np.stack([probs, np.full((8, 8), 0.9, dtype=np.float32)])[None].astype(mask_dtype)
        return _FakeSession(["scores", "boxes", "masks"], [scores, boxes, masks])

    def test_decodes_sigmoided_masks_at_half_probability(self):
        predict = make_rfdetr_predict_mask(self._session(), imgsz=8, conf_threshold=0.25, num_classes=1)
        pred = predict(_image())
        np.testing.assert_array_equal(pred[0][:, :4], 1)
        np.testing.assert_array_equal(pred[0][:, 4:], 0)

    def test_float16_masks_do_not_crash_the_resize(self):
        # keep_io_types=False fp16 graphs emit fp16 tensors; cv2.resize can't
        # take float16, so the decoder must upcast before resizing.
        predict = make_rfdetr_predict_mask(
            self._session(mask_dtype=np.float16), imgsz=8, conf_threshold=0.25, num_classes=1
        )
        pred = predict(_image())
        np.testing.assert_array_equal(pred[0][:, :4], 1)


class TestContractDetection(unittest.TestCase):
    def test_unrecognized_output_names_raise(self):
        session = _FakeSession(["foo", "bar", "baz"], [])
        with self.assertRaises(ValueError):
            make_rfdetr_predict_mask(session, imgsz=8, conf_threshold=0.25, num_classes=1)

    def test_unknown_framework_raises(self):
        session = _FakeSession(["scores", "boxes", "masks"], [])
        with self.assertRaises(ValueError):
            make_predict_mask("caffe", session, imgsz=8, conf_threshold=0.25, num_classes=1)


if __name__ == "__main__":
    unittest.main()
