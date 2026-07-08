"""Raw-onnxruntime-output -> binary segmentation mask decoding, per framework.

Deliberately hand-rolled against the raw session outputs rather than routed
through ultralytics/rfdetr's own high-level ``.predict()`` — pointing
``ultralytics.YOLO()`` at an exported .onnx file works, but its AutoBackend
runs a dependency "AutoUpdate" on load that shells out to pip and can
silently clobber a carefully-installed ``onnxruntime-gpu`` in place (it did,
during development of this file). Decoding the raw arrays ourselves keeps
the benchmark's own onnxruntime session (and whatever execution provider it
was built with) as the only thing doing inference.

Output format per graph (see utils/models/*_adapter.py and
utils/models/rfdetr_onnx_wrapper.py for the exporters that produce these):

YOLO (Ultralytics end-to-end / NMS-free export):
    output0: (1, 300, 4 + 1 + 1 + nm) -- [x1, y1, x2, y2, conf, cls_id, *mask_coeffs],
             boxes already in absolute input-resolution pixel coords, already
             top-K filtered (no separate NMS pass needed).
    output1: (1, nm, mh, mw) -- mask prototypes.

RF-DETR (via RFDETRSegONNXWrapper):
    scores: (1, num_queries, num_classes + 1) -- per-class sigmoid probs
            (+1 padding/no-object slot beyond the configured num_classes).
    boxes:  (1, num_queries, 4) -- normalized cxcywh (unused here; masks are
            already full query-level maps, no crop-by-box needed).
    masks:  (1, num_queries, imgsz, imgsz) -- sigmoid probs, already
            upsampled to the model's input resolution by the wrapper.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess(image_bgr: np.ndarray, imgsz: int, normalize_imagenet: bool) -> np.ndarray:
    """BGR HWC uint8 -> normalized 1x3xHxW float32, matching int8_converter's calibration reader."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
    if normalize_imagenet:
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr[None]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_yolo_predict_mask(
    session: ort.InferenceSession, imgsz: int, conf_threshold: float, num_classes: int
) -> Callable[[np.ndarray], np.ndarray]:
    input_name = session.get_inputs()[0].name

    def predict_mask(image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x = _preprocess(image_bgr, imgsz, normalize_imagenet=False)
        dets, proto = session.run(None, {input_name: x})
        dets, proto = dets[0], proto[0]  # drop batch dim: (300, 6+nm), (nm, mh, mw)

        pred = np.zeros((num_classes, h, w), dtype=np.uint8)
        keep = dets[:, 4] > conf_threshold
        if not np.any(keep):
            return pred

        nm = proto.shape[0]
        proto_flat = proto.reshape(nm, -1)
        for box_conf_cls_coeffs in dets[keep]:
            x1, y1, x2, y2 = box_conf_cls_coeffs[:4]
            cls_id = int(box_conf_cls_coeffs[5])
            if cls_id >= num_classes:
                continue
            coeffs = box_conf_cls_coeffs[6:]
            mask = _sigmoid(coeffs @ proto_flat).reshape(proto.shape[1:])
            mask = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
            xi2, yi2 = min(int(x2), imgsz), min(int(y2), imgsz)
            cropped = np.zeros_like(mask)
            cropped[yi1:yi2, xi1:xi2] = mask[yi1:yi2, xi1:xi2]
            binary = (cropped > 0.5).astype(np.uint8)
            resized = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            pred[cls_id] |= resized
        return pred

    return predict_mask


def make_rfdetr_predict_mask(
    session: ort.InferenceSession, imgsz: int, conf_threshold: float, num_classes: int
) -> Callable[[np.ndarray], np.ndarray]:
    input_name = session.get_inputs()[0].name

    def predict_mask(image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x = _preprocess(image_bgr, imgsz, normalize_imagenet=True)
        scores, _boxes, masks = session.run(None, {input_name: x})
        scores, masks = scores[0], masks[0]  # (num_queries, C+1), (num_queries, imgsz, imgsz)

        pred = np.zeros((num_classes, h, w), dtype=np.uint8)
        for cls_id in range(num_classes):
            keep = scores[:, cls_id] > conf_threshold
            for mask in masks[keep]:
                binary = (mask > 0.5).astype(np.uint8)
                resized = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                pred[cls_id] |= resized
        return pred

    return predict_mask


def make_predict_mask(
    framework: str, session: ort.InferenceSession, imgsz: int, conf_threshold: float, num_classes: int
) -> Callable[[np.ndarray], np.ndarray]:
    if framework == "yolo":
        return make_yolo_predict_mask(session, imgsz, conf_threshold, num_classes)
    if framework == "rfdetr":
        return make_rfdetr_predict_mask(session, imgsz, conf_threshold, num_classes)
    raise ValueError(f"no ONNX mask decoder for framework: {framework}")
