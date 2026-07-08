"""Static INT8 quantization of an ONNX graph using onnxruntime.

Static (calibration-based), not dynamic: YOLO/RF-DETR are conv-heavy, and
onnxruntime's dynamic quantization only touches MatMul/Gemm weights — it
barely quantizes Conv layers, so it wouldn't produce the FPS speedup this
pipeline's benchmark stage is meant to measure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image

from utils.onnx_utils import graph_input_size

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class _ImageCalibrationReader(CalibrationDataReader):
    # Preprocessing must match what the exporter baked into the graph:
    # YOLOExporter's graph expects raw 0-1 RGB, while RF-DETR (DINOv2 backbone)
    # expects ImageNet mean/std-normalized RGB on top of that (see
    # rfdetr.detr: means/stds = [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]).
    # Get this wrong and quantize_static won't raise — it'll just calibrate on
    # the wrong activation ranges and produce a poorly-calibrated int8 model.
    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int, normalize: bool = False):
        self._paths = iter(image_paths)
        self._input_name = input_name
        self._imgsz = imgsz
        self._normalize = normalize

    def get_next(self) -> dict | None:
        path = next(self._paths, None)
        if path is None:
            return None
        img = Image.open(path).convert("RGB").resize((self._imgsz, self._imgsz))
        arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        if self._normalize:
            arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        return {self._input_name: arr[None]}


def quantize_int8(
    onnx_path: Path,
    output_path: Path,
    calibration_images: list[Path],
    imgsz: int | None = None,
    max_calibration_images: int = 100,
    *,
    normalize_imagenet: bool = False,
) -> Path:
    """Static-quantize ``onnx_path`` to INT8, calibrating on real images.

    ``imgsz``, if given, is only a fallback for graphs with a dynamic input
    shape — otherwise the resolution is read straight off the graph so it can
    never drift from what the exporter actually produced.
    """

    if not calibration_images:
        raise ValueError("static int8 quantization needs at least one calibration image")

    print(f"Quantizing {onnx_path} to INT8 (max_calibration_images={max_calibration_images})...")
    input_name, graph_imgsz = graph_input_size(onnx_path)
    reader = _ImageCalibrationReader(
        calibration_images[:max_calibration_images], input_name, graph_imgsz or imgsz, normalize_imagenet
    )

    # onnxruntime warns that quant_pre_process (shape inference + graph
    # optimization) should run before quantize_static — skipped deliberately:
    # cmd_simplify already ran onnxsim.simplify() on this graph earlier in the
    # pipeline, which does its own full ONNX shape inference and constant
    # folding, populating value_info. Re-running shape inference here would
    # mostly repeat that work, and quant_pre_process's default symbolic shape
    # inference is a heavy pass on transformer-sized graphs (DINOv2/RF-DETR) —
    # the same category of cost that made fp16 conversion pathological.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(onnx_path),
        str(output_path),
        reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    return output_path