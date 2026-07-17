"""Static INT8 quantization of an ONNX graph using onnxruntime.

Static (calibration-based), not dynamic: YOLO/RF-DETR are conv-heavy, and
onnxruntime's dynamic quantization only touches MatMul/Gemm weights — it
barely quantizes Conv layers, so it wouldn't produce the FPS speedup this
pipeline's benchmark stage is meant to measure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image

from utils.onnx_utils import graph_input_size

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_COMPUTE_OPS = frozenset({"Conv", "ConvTranspose", "Gemm", "MatMul"})


def postprocess_tail_node_names(model: onnx.ModelProto) -> list[str]:
    """Names of the graph's pure-postprocessing tail: every node from which no
    Conv/ConvTranspose/Gemm/MatMul is reachable downstream.

    Quantizing that tail is all risk and no reward. It holds no math worth
    accelerating, but it's where YOLO's end-to-end export concatenates absolute
    pixel coordinates (0..imgsz), confidences (0..1), and mask coefficients
    into one output tensor — QDQ's single per-tensor int8 scale there is
    dictated by the coordinates (~imgsz/255 per step), which quantizes every
    confidence to 0 and turns the model into an empty-mask predictor (the
    IoU=1.5 / accuracy=48.1 int8 rows in SUMMARY.md). RF-DETR only dodged the
    same failure because its export happens to emit scores/boxes/masks as
    separate, uniformly-scaled outputs.

    Relies on ``graph.node`` being topologically sorted (guaranteed by the
    ONNX IR spec, and cmd_simplify re-runs onnxsim before this anyway).
    Control-flow subgraphs (If/Loop bodies) are not traversed — neither
    exporter emits them.
    """
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)
    reaches_compute: dict[int, bool] = {}
    for node in reversed(model.graph.node):
        reaches_compute[id(node)] = node.op_type in _COMPUTE_OPS or any(
            reaches_compute.get(id(consumer), False)
            for tensor in node.output
            for consumer in consumers.get(tensor, ())
        )
    return [node.name for node in model.graph.node if not reaches_compute[id(node)]]


def _ensure_node_names(model: onnx.ModelProto) -> None:
    """Give every unnamed node a unique name, in place.

    ``nodes_to_exclude`` matches nodes by name and ONNX allows empty names, so
    an unnamed tail node would silently stay quantizable without this.
    """
    taken = {node.name for node in model.graph.node if node.name}
    for i, node in enumerate(model.graph.node):
        if not node.name:
            name = f"{node.op_type}_{i}"
            while name in taken:
                name = "_" + name
            node.name = name
            taken.add(name)


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

    model = onnx.load(str(onnx_path))
    _ensure_node_names(model)
    nodes_to_exclude = postprocess_tail_node_names(model)
    print(f"Excluding {len(nodes_to_exclude)} postprocessing-tail nodes from quantization.")

    # onnxruntime warns that quant_pre_process (shape inference + graph
    # optimization) should run before quantize_static — skipped deliberately:
    # cmd_simplify already ran onnxsim.simplify() on this graph earlier in the
    # pipeline, which does its own full ONNX shape inference and constant
    # folding, populating value_info. Re-running shape inference here would
    # mostly repeat that work, and quant_pre_process's default symbolic shape
    # inference is a heavy pass on transformer-sized graphs (DINOv2/RF-DETR) —
    # the same category of cost that made fp16 conversion pathological.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ponytail: passing the (renamed-in-memory) ModelProto instead of a path is
    # supported by the installed onnxruntime's quantize_static signature
    # (model_input: str | Path | onnx.ModelProto) — re-verify on upgrade.
    quantize_static(
        model,
        str(output_path),
        reader,
        quant_format=QuantFormat.QDQ,
        # ponytail: per_channel must stay False on this onnxruntime (1.27). With
        # per_channel=True the QDQ quantizer gives MatMul weights per-column
        # scales/zero-points (operators/matmul.py, default_axis=1), and at load
        # time the CPU EP fuses DQ+MatMul into QLinearMatMul, whose kernel
        # requires a SCALAR weight zero point — the session builds fine and then
        # dies at run time on transformer graphs (RF-DETR: "weight zero point
        # must be a scalar" on /transformer/decoder/.../MatMul). Conv-only
        # graphs (YOLO) tolerate it, but a flag that bricks one backend's int8
        # at runtime isn't worth the accuracy nudge. Re-verify on ORT upgrade.
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
    )
    return output_path