"""Small shared helpers for introspecting ONNX graphs."""
from __future__ import annotations

from pathlib import Path

import onnx


def graph_input_size(onnx_path: Path) -> tuple[str, int]:
    """Read the input name and spatial resolution an exported graph actually expects.

    A graph's baked-in resolution (e.g. RF-DETR exports at its checkpoint's own
    fixed resolution, independent of ``cfg.inference.imgsz``) is authoritative —
    trusting a separately-configured imgsz is what causes calibration/benchmark
    inputs to be built with the wrong shape and onnxruntime to reject them.
    """
    graph_input = onnx.load(str(onnx_path)).graph.input[0]
    dims = graph_input.type.tensor_type.shape.dim
    height = dims[2].dim_value
    if not height:
        raise ValueError(f"{onnx_path} has a dynamic input height — pass imgsz explicitly")
    return graph_input.name, height
