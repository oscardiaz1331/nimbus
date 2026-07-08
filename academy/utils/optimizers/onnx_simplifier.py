"""Runs onnx-simplifier over an exported ONNX graph (``pip install onnxsim``)."""
from __future__ import annotations

from pathlib import Path

import onnx
from onnxsim import simplify as onnxsim_simplify


def simplify(onnx_path: Path) -> Path:
    """Simplify the ONNX graph in place, overwriting ``onnx_path``."""
    

    model = onnx.load(str(onnx_path))
    simplified, ok = onnxsim_simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim could not validate the simplified model: {onnx_path}")
    onnx.save(simplified, str(onnx_path))
    return onnx_path