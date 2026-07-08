"""Embeds pipeline metadata into an ONNX model's ``metadata_props``.

Lets downstream consumers (benchmark/, deployment code) read framework,
task, input size, class names, etc. straight off the .onnx file instead of
needing the original config.yaml alongside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import onnx


def write_metadata(onnx_path: Path, metadata: dict) -> Path:
    """Set ``metadata`` as key/value pairs on the ONNX model, in place.

    Non-string values are JSON-encoded so callers don't have to pre-stringify
    lists/dicts (e.g. a class-names list) themselves.
    """
    model = onnx.load(str(onnx_path))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = str(key)
        entry.value = value if isinstance(value, str) else json.dumps(value)
    onnx.save(model, str(onnx_path))
    return onnx_path


def read_metadata(onnx_path: Path) -> dict:
    """Read back what :func:`write_metadata` wrote, JSON-decoding where possible."""
    model = onnx.load(str(onnx_path))
    result = {}
    for entry in model.metadata_props:
        try:
            result[entry.key] = json.loads(entry.value)
        except json.JSONDecodeError:
            result[entry.key] = entry.value
    return result