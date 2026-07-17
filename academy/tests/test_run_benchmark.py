"""Tests for optimize.py::_run_benchmark's failure containment: a variant
whose session loads but blows up at run time (how a bad int8 quantization
actually fails — ORT validates some kernels only at execute time) must lose
its own row, not the whole trial's. onnxruntime is monkeypatched; the .onnx
files on disk are real-but-tiny graphs so graph_input_size can read them.
"""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import onnx
import onnxruntime
from onnx import TensorProto, helper

from optimize import _run_benchmark


def _save_tiny_onnx(path: Path) -> None:
    node = helper.make_node("Identity", ["input"], ["out"], name="id1")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 8, 8])
    onnx.save(helper.make_model(helper.make_graph([node], "g", [inp], [out])), str(path))


class _FakeSession:
    """Loads fine for every variant; running the int8 one raises — mirroring
    the QLinearMatMul zero-point failure mode."""

    def __init__(self, path: str, providers=None):
        self._path = path

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, _outputs, _feeds):
        if "int8" in self._path:
            raise RuntimeError("Non-zero status code returned while running QLinearMatMul node.")
        return []


class TestRunBenchmarkContainsVariantFailures(unittest.TestCase):
    def setUp(self):
        self._real_session = onnxruntime.InferenceSession
        self._real_providers = onnxruntime.get_available_providers
        onnxruntime.InferenceSession = _FakeSession
        onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]

    def tearDown(self):
        onnxruntime.InferenceSession = self._real_session
        onnxruntime.get_available_providers = self._real_providers

    def test_runtime_failure_in_one_variant_keeps_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path, int8_path = Path(tmp) / "model.onnx", Path(tmp) / "model-int8.onnx"
            _save_tiny_onnx(onnx_path)
            _save_tiny_onnx(int8_path)
            cfg = types.SimpleNamespace(
                task="classification",  # skips segmentation eval-sample loading
                inference=types.SimpleNamespace(benchmark_warmup=1, benchmark_iters=2),
            )
            session = types.SimpleNamespace(
                onnx=onnx_path, fp16=Path(tmp) / "missing-fp16.onnx", int8=int8_path
            )

            rows = _run_benchmark(cfg, session)

            self.assertEqual([r.name for r in rows], ["onnx-cpu"])


if __name__ == "__main__":
    unittest.main()
