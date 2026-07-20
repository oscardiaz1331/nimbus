"""Self-check for optimizers/. Run directly: python test_optimizers.py

Plain asserts, no pytest. Covers what's actually non-trivial and cheap to
check without heavy deps: metadata encode/decode roundtrip (needs only
onnx, already required by every file here), the calibration reader's
iteration/exhaustion logic (needs numpy+PIL, no onnxruntime call), and
pruner.prune_state_dict's magnitude math (needs only torch on CPU — no
GPU, no checkpoint I/O). onnx_simplifier / fp16_converter / quantize_static
and the adapters' checkpoint-level prune() are not exercised here: they
need onnxsim / onnxconverter-common installed and a real graph/checkpoint,
which is an integration test, not a unit self-check.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    Config,
    DatasetConfig,
    Framework,
    InferenceConfig,
    ModelConfig,
    StageConfig,
    Task,
    TrainingConfig,
)
from utils.optimizers.int8_converter import _ImageCalibrationReader
from utils.optimizers.metadata import read_metadata, write_metadata
from utils.optimizers.provenance import collect_export_metadata
from utils.optimizers.pruner import prune_state_dict


def _tiny_onnx_model():
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("Identity", ["x"], ["y"])
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([node], "g", [inp], [out])
    return helper.make_model(graph)


def test_metadata_roundtrip() -> None:
    import onnx

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        onnx.save(_tiny_onnx_model(), str(path))

        write_metadata(path, {"framework": "yolo", "imgsz": 640, "classes": ["a", "b"]})
        result = read_metadata(path)

        assert result["framework"] == "yolo"
        assert result["imgsz"] == 640
        assert result["classes"] == ["a", "b"]


def test_write_metadata_is_idempotent() -> None:
    """Re-writing overlapping keys must replace, not duplicate, entries —
    optimize.py calls write_metadata again after every pipeline stage
    (export/simplify/fp16/int8), all on the same handful of keys."""
    import onnx

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        onnx.save(_tiny_onnx_model(), str(path))

        write_metadata(path, {"framework": "yolo", "imgsz": 640})
        write_metadata(path, {"framework": "yolo", "imgsz": 608, "extra": "new"})

        model = onnx.load(str(path))
        keys = [entry.key for entry in model.metadata_props]
        assert sorted(keys) == ["extra", "framework", "imgsz"]  # no duplicates

        result = read_metadata(path)
        assert result["imgsz"] == 608  # second write wins
        assert result["extra"] == "new"


def _make_test_config(dataset_root: str, checkpoint: str | None) -> Config:
    return Config(
        task=Task.SEGMENTATION,
        framework=Framework.YOLO,
        project_name="test",
        output_dir="/tmp/does-not-matter",
        seed=42,
        dataset=DatasetConfig(root=dataset_root),
        model=ModelConfig(checkpoint=checkpoint),
        training=TrainingConfig(stages=[StageConfig(name="only", max_epochs=1, freeze="none")]),
        inference=InferenceConfig(),
    )


def test_collect_export_metadata_reads_dataset_checkpoint_and_git() -> None:
    import onnx

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        (dataset_root / "data.yaml").write_text("names:\n  - cloud\n  - sky\n")

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"fake-weights-content")
        expected_hash = hashlib.sha256(b"fake-weights-content").hexdigest()[:16]

        onnx_path = tmp_path / "model.onnx"
        onnx.save(_tiny_onnx_model(), str(onnx_path))

        cfg = _make_test_config(str(dataset_root), str(checkpoint))
        metadata = collect_export_metadata(cfg, onnx_path)

        assert metadata["framework"] == "yolo"
        assert metadata["task"] == "segmentation"
        assert metadata["nms_embedded"] is True
        assert metadata["class_names"] == ["cloud", "sky"]
        assert metadata["num_classes"] == 2
        assert metadata["checkpoint_sha256"] == expected_hash
        assert metadata["checkpoint_path"] == str(checkpoint)
        # Falls back to the folder's own name: nothing in datasets/configs/*.yaml
        # points its output_dir at this temp directory.
        assert metadata["dataset_name"] == "dataset"
        # Best-effort: either a real repo was found (40-char hex commit) or
        # git genuinely isn't available/configured here - both are valid.
        assert metadata["git_commit"] == "unknown" or len(metadata["git_commit"]) == 40


def test_collect_export_metadata_handles_missing_checkpoint() -> None:
    import onnx

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        onnx_path = tmp_path / "model.onnx"
        onnx.save(_tiny_onnx_model(), str(onnx_path))

        cfg = _make_test_config(str(tmp_path / "no-such-dataset"), checkpoint=None)
        metadata = collect_export_metadata(cfg, onnx_path)

        assert metadata["checkpoint_sha256"] == "unknown"
        assert metadata["class_names"] == []
        assert metadata["num_classes"] == 0


def test_calibration_reader_exhausts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image

        paths = []
        for i in range(3):
            p = Path(tmp) / f"img{i}.png"
            Image.new("RGB", (8, 8), color=(i * 10, 0, 0)).save(p)
            paths.append(p)

        reader = _ImageCalibrationReader(paths, input_name="images", imgsz=4)
        seen = 0
        while True:
            batch = reader.get_next()
            if batch is None:
                break
            assert set(batch.keys()) == {"images"}
            assert batch["images"].shape == (1, 3, 4, 4)
            assert batch["images"].max() <= 1.0
            seen += 1
        assert seen == 3


def test_prune_state_dict_zeroes_smallest_magnitude() -> None:
    import torch

    state_dict = {
        "conv.weight": torch.arange(1, 101, dtype=torch.float32).reshape(10, 10),
        "conv.bias": torch.ones(10),
        "bn.weight": torch.ones(10),
    }
    pruned, stats = prune_state_dict(state_dict, amount=0.3)

    assert stats.eligible_tensors == 1  # bias/bn excluded by name
    assert stats.total_params == 120
    assert pruned["conv.bias"].equal(state_dict["conv.bias"])  # untouched
    assert pruned["bn.weight"].equal(state_dict["bn.weight"])  # untouched
    zeroed = int((pruned["conv.weight"] == 0).sum())
    assert zeroed == stats.pruned_params
    assert 25 <= zeroed <= 35  # ~30% of 100, quantile-based so not exact
    # smallest-magnitude entries (top-left of the arange grid) are the ones zeroed
    assert pruned["conv.weight"][0, 0] == 0
    assert pruned["conv.weight"][-1, -1] != 0


def test_prune_state_dict_rejects_bad_amount() -> None:
    try:
        prune_state_dict({}, amount=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for amount >= 1.0")


if __name__ == "__main__":
    test_metadata_roundtrip()
    test_write_metadata_is_idempotent()
    test_collect_export_metadata_reads_dataset_checkpoint_and_git()
    test_collect_export_metadata_handles_missing_checkpoint()
    test_calibration_reader_exhausts()
    test_prune_state_dict_zeroes_smallest_magnitude()
    test_prune_state_dict_rejects_bad_amount()
    print("ok")