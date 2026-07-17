"""Self-check for utils/paths.py and utils/session.py. Run directly: python test_utils.py"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import paths
from utils.session import Session


def _fake_cfg(root: Path):
    return types.SimpleNamespace(
        output_dir=str(root),
        framework="yolo",
        variant="yolo26s-seg.pt",
    )


def test_paths_are_nested_under_optimize_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg(Path(tmp))
        assert paths.onnx_path(cfg) == paths.optimize_dir(cfg) / "model.onnx"
        assert paths.fp16_path(cfg).parent == paths.optimize_dir(cfg)
        assert paths.int8_path(cfg).parent == paths.optimize_dir(cfg)
        assert paths.plots_dir(cfg).parent == paths.report_dir(cfg)
        assert paths.optimize_dir(cfg).is_dir()  # created eagerly, not just computed


def test_session_matches_paths_module() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg(Path(tmp))
        session = Session.from_config(cfg)
        assert session.onnx == paths.onnx_path(cfg)
        assert session.fp16 == paths.fp16_path(cfg)
        assert session.int8 == paths.int8_path(cfg)
        assert session.cfg is cfg


def _fake_cfg_with_checkpoint(root: Path, checkpoint: str | None):
    return types.SimpleNamespace(
        output_dir=str(root),
        framework="yolo",
        variant="yolo11n-seg",
        model=types.SimpleNamespace(checkpoint=checkpoint),
    )


def test_resolve_checkpoint_explicit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "custom.pt"
        ckpt.touch()
        cfg = _fake_cfg_with_checkpoint(Path(tmp), str(ckpt))
        assert paths.resolve_checkpoint(cfg, ("best.pt",)) == ckpt


def test_resolve_checkpoint_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg_with_checkpoint(Path(tmp), None)
        wd = paths.weights_dir(cfg)  # nested under <output_dir>/<framework>/<variant>/weights
        (wd / "best.pt").touch()
        assert paths.resolve_checkpoint(cfg, ("best.pt", "last.pt")) == wd / "best.pt"


def test_resolve_checkpoint_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg_with_checkpoint(Path(tmp), None)
        try:
            paths.resolve_checkpoint(cfg, ("best.pt",))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError when no checkpoint exists")


if __name__ == "__main__":
    test_paths_are_nested_under_optimize_dir()
    test_session_matches_paths_module()
    test_resolve_checkpoint_explicit()
    test_resolve_checkpoint_fallback()
    test_resolve_checkpoint_missing()
    print("ok")