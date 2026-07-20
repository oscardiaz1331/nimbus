"""Self-check for optimize.py. Run directly: python test_optimize.py"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimize import _COMMANDS, _list_images
from utils.commons import checkpoint_candidate_names


def test_list_images_filters_and_sorts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "b.jpg").touch()
        (d / "a.png").touch()
        (d / "notes.txt").touch()
        result = _list_images(d)
        assert [p.name for p in result] == ["a.png", "b.jpg"]


def test_checkpoint_candidate_names_match_framework() -> None:
    yolo_cfg = types.SimpleNamespace(framework="yolo")
    rfdetr_cfg = types.SimpleNamespace(framework="rfdetr")
    assert checkpoint_candidate_names(yolo_cfg) == ("best.pt", "last.pt")
    assert checkpoint_candidate_names(rfdetr_cfg) == ("best.pth", "last.pth")


def test_all_commands_are_callable() -> None:
    expected = {"export", "simplify", "fp16", "int8", "prune", "benchmark", "report", "pipeline"}
    assert set(_COMMANDS) == expected
    assert all(callable(fn) for fn in _COMMANDS.values())


if __name__ == "__main__":
    test_list_images_filters_and_sorts()
    test_checkpoint_candidate_names_match_framework()
    test_all_commands_are_callable()
    print("ok")