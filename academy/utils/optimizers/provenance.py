"""Collects provenance/context metadata for one exported ONNX variant.

Embedded into the .onnx via ``utils.optimizers.metadata.write_metadata`` so a
deployed model carries its own context (what produced it, from what data and
code, by whom) instead of needing the original config.yaml alongside it.

Everything here is best-effort: a value that can't be determined (no git
repo, no resolvable checkpoint, ...) becomes "unknown" (or None) rather than
raising — this is descriptive metadata, not a build requirement, and must
never be what breaks the optimize.py pipeline.
"""
from __future__ import annotations

import datetime
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import onnx
import yaml

from utils.commons import checkpoint_candidate_names, load_class_names
from utils.config import Config
from utils.paths import resolve_checkpoint

# utils/optimizers/provenance.py -> utils/optimizers/ -> utils/ -> academy root
_ACADEMY_ROOT = Path(__file__).resolve().parent.parent.parent


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_ACADEMY_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_provenance() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    return {
        "git_commit": commit or "unknown",
        "git_commit_short": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # Only meaningful if we actually found a commit; None (not False)
        # when there's no repo at all, so it doesn't get read as "clean".
        "git_dirty": bool(_git("status", "--porcelain")) if commit else None,
        "author_name": _git("config", "user.name") or "unknown",
        "author_email": _git("config", "user.email") or "unknown",
    }


def _checkpoint_provenance(cfg: Config) -> dict[str, Any]:
    try:
        checkpoint = resolve_checkpoint(cfg, checkpoint_candidate_names(cfg))
    except FileNotFoundError:
        return {
            "checkpoint_path": cfg.model.checkpoint or "unknown",
            "checkpoint_sha256": "unknown",
        }

    # Content hash, not a hand-maintained version string: uniquely identifies
    # the exact weights that produced this export without keeping a separate
    # version registry. Truncated (still ~2^64 collision space) since this is
    # an identifier, not a security digest.
    digest = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest.hexdigest()[:16],
    }


def _dataset_name(cfg: Config) -> str:
    """Best-effort match of ``cfg.dataset.root`` against a
    ``datasets/configs/*.yaml``'s ``output_dir`` — falls back to the root
    folder's own name if nothing matches (e.g. an ad-hoc dataset root)."""
    root = (_ACADEMY_ROOT / cfg.dataset.root).resolve()
    configs_dir = _ACADEMY_ROOT / "datasets" / "configs"
    if configs_dir.is_dir():
        for config_path in configs_dir.glob("*.yaml"):
            try:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (yaml.YAMLError, OSError):
                continue
            output_dir = raw.get("output_dir") if raw else None
            if output_dir and (_ACADEMY_ROOT / output_dir).resolve() == root:
                return raw["name"]
    return Path(cfg.dataset.root).name


def _class_names(cfg: Config) -> list[str]:
    try:
        names_by_id = load_class_names(cfg.dataset.yaml_path)
    except (OSError, KeyError):
        return []
    return [names_by_id[i] for i in sorted(names_by_id)]


def _graph_input_size(model: onnx.ModelProto) -> int | None:
    try:
        dim = model.graph.input[0].type.tensor_type.shape.dim[2]
    except IndexError:
        return None
    return dim.dim_value or None  # 0 means a dynamic dim


def collect_export_metadata(cfg: Config, onnx_path: Path) -> dict[str, Any]:
    """Everything worth knowing about one exported .onnx variant: what model
    produced it, from what code/data/weights, and by whom.

    Call this again after every stage that (re)writes an onnx file (export,
    simplify, fp16, int8) — see write_metadata's docstring for why this isn't
    embedded just once and left to survive downstream conversions.
    """
    model = onnx.load(str(onnx_path))
    class_names = _class_names(cfg)

    metadata: dict[str, Any] = {
        "framework": cfg.framework.value,
        "task": cfg.task.value,
        "variant": cfg.variant,
        "imgsz": _graph_input_size(model),
        # Ultralytics YOLO export bakes NMS in iff model.yolo.export_nms (see
        # YoloAdapter.export_onnx); RF-DETR's query-based decoding has no
        # separate NMS stage at all regardless of config.
        "nms_embedded": cfg.framework.value == "yolo" and cfg.model.yolo.export_nms,
        "onnx_opset": model.opset_import[0].version if model.opset_import else None,
        "num_classes": len(class_names),
        "class_names": class_names,
        "dataset_name": _dataset_name(cfg),
        "dataset_root": cfg.dataset.root,
        "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    metadata.update(_git_provenance())
    metadata.update(_checkpoint_provenance(cfg))
    return metadata
