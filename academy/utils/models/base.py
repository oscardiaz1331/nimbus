"""Abstract interface every training/inference backend must implement.

Keeping this thin is what lets ``train.py`` and ``infer.py`` stay
completely framework-agnostic: they only ever talk to this interface,
never to ``ultralytics`` or ``rfdetr`` directly.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any


class ModelAdapter(abc.ABC):
    """Common contract for the YOLO and RF-DETR backends."""

    @abc.abstractmethod
    def export_onnx(self, target: str) -> None:
        """Export the model to ONNX format."""

    @abc.abstractmethod
    def prune(self, checkpoint_path: Path, output_path: Path, amount: float) -> Path:
        """Magnitude-prune a checkpoint's weights, writing a new checkpoint file.

        Runs before ``export_onnx`` — pruning zeroes weights inside a live
        PyTorch checkpoint, which no longer exists as a distinct concept once
        the graph is frozen into ONNX. Each backend stores checkpoints in an
        incompatible shape (see the two adapters' implementations), so only the
        unwrap/rewrap is framework-specific here; the actual pruning math is
        shared via :func:`utils.optimizers.pruner.prune_state_dict`.
        """

    @abc.abstractmethod
    def detect_task(self) -> str:
        """Return the task this checkpoint/config actually performs
        (``"segment"``, ``"classify"``, or ``"detect"``)."""

    @abc.abstractmethod
    def apply_freeze(self, freeze_mode: str, unfreeze_fraction: float = 0.3) -> Any:
        """Freeze/unfreeze layers for ``"backbone"`` | ``"partial"`` | ``"none"``."""

    @abc.abstractmethod
    def run_stage(self, stage, plotter, stage_controller=None) -> dict[str, Any]:
        """Train for one stage of the 3-stage schedule; return stage metadata."""

    @abc.abstractmethod
    def predict(self, image) -> Any:
        """Raw single-image forward pass — used for FPS benchmarking."""

    @abc.abstractmethod
    def predict_segmentation_masks(self, image, num_classes: int) -> Any:
        """Return a ``(num_classes, H, W)`` binary prediction mask."""

    @abc.abstractmethod
    def predict_classification_label(self, image) -> int:
        """Return the predicted top-1 class id."""
