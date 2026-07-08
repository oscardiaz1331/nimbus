"""Typed configuration objects loaded from ``config.yaml``.

This is the single source of truth consumed by both ``train.py`` and
``infer.py``, so the two entry points can never silently drift apart on
preprocessing, model selection, or dataset paths.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

VALID_TASKS = {"segmentation", "classification"}
VALID_FRAMEWORKS = {"yolo", "rfdetr"}
VALID_FREEZE_MODES = {"backbone", "partial", "none"}


@dataclasses.dataclass
class StageConfig:
    """One phase of the 3-stage progressive training schedule.

    Attributes:
        name: Human-readable stage identifier (used for output dirs/logs).
        max_epochs: Hard cap on epochs for this stage.
        freeze: One of "backbone" (head only trains), "partial" (head +
            a tail fraction of backbone blocks), or "none" (full model).
        lr_factor: Multiplier applied to ``training.base_lr`` for this stage.
        patience: Epochs without improvement before a non-final stage
            advances early. Ignored on the final stage.
        unfreeze_fraction: Fraction of backbone blocks (from the end) to
            unfreeze when ``freeze == "partial"``.
        scheduler: LR scheduler used in this stage ("cosine" | "plateau").
        early_stopping_patience: Native early-stopping patience used only
            by the final stage to terminate the whole run.
    """

    name: str
    max_epochs: int
    freeze: str
    lr_factor: float = 1.0
    patience: int = 10
    unfreeze_fraction: float = 0.3
    scheduler: str = "cosine"
    early_stopping_patience: int = 20

    def __post_init__(self) -> None:
        if self.freeze not in VALID_FREEZE_MODES:
            raise ValueError(
                f"stage '{self.name}': freeze must be one of {VALID_FREEZE_MODES}, got '{self.freeze}'"
            )
        if self.max_epochs <= 0:
            raise ValueError(f"stage '{self.name}': max_epochs must be > 0")


@dataclasses.dataclass
class DatasetConfig:
    """YOLO-format dataset layout: ``root/images/<split>``, ``root/labels/<split>``."""

    root: str
    yaml_file: str = "data.yaml"
    test_split: str = "test"

    @property
    def yaml_path(self) -> Path:
        return Path(self.root) / self.yaml_file

    def images_dir(self, split: str) -> Path:
        return Path(self.root) / split / "images"

    def labels_dir(self, split: str) -> Path:
        return Path(self.root) / split / "labels"


@dataclasses.dataclass
class YoloModelConfig:
    variant: str = "yolo11m-seg.pt"
    imgsz: int = 608


@dataclasses.dataclass
class RFDETRModelConfig:
    variant: str = "small"  # "small" | "medium"
    segmentation: bool = True  # explicit task toggle requested by spec
    export_fallback: bool = False  # use the hand-rolled ONNX wrapper instead of rfdetr's native .export()


@dataclasses.dataclass
class ModelConfig:
    yolo: YoloModelConfig = dataclasses.field(default_factory=YoloModelConfig)
    rfdetr: RFDETRModelConfig = dataclasses.field(default_factory=RFDETRModelConfig)
    checkpoint: str | None = None


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 8
    grad_accum: int = 1
    optimizer: str = "AdamW"
    base_lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    plateau_monitor: str = "val_loss"
    plateau_min_delta: float = 1e-3
    stages: list[StageConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class InferenceConfig:
    conf_threshold: float = 0.25
    imgsz: int = 640
    max_overlay_images: int = 30
    max_eval_images: int = 50  # test images used for the optimize-pipeline accuracy comparison; <0 = all
    benchmark_warmup: int = 10
    benchmark_iters: int = 50


@dataclasses.dataclass
class Config:
    task: str
    framework: str
    project_name: str
    output_dir: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    augmentation: dict[str, Any] = dataclasses.field(default_factory=dict)
    plot_every: int = 5

    def __post_init__(self) -> None:
        if self.task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{self.task}'")
        if self.framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"framework must be one of {VALID_FRAMEWORKS}, got '{self.framework}'"
            )
        if self.framework == "rfdetr" and self.task == "classification":
            # rfdetr is a detection/segmentation library only — fail loudly
            # at config-load time instead of silently training the wrong thing.
            raise ValueError(
                "framework='rfdetr' does not support task='classification'. "
                "Use framework='yolo' for classification, or task='segmentation' for rfdetr."
            )
        if not self.training.stages:
            raise ValueError("training.stages must define at least one stage")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load, parse and validate a :class:`Config` from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        training_raw = dict(raw["training"])
        stages = [StageConfig(**s) for s in training_raw.pop("stages", [])]
        training = TrainingConfig(stages=stages, **training_raw)

        model_raw = raw.get("model", {})
        model = ModelConfig(
            yolo=YoloModelConfig(**model_raw.get("yolo", {})),
            rfdetr=RFDETRModelConfig(**model_raw.get("rfdetr", {})),
            checkpoint=model_raw.get("checkpoint"),
        )
        script_dir = (
            Path(__file__).resolve().parent.parent
        )  # utils/config.py -> utils/ -> project root
        output_path_raw = Path(raw["project"]["output_dir"])
        if output_path_raw.is_absolute():
            output_dir = output_path_raw
        else:
            output_dir = (script_dir / output_path_raw).resolve()

        return cls(
            task=raw["task"],
            framework=raw["framework"],
            project_name=raw["project"]["name"],
            output_dir=output_dir,
            seed=raw["project"].get("seed", 42),
            dataset=DatasetConfig(**raw["dataset"]),
            model=model,
            training=training,
            inference=InferenceConfig(**raw.get("inference", {})),
            augmentation=raw.get("augmentation", {}),
            plot_every=raw.get("logging", {}).get("plot_every", 5),
        )
