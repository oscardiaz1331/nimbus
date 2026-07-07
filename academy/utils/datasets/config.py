"""Typed configuration for dataset-builder scripts, loaded from a per-dataset
YAML file (mirrors how ``utils/config.py`` drives train.py/infer.py — one
YAML per concern instead of hardcoded constants in each script).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

VALID_KINDS = {"kontas", "binary_mask", "merge"}


@dataclasses.dataclass
class SplitRatios:
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


@dataclasses.dataclass
class KontasSourceConfig:
    """Source layout for the Kontas-2017 / Zenodo-16647156 dataset.

    ``test_root`` is optional: the 2021 multi-camera test set lives in a
    separate directory tree (one subfolder per camera under both
    ``images/`` and ``seg_masks/``), with image/mask pairs matched by
    stem across a JPEG/PNG extension mismatch — so it's kept distinct
    from the train/val ``root`` rather than folded into the same fields.
    """

    root: str = "datasets/16647156/kontas_2017"
    images_subdir: str = "images"
    masks_subdir: str = "seg_masks"
    validation_csv: str = "validation.csv"
    cloud_values: tuple[int, ...] = (2, 3, 4)
    test_root: str | None = None
    test_images_subdir: str = "images"
    test_masks_subdir: str = "seg_masks"


@dataclasses.dataclass
class BinaryMaskSourceConfig:
    """Source layout for datasets whose ground truth is a single binary
    (sky/cloud) mask per image — SWIMSEG, SWINSEG, and alike."""

    root: str = ""
    images_subdir: str = "images"
    masks_subdir: str = "GTmaps"
    mask_suffixes: list[str] = dataclasses.field(
        default_factory=lambda: ["_GT.png", "_GT.jpg"]
    )
    image_extensions: list[str] = dataclasses.field(
        default_factory=lambda: [".png", ".jpg"]
    )
    split_ratios: SplitRatios = dataclasses.field(default_factory=SplitRatios)
    random_seed: int = 42


@dataclasses.dataclass
class DatasetBuilderConfig:
    kind: str
    name: str
    output_dir: str
    class_id: int = 0
    class_name: str = "cloud"
    contour_epsilon: float = 0.001
    min_contour_area: float = 0.0
    kontas: KontasSourceConfig | None = None
    binary_mask: BinaryMaskSourceConfig | None = None
    members: list[str] = dataclasses.field(default_factory=list)  # kind == "merge" only

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got '{self.kind}'")
        if self.kind == "kontas" and self.kontas is None:
            raise ValueError(f"'{self.name}': kind=kontas requires a 'kontas:' section")
        if self.kind == "binary_mask" and self.binary_mask is None:
            raise ValueError(
                f"'{self.name}': kind=binary_mask requires a 'binary_mask:' section"
            )
        if self.kind == "merge" and not self.members:
            raise ValueError(f"'{self.name}': kind=merge requires a non-empty 'members:' list")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetBuilderConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        kontas_raw = raw.get("kontas")
        binary_mask_raw = raw.get("binary_mask")

        members = raw.get("members", [])
        if members:
            # Member paths are written relative to the config file that lists them.
            members = [str((path.parent / m).resolve()) for m in members]

        return cls(
            kind=raw["kind"],
            name=raw["name"],
            output_dir=raw["output_dir"],
            class_id=raw.get("class_id", 0),
            class_name=raw.get("class_name", "cloud"),
            contour_epsilon=raw.get("contour_epsilon", 0.001),
            min_contour_area=raw.get("min_contour_area", 0.0),
            kontas=KontasSourceConfig(**kontas_raw) if kontas_raw else None,
            binary_mask=(
                BinaryMaskSourceConfig(
                    **{
                        **{k: v for k, v in binary_mask_raw.items() if k != "split_ratios"},
                        "split_ratios": SplitRatios(**binary_mask_raw["split_ratios"])
                        if "split_ratios" in binary_mask_raw
                        else SplitRatios(),
                    }
                )
                if binary_mask_raw
                else None
            ),
            members=members,
        )


def load_member_configs(config: DatasetBuilderConfig) -> list[DatasetBuilderConfig]:
    """Resolve a merge config's ``members:`` list into their own configs."""
    return [DatasetBuilderConfig.from_yaml(member_path) for member_path in config.members]
