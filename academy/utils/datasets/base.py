"""Shared contract + shared mask -> YOLO-polygon conversion pipeline for every
cloud-segmentation dataset builder (Kontas, SWIMSEG/SWINSEG, ...).

Concrete builders only supply: where their (image, mask) pairs live, how to
binarize their own mask format into a 0/255 cloud mask, and how to split
pairs into named splits. Everything else — contour extraction, polygon
simplification, name-prefixed copying, dataset.yaml — lives here once.
"""

from __future__ import annotations

import abc
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils.datasets.config import DatasetBuilderConfig

Pair = tuple[Path, Path]  # (image_path, mask_path)
TRAIN_NAME = "train"
TRAIN_FOLDER = "train"
VALID_NAME = "val"
VALID_FOLDER = "valid"
TEST_NAME = "test"
TEST_FOLDER = "test"
SPLIT_DICT: dict[str, str] = {TRAIN_NAME : TRAIN_FOLDER, VALID_NAME : VALID_FOLDER, TEST_NAME : TEST_FOLDER}


class DatasetBuilder(abc.ABC):
    """Common contract + shared conversion pipeline for cloud-segmentation builders."""

    def __init__(self, config: DatasetBuilderConfig):
        self.cfg = config
        self.output_dir = Path(config.output_dir)

    # ---- per-dataset contract --------------------------------------------

    @abc.abstractmethod
    def find_pairs(self) -> list[Pair]:
        """Locate every (image, mask) pair this dataset provides."""

    @abc.abstractmethod
    def mask_to_binary(self, mask: np.ndarray) -> np.ndarray:
        """Convert this dataset's raw grayscale mask into a 0/255 cloud mask."""

    @abc.abstractmethod
    def split(self, pairs: list[Pair]) -> dict[str, list[Pair]]:
        """Partition pairs into named splits (subset of ``SPLIT_DICT``)."""

    # ---- shared pipeline ---------------------------------------------------

    def mask_to_yolo_segmentation(self, mask_path: Path) -> list[str]:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARNING: could not read mask: {mask_path}")
            return []

        h, w = mask.shape
        binary = self.mask_to_binary(mask)
        if binary.max() == 0:
            return []

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        lines = []
        for contour in contours:
            if len(contour) < 3:
                continue  # degenerate contour — skip
            if cv2.contourArea(contour) < self.cfg.min_contour_area:
                continue

            epsilon = self.cfg.contour_epsilon * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) < 3:
                continue  # simplification collapsed the polygon — skip

            points = simplified.reshape(-1, 2).astype(np.float64)
            points[:, 0] /= w
            points[:, 1] /= h
            points = np.clip(points, 0.0, 1.0)

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            lines.append(f"{self.cfg.class_id} {coords}")

        return lines

    def process_sample(
        self, image_path: Path, mask_path: Path, dst_image_dir: Path, dst_label_dir: Path
    ) -> bool:
        """Copy one image + write its YOLO label file, both name-prefixed by
        this dataset's name so a later merge across datasets never collides.

        Returns True if the image contains at least one cloud polygon.
        """
        shutil.copy2(image_path, dst_image_dir / f"{self.cfg.name}_{image_path.name}")

        label_lines = self.mask_to_yolo_segmentation(mask_path)
        label_path = dst_label_dir / f"{self.cfg.name}_{image_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        return bool(label_lines)

    def make_split_dirs(self, SPLIT_DICT) -> dict[str, tuple[Path, Path]]:
        dirs = {}
        for split_name, split_folder in SPLIT_DICT.items():
            img_dir = self.output_dir / split_folder / "images"
            lbl_dir = self.output_dir / split_folder / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            dirs[split_name] = (img_dir, lbl_dir)
        return dirs

    def write_dataset_yaml(self, SPLIT_DICT) -> Path:
        yaml_dict = {
            "path": str(self.output_dir.resolve()),
            **{split_name: f"{split_folder}/images" for split_name, split_folder in SPLIT_DICT.items()},
            "nc": 1,
            "names": {self.cfg.class_id: self.cfg.class_name},
        }
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_dict, f, sort_keys=False)
        return yaml_path

    def collect_splits(self) -> dict[str, list[Pair]]:
        """Find this dataset's pairs and partition them into named splits."""
        pairs = self.find_pairs()
        if not pairs:
            raise RuntimeError(f"[{self.cfg.name}] no image/mask pairs found")
        return self.split(pairs)

    def process_splits(
        self, splits: dict[str, list[Pair]], dirs: dict[str, tuple[Path, Path]]
    ) -> tuple[dict[str, int], int]:
        """Convert+copy every pair in ``splits`` into the matching ``dirs``.

        ``dirs`` need not belong to this builder's own ``output_dir`` — the
        merge builder passes in its own combined split directories so member
        datasets are written straight to the merged output, never to their
        own standalone copy first.
        """
        stats: dict[str, int] = {}
        with_clouds = 0
        for split_name, split_pairs in splits.items():
            img_dir, lbl_dir = dirs[split_name]
            for image_path, mask_path in split_pairs:
                if self.process_sample(image_path, mask_path, img_dir, lbl_dir):
                    with_clouds += 1
            stats[split_name] = len(split_pairs)
            print(f"  {split_name:5s}: {len(split_pairs)} samples")
        return stats, with_clouds

    def build(self) -> dict[str, int]:
        """Run the full pipeline: find pairs -> split -> convert+copy -> data.yaml."""
        print(f"[{self.cfg.name}] building dataset -> {self.output_dir}")

        splits = self.collect_splits()
        dirs = self.make_split_dirs(splits.keys())
        stats, with_clouds = self.process_splits(splits, dirs)

        yaml_path = self.write_dataset_yaml(splits.keys())
        print(f"  with clouds: {with_clouds} / {sum(stats.values())}")
        print(f"  data.yaml -> {yaml_path}")
        return stats
