"""Kontas-2017 (Zenodo 16647156) builder: multi-class mask -> single 'cloud' class.

Sky (original class 1) and the camera mask (0) are treated as background —
no annotation needed, since the model only needs to learn where clouds are.
Layers 2/3/4 (low/mid/high clouds) merge into one 'cloud' instance mask.

Optionally also pulls in the 2021 multi-camera test set (test_set/), which
lives in its own directory tree — one subfolder per camera under both
images/ and seg_masks/ — with images as .jpg and masks as .png sharing the
same filename stem.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from utils.datasets.base import TRAIN_NAME, VALID_NAME, TEST_NAME, DatasetBuilder, Pair
from utils.datasets.config import DatasetBuilderConfig

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class KontasBuilder(DatasetBuilder):
    def __init__(self, config: DatasetBuilderConfig):
        super().__init__(config)
        src = config.kontas
        self.src_root = Path(src.root)
        self.images_dir = self.src_root / src.images_subdir
        self.masks_dir = self.src_root / src.masks_subdir
        self.validation_csv = self.src_root / src.validation_csv
        self.cloud_values = set(src.cloud_values)

        self.test_root = Path(src.test_root) if src.test_root else None
        self.test_images_dir = (
            self.test_root / src.images_subdir if self.test_root else None
        )
        self.test_masks_dir = (
            self.test_root / src.masks_subdir if self.test_root else None
        )
        # Populated as a side effect of find_pairs(); split() folds it in as
        # the "test" split. It isn't threaded through find_pairs()'s return
        # value because it comes from a differently-shaped source tree
        # (per-camera subfolders) that split() has no business re-deriving.
        self._test_pairs: list[Pair] = []

    def mask_to_binary(self, mask: np.ndarray) -> np.ndarray:
        return np.isin(mask, list(self.cloud_values)).astype(np.uint8) * 255

    def find_pairs(self) -> list[Pair]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"masks dir not found: {self.masks_dir}")

        pairs = []
        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            mask_path = self._find_mask(self.masks_dir, image_path.stem)
            if mask_path is None:
                print(f"  WARNING: no mask for '{image_path.name}' — skipping.")
                continue
            pairs.append((image_path, mask_path))

        self._test_pairs = self._find_test_pairs()
        return pairs

    def _find_test_pairs(self) -> list[Pair]:
        if self.test_root is None:
            return []
        if not self.test_images_dir.exists():
            print(f"  WARNING: test images dir not found: {self.test_images_dir}")
            return []

        pairs = []
        for camera_dir in sorted(self.test_images_dir.iterdir()):
            if not camera_dir.is_dir():
                continue
            mask_camera_dir = self.test_masks_dir / camera_dir.name
            for image_path in sorted(camera_dir.iterdir()):
                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                mask_path = self._find_mask(mask_camera_dir, image_path.stem)
                if mask_path is None:
                    print(
                        f"  WARNING: no test mask for '{camera_dir.name}/{image_path.name}' — skipping."
                    )
                    continue
                pairs.append((image_path, mask_path))
        return pairs

    def _find_mask(self, masks_dir: Path, stem: str) -> Path | None:
        for ext in VALID_EXTENSIONS:
            candidate = masks_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _load_val_filenames(self) -> set[str]:
        if not self.validation_csv.exists():
            print(f"  WARNING: {self.validation_csv} not found — all data routed to train.")
            return set()
        with open(self.validation_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {
                row["fileNames"].strip().strip(",")
                for row in reader
                if row["fileNames"].strip()
            }

    def split(self, pairs: list[Pair]) -> dict[str, list[Pair]]:
        val_filenames = self._load_val_filenames()
        train, val = [], []
        for image_path, mask_path in pairs:
            bucket = val if image_path.stem in val_filenames else train
            bucket.append((image_path, mask_path))

        splits = {TRAIN_NAME: train, VALID_NAME: val}
        if self._test_pairs:
            splits[TEST_NAME] = self._test_pairs
        return splits
