"""Generic builder for datasets whose ground truth is a single binary mask
per image — white (255) = cloud, black = sky. Covers SWIMSEG and SWINSEG,
and any future dataset sharing this layout, purely by config.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from utils.datasets.base import TRAIN_NAME, VALID_NAME, TEST_NAME, DatasetBuilder, Pair
from utils.datasets.config import DatasetBuilderConfig


class BinaryMaskBuilder(DatasetBuilder):
    def __init__(self, config: DatasetBuilderConfig):
        super().__init__(config)
        src = config.binary_mask
        self.src_root = Path(src.root)
        self.images_dir = self.src_root / src.images_subdir
        self.masks_dir = self.src_root / src.masks_subdir
        self.mask_suffixes = src.mask_suffixes
        self.image_extensions = src.image_extensions
        self.ratios = src.split_ratios
        self.random_seed = src.random_seed

    def mask_to_binary(self, mask: np.ndarray) -> np.ndarray:
        return (mask == 255).astype(np.uint8) * 255

    def find_pairs(self) -> list[Pair]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"masks dir not found: {self.masks_dir}")

        pairs = []
        for ext in self.image_extensions:
            for image_path in sorted(self.images_dir.glob(f"*{ext}")):
                mask_path = self._find_mask(image_path.stem)
                if mask_path is None:
                    print(f"  WARNING: no mask for '{image_path.name}' — skipping.")
                    continue
                pairs.append((image_path, mask_path))
        return pairs

    def _find_mask(self, stem: str) -> Path | None:
        for suffix in self.mask_suffixes:
            candidate = self.masks_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def split(self, pairs: list[Pair]) -> dict[str, list[Pair]]:
        rng = random.Random(self.random_seed)
        shuffled = list(pairs)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.ratios.train)
        val_end = train_end + int(n * self.ratios.valid)
        return {
            TRAIN_NAME: shuffled[:train_end],
            VALID_NAME: shuffled[train_end:val_end],
            TEST_NAME: shuffled[val_end:],
        }
