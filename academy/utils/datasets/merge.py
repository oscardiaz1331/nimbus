"""Fuses several datasets into one combined train/val/test set, so a single
training batch can mix images from different sources (e.g. SWIMSEG + the
more heterogeneous Almeria/Kontas set), instead of training on just one at
a time.

Each member is only ever converted+copied straight into the *merged*
split directories — its own standalone dataset (what running its config
directly would produce) is never written, so a merge build doesn't cost
double the disk. Every builder already name-prefixes its files with its
own dataset name (see DatasetBuilder.process_sample), so merged filenames
never collide.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from utils.datasets.base import SPLIT_NAMES
from utils.datasets.config import DatasetBuilderConfig


def merge_datasets(
    member_configs: list[DatasetBuilderConfig], config: DatasetBuilderConfig
) -> dict[str, int]:
    from utils.datasets import get_builder  # deferred: avoid import cycle

    output_dir = Path(config.output_dir)
    dirs = {}
    for split in SPLIT_NAMES:
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dirs[split] = (img_dir, lbl_dir)

    stats = {split: 0 for split in SPLIT_NAMES}
    for member_config in member_configs:
        builder = get_builder(member_config)
        print(f"[{config.name}] member '{member_config.name}' -> merged output directly")

        splits = builder.collect_splits()
        member_stats, _ = builder.process_splits(splits, dirs)
        for split_name, count in member_stats.items():
            stats[split_name] += count

    used_splits = [split for split in SPLIT_NAMES if stats[split] > 0]
    yaml_dict = {
        "path": str(output_dir.resolve()),
        **{split: f"{split}/images" for split in used_splits},
        "nc": 1,
        "names": {config.class_id: config.class_name},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"\n[{config.name}] merge complete:")
    for split in SPLIT_NAMES:
        print(f"  {split:5s}: {stats[split]} samples (merged)")
    print(f"  data.yaml -> {yaml_path}")
    return stats
