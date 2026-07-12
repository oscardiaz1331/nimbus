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

from utils.datasets.base import SPLIT_DICT
from utils.datasets.config import DatasetBuilderConfig


def merge_datasets(
    member_configs: list[DatasetBuilderConfig], config: DatasetBuilderConfig
) -> dict[str, int]:
    from utils.datasets import get_builder  # deferred: avoid import cycle

    output_dir = Path(config.output_dir)
    dirs = {}
    for split_name, split_folder in SPLIT_DICT.items():
        img_dir = output_dir / split_folder / "images"
        lbl_dir = output_dir / split_folder / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dirs[split_name] = (img_dir, lbl_dir)

    stats = {split: 0 for split in SPLIT_DICT.keys()}
    for member_config in member_configs:
        builder = get_builder(member_config)
        print(f"[{config.name}] member '{member_config.name}' -> merged output directly")

        splits = builder.collect_splits()
        member_stats, _ = builder.process_splits(splits, dirs)
        for split_name, count in member_stats.items():
            stats[split_name] += count

    used_splits = {split_name: split_folder for split_name, split_folder in SPLIT_DICT.items() if stats[split_name] > 0}
    yaml_dict = {
        "path": str(output_dir.resolve()),
        **{split_name: f"{split_folder}/images" for split_name, split_folder in used_splits.items()},
        "nc": 1,
        "names": {config.class_id: config.class_name},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"\n[{config.name}] merge complete:")
    for split in SPLIT_DICT.keys():
        print(f"  {split:5s}: {stats[split]} samples (merged)")
    print(f"  data.yaml -> {yaml_path}")
    return stats
