"""Single entry point for every dataset builder — config-driven, no
per-dataset script duplication. Each dataset (or merge of datasets) is
described by its own YAML file under datasets/configs/.

Usage:
    python datasets/build_dataset.py datasets/configs/kontas2017.yaml
    python datasets/build_dataset.py datasets/configs/swimseg.yaml
    python datasets/build_dataset.py datasets/configs/swinseg.yaml
    python datasets/build_dataset.py datasets/configs/merged_cloud.yaml   # kind: merge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.datasets.config import DatasetBuilderConfig, load_member_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="datasets/configs/merged_cloud.yaml", help="Path to a dataset-builder YAML config")
    args = parser.parse_args()

    config = DatasetBuilderConfig.from_yaml(args.config)

    if config.kind == "merge":
        from utils.datasets.merge import merge_datasets

        member_configs = load_member_configs(config)
        print(f"[{config.name}] merging {len(member_configs)} datasets -> {config.output_dir}")
        merge_datasets(member_configs, config)
        return

    from utils.datasets import get_builder

    get_builder(config).build()


if __name__ == "__main__":
    main()
