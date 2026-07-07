"""Dataset-builder factory: turns a DatasetBuilderConfig.kind into a concrete builder."""

from __future__ import annotations

from utils.datasets.config import DatasetBuilderConfig


def get_builder(config: DatasetBuilderConfig):
    """Imports are deferred into each branch, mirroring utils/models/__init__.py."""
    if config.kind == "kontas":
        from utils.datasets.kontas_builder import KontasBuilder

        return KontasBuilder(config)
    if config.kind == "binary_mask":
        from utils.datasets.binary_mask_builder import BinaryMaskBuilder

        return BinaryMaskBuilder(config)
    raise ValueError(f"Unknown dataset kind: '{config.kind}' (kind=merge is handled separately)")
