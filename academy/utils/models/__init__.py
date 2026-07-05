"""Backend factory: turns ``config.framework`` into a concrete adapter."""

from __future__ import annotations


def get_adapter(config):
    """Return the ModelAdapter for ``config.framework``.

    Imports are deferred into each branch so that, e.g., installing only
    `ultralytics` (no `rfdetr`) is enough to run a YOLO-only pipeline.
    """
    if config.framework == "yolo":
        from utils.models.yolo_adapter import YoloAdapter

        return YoloAdapter(config)
    if config.framework == "rfdetr":
        from utils.models.rfdetr_adapter import RFDETRAdapter

        return RFDETRAdapter(config)
    raise ValueError(f"Unknown framework: {config.framework}")
