"""Entry point for the 3-stage progressive training pipeline.

Reads ``config.yaml`` as the single source of truth, builds the right
backend adapter (YOLO or RF-DETR), and runs each configured stage in
order. Stages before the last one can exit early on a validation
plateau; the final stage relies on the backend's own early stopping.

Usage:
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.commons import set_seed
from utils.config import Config
from utils.models import get_adapter
from utils.plotter import TrainingPlotter
from utils.stages import StageController

SEG_PANELS = [
    (
        "Validation loss",
        [
            ("warmup", "warmup/val_loss", "#e74c3c"),
            ("intermediate", "intermediate/val_loss", "#f39c12"),
            ("finetune", "finetune/val_loss", "#3498db"),
        ],
    ),
]
CLS_PANELS = SEG_PANELS  # same single-loss-per-stage shape; kept distinct for clarity at call sites


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.seed)

    adapter = get_adapter(cfg)
    panels = SEG_PANELS if cfg.task == "segmentation" else CLS_PANELS
    print(f"\n  {cfg.framework}/{cfg.task} model: {cfg.variant}")
    plotter = TrainingPlotter(
        out_dir=Path(cfg.output_dir) / cfg.framework / cfg.variant / "plots",
        panels=panels,
        plot_every=cfg.plot_every,
        title=cfg.project_name,
    )

    print(
        f"\n{'=' * 60}\n  {cfg.project_name} — {cfg.framework}/{cfg.task}\n{'=' * 60}"
    )

    epoch_offset = 0
    n_stages = len(cfg.training.stages)
    for i, stage in enumerate(cfg.training.stages):
        is_final = i == n_stages - 1
        controller = (
            None
            if is_final
            else StageController(
                patience=stage.patience, min_delta=cfg.training.plateau_min_delta
            )
        )

        print(
            f"\n  >> Stage {i + 1}/{n_stages}: '{stage.name}' "
            f"(freeze={stage.freeze}, max_epochs={stage.max_epochs})"
        )
        adapter.run_stage(stage, plotter, controller)

        epoch_offset += stage.max_epochs
        plotter.mark_stage_boundary(epoch_offset)

    plotter.finalize()
    print(f"\n  Training complete. Outputs in: {cfg.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-stage progressive training")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
