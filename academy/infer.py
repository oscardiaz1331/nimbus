"""Test-set evaluation: pixel metrics + overlays (segmentation) or
classification metrics, plus an FPS benchmark — all driven by the same
``config.yaml`` used for training so preprocessing never drifts between
the two scripts.

Usage:
    python infer.py --config config.yaml --checkpoint runs/.../best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.benchmark.profiler import ProfileResult, profile
from utils.commons import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    load_class_names,
    rasterize_polygon_mask,
    save_metrics_table,
    save_overlay,
)
from utils.config import Config
from utils.models import get_adapter


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )


def run_segmentation_eval(cfg: Config, adapter, out_dir: Path) -> pd.DataFrame:
    class_names = load_class_names(cfg.dataset.yaml_path)
    num_classes = len(class_names)
    images_dir = cfg.dataset.images_dir(cfg.dataset.test_split)
    labels_dir = cfg.dataset.labels_dir(cfg.dataset.test_split)
    overlays_dir = out_dir / "overlays"

    image_paths = _list_images(images_dir)
    print(f"  Evaluating {len(image_paths)} test images ({cfg.framework}/segmentation)")

    rows = []
    for i, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        gt = rasterize_polygon_mask(
            labels_dir / f"{img_path.stem}.txt", h, w, num_classes
        )
        pred = adapter.predict_segmentation_masks(image, num_classes)

        for cls_id, cls_name in class_names.items():
            m = compute_segmentation_metrics(gt[cls_id], pred[cls_id])
            m.update(
                {"image": img_path.name, "class_id": cls_id, "class_name": cls_name}
            )
            rows.append(m)

        if cfg.inference.max_overlay_images < 0 or i < cfg.inference.max_overlay_images:
            save_overlay(
                image,
                gt.any(axis=0).astype(np.uint8),
                pred.any(axis=0).astype(np.uint8),
                overlays_dir / f"{img_path.stem}.png",
                title=img_path.name,
            )

    df = pd.DataFrame(rows)
    summary = save_metrics_table(
        df, out_dir, "class_name", ["iou", "dice", "precision", "recall", "accuracy"]
    )
    print(f"\n{summary}\n")
    return summary


def run_classification_eval(cfg: Config, adapter, out_dir: Path) -> None:
    class_names = load_class_names(cfg.dataset.yaml_path)
    test_root = cfg.dataset.images_dir(cfg.dataset.test_split)
    name_to_id = {v: k for k, v in class_names.items()}

    y_true, y_pred, rows = [], [], []
    for cls_dir in (
        sorted(p for p in test_root.iterdir() if p.is_dir())
        if test_root.exists()
        else []
    ):
        gt_id = name_to_id.get(cls_dir.name)
        if gt_id is None:
            continue
        for img_path in _list_images(cls_dir):
            image = cv2.imread(str(img_path))
            pred_id = adapter.predict_classification_label(image)
            y_true.append(gt_id)
            y_pred.append(pred_id)
            rows.append(
                {
                    "image": img_path.name,
                    "gt": class_names[gt_id],
                    "pred": class_names.get(pred_id, "?"),
                }
            )

    metrics = compute_classification_metrics(
        np.array(y_true), np.array(y_pred), len(class_names)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "per_image_predictions.csv", index=False)
    print(
        f"\n  Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
    )


def benchmark_fps(
    cfg: Config, adapter, sample_image: np.ndarray
) -> tuple[ProfileResult, float | None]:
    """FPS/latency/RAM via ``profile()`` plus peak VRAM across the whole run.

    VRAM is tracked here via ``torch.cuda.reset_peak_memory_stats`` — the
    pytorch path runs inside torch's own allocator, so its peak stats are both
    exact and attributable, and the peak (not a steady-state reading) is the
    number that actually determines whether a given card OOMs. ``profile()``'s
    ``vram_mb`` stays None on this path: its NVML device-delta measurement
    exists for onnxruntime sessions, which torch's allocator can't see
    (see ``utils.benchmark.profiler.device_vram_used_mb``).
    """
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        torch, has_cuda = None, False
    if has_cuda:
        torch.cuda.reset_peak_memory_stats()

    result = profile(
        lambda: adapter.predict(sample_image),
        warmup=cfg.inference.benchmark_warmup,
        iters=cfg.inference.benchmark_iters,
    )
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if has_cuda else None

    if peak_vram_mb is not None:
        vram_str = f"peak {peak_vram_mb:.0f} MB"
    elif result.vram_mb is not None:
        vram_str = f"{result.vram_mb:.0f} MB"
    else:
        vram_str = "n/a"
    print(
        f"\n  Benchmark: {result.fps:.1f} FPS  {result.mean_latency_ms:.1f} ms  "
        f"RAM={result.ram_mb:.0f} MB  VRAM={vram_str}  "
        f"(warmup={cfg.inference.benchmark_warmup}, iters={cfg.inference.benchmark_iters})"
    )
    return result, peak_vram_mb


def main(config_path: str, checkpoint: str | None) -> None:
    cfg = Config.from_yaml(config_path)
    if checkpoint:
        cfg.model.checkpoint = checkpoint

    adapter = get_adapter(cfg)
    out_dir = Path(cfg.output_dir) / cfg.framework / cfg.variant / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.task == "segmentation":
        run_segmentation_eval(cfg, adapter, out_dir)
    else:
        run_classification_eval(cfg, adapter, out_dir)

    sample_images = _list_images(cfg.dataset.images_dir(cfg.dataset.test_split))
    if sample_images:
        sample = cv2.imread(str(sample_images[0]))
        benchmark_fps(cfg, adapter, sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-set evaluation + FPS benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint", default=None, help="Override model.checkpoint from config.yaml"
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint)
