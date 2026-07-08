#!/usr/bin/env python3
"""CLI entrypoint for the Optimize pipeline.

Usage:
    python optimize.py <command> [--config config.yaml]

Commands: simplify, fp16, int8, prune, benchmark, report, pipeline.

Getting a checkpoint into model.onnx now happens inside training itself
(one line added to the existing YOLO/RF-DETR training classes) rather than
through a dedicated exporter here, so every command below assumes
model.onnx already exists and works with it as a plain ONNX graph.
`simplify` onward never touches PyTorch/ultralytics/rfdetr — only `prune`
does, since pruning has to happen on the checkpoint before it's exported.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from utils.models import get_adapter
from utils.config import Config
from utils.paths import resolve_checkpoint, weights_dir
from utils.session import Session


from utils.benchmark.benchmark import BenchmarkRow

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def _list_images(directory: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in _IMAGE_EXTS:
        found.extend(directory.glob(pattern))
    return sorted(found)


def _require(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path


def _checkpoint_candidate_names(cfg: Config) -> tuple[str, ...]:
    return ("best.pt", "last.pt") if cfg.framework == "yolo" else ("best.pth", "last.pth")


def cmd_simplify(cfg: Config, session: Session) -> None:
    from utils.optimizers import onnx_simplifier

    _require(session.onnx, "Export the checkpoint to ONNX first, Run `export`.")
    onnx_simplifier.simplify(session.onnx)
    print(f"simplified: {session.onnx}")


def cmd_fp16(cfg: Config, session: Session) -> None:
    from utils.optimizers import fp16_converter

    _require(session.onnx, "Run `simplify` (or export) first.")
    fp16_converter.convert_fp16(session.onnx, session.fp16)
    print(f"fp16: {session.fp16}")


def cmd_int8(cfg: Config, session: Session) -> None:
    from utils.optimizers import int8_converter

    _require(session.onnx, "Run `simplify` (or export) first.")
    calibration_images = _list_images(cfg.dataset.images_dir("train"))
    print(f"Found {len(calibration_images)} calibration images.")
    int8_converter.quantize_int8(
        session.onnx,
        session.int8,
        calibration_images,
        cfg.inference.imgsz,
        normalize_imagenet=(cfg.framework == "rfdetr"),
    )
    print(f"int8: {session.int8}")


def cmd_prune(cfg: Config, session: Session, amount: float = 0.3) -> None:
    adapter = get_adapter(cfg)
    checkpoint = resolve_checkpoint(cfg, _checkpoint_candidate_names(cfg))
    output = checkpoint.parent / f"{checkpoint.stem}-pruned{checkpoint.suffix}"
    adapter.prune(checkpoint, output, amount)
    print(f"pruned checkpoint: {output}")


def _load_eval_samples(cfg: Config) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """(image_bgr, gt_mask) pairs from the test split, shared across every
    variant/provider so accuracy is compared on identical inputs.
    """
    import cv2

    from utils.commons import load_class_names, rasterize_polygon_mask

    num_classes = len(load_class_names(cfg.dataset.yaml_path))
    images_dir = cfg.dataset.images_dir(cfg.dataset.test_split)
    labels_dir = cfg.dataset.labels_dir(cfg.dataset.test_split)
    image_paths = _list_images(images_dir)
    if cfg.inference.max_eval_images >= 0:
        image_paths = image_paths[: cfg.inference.max_eval_images]

    samples = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        gt = rasterize_polygon_mask(labels_dir / f"{img_path.stem}.txt", h, w, num_classes)
        samples.append((image, gt))
    return samples, num_classes


def _run_benchmark(cfg: Config, session: Session) -> list[BenchmarkRow]:
    """One row per (onnx/fp16/int8 variant) x (execution provider), skipping
    variants that haven't been generated yet. CPU is always benchmarked;
    GPU is added automatically when onnxruntime reports a CUDA device
    (needs the ``onnxruntime-gpu`` package, which replaces plain
    ``onnxruntime`` — they can't coexist).
    """
    import onnxruntime as ort

    from utils.benchmark import evaluator, onnx_decode
    from utils.benchmark.benchmark import benchmark_variant
    from utils.onnx_utils import graph_input_size

    device_providers = [("cpu", ["CPUExecutionProvider"])]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        device_providers.append(("gpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]))

    eval_samples, num_classes = (
        _load_eval_samples(cfg) if cfg.task == "segmentation" else ([], 0)
    )

    variants = [("onnx", session.onnx), ("fp16", session.fp16), ("int8", session.int8)]
    rows = []
    for variant_name, path in variants:
        if not path.is_file():
            continue
        # Read the resolution off the graph itself rather than cfg.inference.imgsz —
        # RF-DETR bakes in its own checkpoint resolution, independent of that config
        # value, and a mismatch here crashes onnxruntime with the same invalid-shape
        # error int8 calibration used to hit before it did the same fix.
        _, model_imgsz = graph_input_size(path)
        dummy = np.zeros((1, 3, model_imgsz, model_imgsz), dtype=np.float32)

        for device_label, providers in device_providers:
            try:
                sess = ort.InferenceSession(str(path), providers=providers)
            except Exception as e:
                print(f"skipping {variant_name}-{device_label}: {e}")
                continue
            # onnxruntime doesn't raise when a requested provider fails to load
            # (missing CUDA/cuDNN runtime DLLs, wrong CUDA major version, etc.) —
            # it just logs an error and silently falls back to the next one in
            # the list. Without this check a "gpu" row would silently be a CPU
            # run wearing a GPU label, which is worse than not having the row.
            active = sess.get_providers()
            if providers[0] not in active:
                print(f"skipping {variant_name}-{device_label}: {providers[0]} did not load, got {active}")
                continue
            input_name = sess.get_inputs()[0].name
            print(f"{variant_name}-{device_label}: providers={active}")

            def run_once(sess=sess, input_name=input_name, dummy=dummy) -> None:
                sess.run(None, {input_name: dummy})

            if eval_samples:
                predict_mask = onnx_decode.make_predict_mask(
                    cfg.framework, sess, model_imgsz, cfg.inference.conf_threshold, num_classes
                )
                evaluate = lambda predict_mask=predict_mask: evaluator.evaluate(eval_samples, predict_mask)
            else:
                evaluate = lambda: {}

            rows.append(
                benchmark_variant(
                    name=f"{variant_name}-{device_label}",
                    model_path=path,
                    run_once=run_once,
                    evaluate=evaluate,
                    warmup=cfg.inference.benchmark_warmup,
                    iters=cfg.inference.benchmark_iters,
                )
            )
    if not rows:
        raise FileNotFoundError("no variants found to benchmark — run simplify/fp16/int8 first")
    return rows


def cmd_benchmark(cfg: Config, session: Session) -> None:
    rows = _run_benchmark(cfg, session)
    for r in rows:
        print(f"{r.name:8s} {r.fps:8.1f} fps  {r.latency_ms:8.1f} ms  {r.size_mb:8.1f} MB")


def cmd_report(cfg: Config, session: Session) -> None:
    from utils.reports import html_report, markdown_report, plots

    rows = _run_benchmark(cfg, session)
    accuracy_keys = sorted({k for r in rows for k in r.accuracy})
    try:
        checkpoint = str(resolve_checkpoint(cfg, _checkpoint_candidate_names(cfg)))
    except FileNotFoundError:
        checkpoint = cfg.model.checkpoint or "(unresolved)"

    markdown_report.write(
        rows,
        session.report_dir / "report.md",
        framework=cfg.framework,
        checkpoint=checkpoint,
        date=datetime.date.today().isoformat(),
        accuracy_keys=accuracy_keys,
    )
    plot_paths = plots.generate_all(rows, session.plots_dir, accuracy_keys=accuracy_keys)
    html_report.write(
        rows,
        session.report_dir / "report.html",
        plot_filenames=[p.name for p in plot_paths.values()],
        accuracy_keys=accuracy_keys,
    )
    print(f"report: {session.report_dir}")

def cmd_export(cfg: Config, session: Session) -> None:
    """Export the checkpoint to ONNX with post-processing wrapper."""
    adapter = get_adapter(cfg)
    adapter.export_onnx(session.onnx)
    
def cmd_pipeline(cfg: Config, session: Session) -> None:
    """export -> simplify -> fp16 -> int8 -> report. Matches the design's flowchart; prune is opt-in, not automatic."""
    cmd_export(cfg, session)
    cmd_simplify(cfg, session)
    cmd_fp16(cfg, session)
    cmd_int8(cfg, session)
    cmd_report(cfg, session)


_COMMANDS = {
    "export": cmd_export,
    "simplify": cmd_simplify,
    "fp16": cmd_fp16,
    "int8": cmd_int8,
    "prune": cmd_prune,
    "benchmark": cmd_benchmark,
    "report": cmd_report,
    "pipeline": cmd_pipeline,
}

import faulthandler
import sys

faulthandler.enable(all_threads=True, file=sys.stderr)
def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize pipeline CLI")
    parser.add_argument("--command", choices=sorted(_COMMANDS), default="pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--prune-amount", type=float, default=0.3, help="Fraction of weights to zero out (prune command only)."
    )
    args = parser.parse_args()
    try:
        cfg = Config.from_yaml(args.config)
        session = Session.from_config(cfg)
        if args.command == "prune":
            cmd_prune(cfg, session, amount=args.prune_amount)
        else:
            _COMMANDS[args.command](cfg, session)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()