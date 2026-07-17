"""Benchmarks a list of trained checkpoints — raw and ONNX-optimized — and
writes one combined comparison table plus a deployment recommendation.

Each trial gets two kinds of rows:

- ``pytorch``: straight through the exact same path as ``python infer.py
  --checkpoint ...`` — same accuracy metrics (``infer.py::run_segmentation_eval``),
  same FPS benchmark (``infer.py::benchmark_fps``) — via each backend's own
  library (``ultralytics``/``rfdetr``) through the ``ModelAdapter`` interface.
- ``onnx-cpu``/``onnx-gpu``/``fp16-cpu``/``fp16-gpu``/``int8-cpu``/``int8-gpu``:
  the exact same path as ``python optimize.py --command pipeline``/``benchmark``
  — onnxruntime CPU/CUDA execution providers via ``optimize.py::_run_benchmark``.
  Pass ``--pipeline`` to also (re)generate the ONNX/fp16/int8 artifacts for
  each trial first; without it, only artifacts that already exist are
  benchmarked (nothing crashes if a trial has none yet).

IoU/Dice/Precision/Recall/Accuracy are all on a shared 0-100 scale and
directly comparable across every row — ``utils/benchmark/evaluator.py``'s
``evaluate()`` computes the same five pixel metrics as
``infer.py::run_segmentation_eval``'s pytorch baseline, just against ONNX
Runtime's decoded output instead of the adapter's own ``.predict()``.

The final recommendation always names a deployable ONNX/fp16/int8 variant,
never a ``pytorch`` row: Observatory (the deployment target) only ever loads
the exported ``.onnx`` file, never a PyTorch checkpoint.

Usage:
    python benchmark_checkpoints.py --config config.yaml --trials benchmark_trials.yaml
    python benchmark_checkpoints.py --config config.yaml --trials benchmark_trials.yaml --pipeline
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
from pathlib import Path

import cv2

from infer import _list_images, benchmark_fps, run_segmentation_eval
from optimize import cmd_export, cmd_fp16, cmd_int8, cmd_simplify, _run_benchmark
from utils.benchmark.benchmark import BenchmarkRow
from utils.benchmark.trials import Trial, load_trials
from utils.config import Config
from utils.models import get_adapter
from utils.session import Session

GPU_LABEL = "RTX-3060-Ti"

_STAGE_ORDER = ["pytorch", "onnx-cpu", "onnx-gpu", "fp16-cpu", "fp16-gpu", "int8-cpu", "int8-gpu"]


def _cfg_for_trial(base_cfg: Config, trial: Trial) -> Config:
    model = dataclasses.replace(base_cfg.model, checkpoint=trial.checkpoint)
    if trial.framework == "yolo":
        model = dataclasses.replace(
            model, yolo=dataclasses.replace(model.yolo, variant=trial.variant)
        )
    else:
        model = dataclasses.replace(
            model, rfdetr=dataclasses.replace(model.rfdetr, variant=trial.variant)
        )
    return dataclasses.replace(base_cfg, framework=trial.framework, model=model)


def _mean_across_classes(summary) -> dict[str, float]:
    """Collapse the per-class mean/std table from run_segmentation_eval into
    one mean per metric — fine for the single-class ``cloud`` dataset this
    pipeline targets, and a reasonable unweighted summary if that changes.
    """
    means = summary.xs("mean", axis=1, level=1).mean()
    return {k: float(v) for k, v in means.items()}


def _gc_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_trial(base_cfg: Config, trial: Trial) -> dict:
    """One ``pytorch`` row: raw checkpoint through infer.py's own eval/FPS path."""
    cfg = _cfg_for_trial(base_cfg, trial)
    print(f"\n=== {trial.name} ({trial.framework}/{trial.variant}) — pytorch/{trial.checkpoint} ===")

    adapter = get_adapter(cfg)
    out_dir = Path(cfg.output_dir) / cfg.framework / trial.variant / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_segmentation_eval(cfg, adapter, out_dir)
    metrics = _mean_across_classes(summary)

    sample_images = _list_images(cfg.dataset.images_dir(cfg.dataset.test_split))
    if not sample_images:
        raise FileNotFoundError(f"no test images under {cfg.dataset.images_dir(cfg.dataset.test_split)}")
    sample = cv2.imread(str(sample_images[0]))
    result, peak_vram_mb = benchmark_fps(cfg, adapter, sample)

    checkpoint_path = Path(trial.checkpoint)
    row = {
        "model": trial.name,
        "stage": "pytorch",
        "iou": 100 * metrics["iou"],
        "dice": 100 * metrics["dice"],
        "precision": 100 * metrics["precision"],
        "recall": 100 * metrics["recall"],
        "accuracy": 100 * metrics["accuracy"],
        "fps": result.fps,
        "latency_ms": result.mean_latency_ms,
        "size_mb": checkpoint_path.stat().st_size / (1024 ** 2) if checkpoint_path.is_file() else None,
        # torch's own peak stats; profile()'s NVML delta is for ONNX rows only
        # and stays None on this path (no baseline is passed).
        "vram_mb": peak_vram_mb,
    }

    del adapter
    _gc_cuda()
    return row


def _row_from_benchmark(model: str, r: BenchmarkRow) -> dict:
    return {
        "model": model,
        "stage": r.name,
        "iou": r.accuracy.get("mIoU"),
        "dice": r.accuracy.get("dice"),
        "precision": r.accuracy.get("precision"),
        "recall": r.accuracy.get("recall"),
        "accuracy": r.accuracy.get("accuracy"),
        "fps": r.fps,
        "latency_ms": r.latency_ms,
        "size_mb": r.size_mb,
        "vram_mb": r.vram_mb,
    }


def run_onnx_variants(base_cfg: Config, trial: Trial, run_pipeline: bool) -> list[dict]:
    """One row per (onnx/fp16/int8) x (cpu/gpu) that exists for this trial,
    through the exact same path as ``optimize.py --command benchmark``.

    If ``run_pipeline``, first (re)generates export->simplify->fp16->int8 for
    this trial's checkpoint — same stages as ``optimize.py --command
    pipeline``, minus ``report`` (this script writes its own combined one).
    """
    cfg = _cfg_for_trial(base_cfg, trial)
    session = Session.from_config(cfg)

    if run_pipeline:
        print(f"--- {trial.name}: export -> simplify -> fp16 -> int8 ---")
        cmd_export(cfg, session)
        cmd_simplify(cfg, session)
        cmd_fp16(cfg, session)
        cmd_int8(cfg, session)
        _gc_cuda()

    try:
        bench_rows = _run_benchmark(cfg, session)
    except FileNotFoundError:
        print(f"  (no ONNX artifacts yet for {trial.name} — pass --pipeline to generate them)")
        return []

    return [_row_from_benchmark(trial.name, r) for r in bench_rows]


def recommend(rows: list[dict], max_accuracy_drop_pts: float = 2.0, accuracy_key: str = "iou") -> str:
    """Fastest ONNX-deployable variant within ``max_accuracy_drop_pts`` of the
    best accuracy seen across every row (usually a ``pytorch``/fp32 checkpoint).

    # ponytail: naive heuristic, same spirit as utils/reports/markdown_report.recommend
    # — one hard cutoff on a single accuracy key, no combined score with size/VRAM.
    # ``pytorch`` rows are the accuracy ceiling only, never a candidate: Observatory
    # only ever loads the exported .onnx (see repo README), so the recommendation
    # must always be a deployable onnx/fp16/int8 variant.
    """
    scored = [r for r in rows if r.get(accuracy_key) is not None]
    if not scored:
        return "Not enough data to make a recommendation."
    ceiling = max(r[accuracy_key] for r in scored)
    deployable = [r for r in scored if r["stage"] != "pytorch"]
    if not deployable:
        return "No ONNX variants were benchmarked yet — run with `--pipeline` to generate them."
    candidates = [r for r in deployable if ceiling - r[accuracy_key] <= max_accuracy_drop_pts]
    best = max(candidates or deployable, key=lambda r: r["fps"])
    note = "" if candidates else " — nothing stayed within budget, showing the fastest deployable variant instead"
    return (
        f"✔ **{best['model']} / {best['stage']}** — {best['fps']:.1f} FPS, "
        f"{accuracy_key}={best[accuracy_key]:.1f} (best {accuracy_key} seen: {ceiling:.1f}, "
        f"budget {max_accuracy_drop_pts} pts){note}."
    )


def _fmt(value: float | None, fmt: str = "{:.1f}") -> str:
    return fmt.format(value) if value is not None else "—"


def render_summary_md(rows: list[dict], max_accuracy_drop_pts: float = 2.0) -> str:
    header = [
        "Model", "Stage", "IoU", "Dice", "Precision", "Recall", "Accuracy",
        f"FPS ({GPU_LABEL})", "Latency (ms)", "Size (MB)", "VRAM (MB)",
    ]
    order = {name: i for i, name in enumerate(_STAGE_ORDER)}
    rows_sorted = sorted(rows, key=lambda r: (r["model"], order.get(r["stage"], len(order))))
    lines = [
        "# Summary",
        "",
        f"Finetuned from pretrained default models using a {GPU_LABEL}.",
        "",
        "**Methodology:** generated by `benchmark_checkpoints.py`. `pytorch` rows drive "
        "each checkpoint straight through `infer.py`'s evaluation/benchmark path — each "
        "backend's own library (`ultralytics`/`rfdetr`) through the `ModelAdapter` "
        "interface. `onnx`/`fp16`/`int8` rows (`-cpu`/`-gpu`) drive the exact same path "
        "as `optimize.py --command pipeline`/`benchmark` — onnxruntime CPU/CUDA "
        "execution providers, generated with `--pipeline`. IoU/Dice/Precision/Recall/"
        "Accuracy all share a 0-100 scale and the same pixel-metric formulas across "
        "every row (`utils/benchmark/evaluator.py` mirrors `infer.py`'s pytorch-side "
        "metrics for ONNX rows). VRAM is peak CUDA memory (`pytorch` rows, torch's own "
        "allocator stats) or, for ONNX rows, the NVML device-level VRAM growth from "
        "just before the onnxruntime session is created to a post-warmup sample — "
        "onnxruntime allocates outside torch's allocator, so torch can't measure it; "
        "'—' where no NVIDIA driver is available.",
        "",
        f"Generated: {datetime.date.today().isoformat()}",
        "",
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * len(header))} |",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['model']} | {r['stage']} | {_fmt(r['iou'])} | {_fmt(r['dice'])} | "
            f"{_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['accuracy'])} | "
            f"{r['fps']:.1f} | {_fmt(r['latency_ms'])} | {_fmt(r['size_mb'])} | "
            f"{_fmt(r['vram_mb'], '{:.0f}')} |"
        )
    lines += ["", "---", "", "## Recommendation", "", recommend(rows, max_accuracy_drop_pts)]
    return "\n".join(lines) + "\n"


def main(
    config_path: str,
    trials_path: str,
    output_md: str,
    run_pipeline: bool,
    skip_onnx: bool,
    max_accuracy_drop: float,
) -> None:
    base_cfg = Config.from_yaml(config_path)
    trials = load_trials(trials_path)

    rows: list[dict] = []
    for trial in trials:
        try:
            rows.append(run_trial(base_cfg, trial))
        except Exception as e:
            print(f"!! {trial.name} (pytorch) failed: {e}")

        if skip_onnx:
            continue
        try:
            rows.extend(run_onnx_variants(base_cfg, trial, run_pipeline))
        except Exception as e:
            print(f"!! {trial.name} (onnx) failed: {e}")

    Path(output_md).write_text(render_summary_md(rows, max_accuracy_drop), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows ({len(trials)} trials) to {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark a list of trained checkpoints (raw + ONNX-optimized) and write SUMMARY.md"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--trials", default="benchmark_trials.yaml")
    parser.add_argument("--output", default="SUMMARY.md")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Also (re)generate export->simplify->fp16->int8 per trial before benchmarking ONNX "
        "variants (same stages as `optimize.py --command pipeline`). Without this flag, only "
        "ONNX artifacts that already exist are benchmarked.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Only benchmark the raw pytorch checkpoints — skip ONNX/fp16/int8 rows entirely.",
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=2.0,
        help="Accuracy budget (IoU points, 0-100 scale) for the final recommendation.",
    )
    args = parser.parse_args()
    main(args.config, args.trials, args.output, args.pipeline, args.skip_onnx, args.max_accuracy_drop)
