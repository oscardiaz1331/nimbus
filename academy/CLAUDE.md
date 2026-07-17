# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Nimbus Academy: a config-driven training / inference / ONNX-optimization pipeline for cloud segmentation (YOLO11/26-seg and RF-DETR-seg backends). It is the training half of Nimbus; the trained model is deployed in the separate Nimbus Observatory. See `README.md` for the full architecture write-up and `datasets/datasets.md` for per-dataset sources and layouts.

## Commands

```bash
# Install (uv is much faster than pip)
uv venv && uv pip install -r requirements.txt --index-strategy unsafe-best-match

# Full test suite (CPU-only, no dataset or checkpoint needed)
python -m unittest discover -s tests -v

# Single test module / case
python -m unittest tests.test_stages -v
python -m unittest tests.test_optimizers.TestFp16Converter.test_convert -v

# Train (all configured stages), evaluate, optimize — all driven by config.yaml
python train.py --config config.yaml
python infer.py --config config.yaml --checkpoint runs/<...>/weights/best.pt
python optimize.py --config config.yaml --command pipeline   # export→simplify→fp16→int8→report
python optimize.py --config config.yaml --command prune --prune-amount 0.3  # opt-in, not in pipeline

# Benchmark a list of checkpoints (benchmark_trials.yaml): one `pytorch` row per
# checkpoint (infer.py's own eval+FPS/VRAM path, raw framework libraries) plus, unless
# --skip-onnx, one row per existing onnx/fp16/int8 x cpu/gpu variant (optimize.py's own
# benchmark path). --pipeline (re)generates those ONNX variants first. Regenerates
# SUMMARY.md with a combined table and a final deployment recommendation.
python benchmark_checkpoints.py --config config.yaml --trials benchmark_trials.yaml --pipeline

# Build datasets (config-driven; raw sources must be downloaded manually first)
python datasets/build_dataset.py --config datasets/configs/merged_cloud.yaml
```

There is no CI and no lint config; `black` and `pytest` are installed but unconfigured. Tests are plain `unittest`.

## Architecture — the rules that span files

- **`config.yaml` is the single source of truth** for `train.py`, `infer.py`, and `optimize.py`, loaded into typed dataclasses by `utils/config.py`, which validates at load time (e.g. it rejects `framework: rfdetr` + `task: classification`). Dataset builders have a parallel system: `utils/datasets/config.py` + one YAML per source under `datasets/configs/`.
- **Adapter boundary:** `train.py` / `infer.py` never import `ultralytics` or `rfdetr` — they only talk to the `ModelAdapter` interface (`utils/models/base.py`). Adding or changing a backend must not touch the entry points. `optimize.py` follows the same rule from `simplify` onward: only `export` and `prune` touch the adapter/PyTorch; everything downstream of ONNX export is framework-agnostic (`utils/optimizers/` has no PyTorch dependency except `pruner.py`).
- **3-stage training:** stages 1–2 (`warmup`, `intermediate`) use the pure-Python `PlateauDetector` in `utils/stages.py` for early exit, wired into each framework's own stop hook (`trainer.stop` for Ultralytics, `trainer.should_stop` for Lightning/RF-DETR). Stage 3 (`finetune`) deliberately uses the backend's native early stopping instead — do not add plateau logic to it.
- **Artifact paths** are derived once per run: `utils/paths.py` → resolved into a `Session` (`utils/session.py`). Don't hand-build output paths.

## Conventions and gotchas

- **"Ponytail" comments** mark places where the code depends on framework-internal behavior (e.g. `trainer.stop`) that should be re-verified against the installed `ultralytics`/`rfdetr`/`onnxruntime` version before a long run. Preserve them, and add one when introducing a new framework-internal dependency.
- **`config.yaml` is usually mid-experiment.** Commented-out stages, alternate checkpoint paths, etc. are deliberate experiment state — don't "clean them up" or re-enable them unless asked.
- **Tests cover only framework-independent logic** (metrics, rasterization, plateau state machine, optimizer transforms, int8's postprocessing-tail exclusion, command wiring, evaluator, RF-DETR ONNX output decoding, the profiler's VRAM-delta arithmetic, the `benchmark_trials.yaml` loader, and `benchmark_checkpoints.py`'s row-building/recommendation heuristic). The live training loops, `infer.py`/`optimize.py` paths (and `benchmark_checkpoints.py`, which just drives both per checkpoint), and real ONNX conversions need a GPU + data and are untested — never launch `train.py` to verify a change; run the unittest suite instead.
- **Don't search or scan**: `.venv/` (full CUDA torch stack — broad globs return thousands of site-packages hits), `runs/`, `checkpoints/`, the ~800 MB of `*.pt` weights at the repo root, or local dataset dirs (`datasets/16647156/`, `datasets/swinseg_yolo/`, `datasets/merged_yolo/`).
