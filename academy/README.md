# Nimbus Academy

A config-driven training/inference/optimization pipeline for cloud
segmentation, built on top of YOLO11/26-seg and RF-DETR-seg, with a
custom 3-stage progressive fine-tuning strategy, dual-task
(segmentation / classification) support, and an ONNX export →
simplify → fp16/int8 → benchmark → report optimization pipeline.

This is the training/optimization component of Nimbus; the resulting model is deployed in the Nimbus Observatory.

## Architecture

```
nimbus/academy/
├── config.yaml              # single source of truth for train.py, infer.py and optimize.py
├── train.py                 # 3-stage training entry point
├── infer.py                 # test-set evaluation + FPS/VRAM benchmark
├── optimize.py               # ONNX export/simplify/fp16/int8/prune/benchmark/report CLI
├── benchmark_checkpoints.py  # runs infer.py's eval+benchmark across a list of checkpoints, writes SUMMARY.md
├── benchmark_trials.yaml     # the checkpoints benchmark_checkpoints.py compares (framework/checkpoint/variant per row)
├── datasets/
│   ├── datasets.md           # per-source dataset docs: authors, DOI/citation, labels, file layout
│   ├── build_dataset.py      # single entry point for every dataset builder (config-driven)
│   └── configs/              # one YAML per dataset source, plus merge configs
│       ├── kontas2017.yaml
│       ├── swimseg.yaml
│       ├── swinseg.yaml
│       └── merged_cloud.yaml # kind: merge — fuses the sources above into one split dataset
├── utils/
│   ├── config.py             # typed Config dataclasses + YAML loader/validator
│   ├── commons.py            # dataset I/O, segmentation/classification metrics,
│   │                         # overlay rendering, FPS benchmarking
│   ├── datasets/              # dataset-builder implementation behind build_dataset.py
│   │   ├── config.py         # typed DatasetBuilderConfig + YAML loader (mirrors utils/config.py)
│   │   ├── base.py            # shared builder scaffolding (split, write YOLO labels, etc.)
│   │   ├── binary_mask_builder.py # builder for single binary sky/cloud mask sources (SWIMSEG/SWINSEG)
│   │   ├── kontas_builder.py  # builder for the multi-class Kontas-2017 masks
│   │   └── merge.py           # combines same-named splits across already-built member datasets
│   ├── plotter.py            # one push-based TrainingPlotter shared by every backend
│   ├── stages.py             # framework-agnostic plateau-detection / stage-advance logic
│   ├── paths.py              # canonical artifact paths for one run's output_dir
│   ├── session.py            # Session: paths.py resolved once per optimize.py run
│   ├── onnx_utils.py         # small ONNX graph introspection helpers (e.g. input size)
│   ├── models/
│   │   ├── base.py           # ModelAdapter interface (export_onnx, prune, + training/infer methods)
│   │   ├── yolo_adapter.py   # Ultralytics backend (segmentation + classification)
│   │   └── rfdetr_adapter.py # RF-DETR backend (segmentation only)
│   ├── optimizers/           # ONNX-graph transforms, no PyTorch dependency (except pruner.py)
│   │   ├── onnx_simplifier.py
│   │   ├── fp16_converter.py
│   │   ├── int8_converter.py # calibrates against real training images
│   │   ├── pruner.py         # magnitude pruning on the checkpoint, pre-export
│   │   └── metadata.py
│   ├── benchmark/            # variant x execution-provider profiling + accuracy
│   │   ├── benchmark.py      # BenchmarkRow + benchmark_variant()
│   │   ├── profiler.py       # latency/FPS/RAM/VRAM measurement (also used by infer.py directly, for raw-framework checkpoints)
│   │   ├── evaluator.py      # accuracy evaluation over a fixed sample set
│   │   ├── onnx_decode.py    # onnxruntime output -> predicted mask per framework
│   │   └── trials.py         # typed loader for benchmark_trials.yaml
│   └── reports/              # optimize.py `report` command output
│       ├── markdown_report.py
│       ├── html_report.py
│       └── plots.py
└── tests/                    # unittest suite for everything that doesn't need a GPU/dataset
```

`train.py` and `infer.py` never import `ultralytics` or `rfdetr`
directly — they only talk to the `ModelAdapter` interface, so swapping
or adding a backend never touches the entry points. `optimize.py`
follows the same rule from `simplify` onward: only `export` and `prune`
touch the adapter/PyTorch, everything past ONNX export is
framework-agnostic.

## The 3-stage training strategy

| Stage | What's trainable | Exit condition |
|---|---|---|
| 1. `warmup` | Head + last few layers only (`freeze: backbone`) | Validation plateaus for `patience` epochs **or** `max_epochs` reached |
| 2. `intermediate` | Head + a tail fraction of the backbone (`freeze: partial`, `unfreeze_fraction`) | Same plateau rule, independent `patience` |
| 3. `finetune` | Entire model (`freeze: none`) | The backend's **native** early stopping (`early_stopping_patience`), with a cosine/plateau LR schedule |

Stages 1 and 2 are intentionally allowed to end early: there's no value
in training a frozen head for 50 epochs if it converges in 12. The
plateau detector (`utils/stages.py::PlateauDetector`) is pure Python —
no framework dependency — so it's the same logic for both backends and
is fully unit-tested. Each adapter wires it into the framework's own
early-stop hook (`trainer.stop = True` for Ultralytics,
`trainer.should_stop = True` for PyTorch Lightning/RF-DETR) rather than
re-implementing training-loop control flow.

Stage 3 deliberately does **not** use the custom plateau detector: both
backends already ship a correct, battle-tested early-stopping
implementation for the "drop everything if we've truly converged" case,
so reusing it is both less code and less risk than reinventing it.

## Dual-task support

- **YOLO**: task is auto-detected from the checkpoint filename
  (`*-seg.pt` → segmentation, `*-cls.pt` → classification, otherwise
  detection) via `utils/models/yolo_adapter.py::detect_yolo_task`.
- **RF-DETR**: toggled explicitly via `model.rfdetr.segmentation` in
  `config.yaml`, per the spec. RF-DETR is a detection/segmentation
  library only — `Config` rejects `framework: rfdetr` combined with
  `task: classification` at load time, and the adapter raises
  `NotImplementedError` if `segmentation: false` is ever set, rather
  than silently doing the wrong thing.

## Datasets

The raw datasets themselves are **not** distributed with this repo — you
need to fetch and unzip each source yourself. Everything downstream of
that (Kontas/SWIMSEG/SWINSEG grayscale or binary masks → YOLO-format
segmentation labels, in a single per-source split or merged across
sources) is config-driven through one entry point:

```bash
# Build a single source
python datasets/build_dataset.py --config datasets/configs/kontas2017.yaml
python datasets/build_dataset.py --config datasets/configs/swimseg.yaml
python datasets/build_dataset.py --config datasets/configs/swinseg.yaml

# Or build the merged dataset (kind: merge) — builds each member source's
# own train/val/test split first, then combines same-named splits so one
# training batch mixes images from all of them
python datasets/build_dataset.py --config datasets/configs/merged_cloud.yaml
```

Each dataset (or merge of datasets) is described by its own YAML under
`datasets/configs/`, loaded into a typed `DatasetBuilderConfig`
(`utils/datasets/config.py`) the same way `config.yaml` drives
`train.py`/`infer.py`/`optimize.py`. `kind: kontas` and `kind:
binary_mask` each point at wherever you've unzipped that source's
images/masks (`root`, `images_subdir`, `masks_subdir`, ...) and are
handled by their own builder (`kontas_builder.py`,
`binary_mask_builder.py`); `kind: merge` instead lists `members:` — the
YAML files of already-defined sources to fuse into one `output_dir`.
Switching between a single-source and a merged dataset, or pointing at
a different unzip location, is a config-only change — no code edits.

Where to get each source, its authors/citation, class labels, and exact
on-disk file layout is documented per-dataset in
[`datasets/datasets.md`](datasets/datasets.md) — read that before
downloading anything.

## Usage

```bash
# Recommended — uv is significantly faster than pip
uv venv && uv pip install -r requirements.txt --index-strategy unsafe-best-match

# Alternative
pip install -r requirements.txt

# Train: runs all 3 stages back to back, writing checkpoints/plots under
# training.output_dir as configured in config.yaml.
python train.py --config config.yaml

# Evaluate the best checkpoint on the test split + benchmark FPS/VRAM.
python infer.py --config config.yaml --checkpoint runs/cloudvision/finetune/weights/best.pt

# Compare a whole list of checkpoints (benchmark_trials.yaml) the same way —
# each row drives infer.py's own eval+benchmark path, then SUMMARY.md is
# regenerated from the results (see the "Benchmarking checkpoints" section below).
python benchmark_checkpoints.py --config config.yaml --trials benchmark_trials.yaml

# Export the trained checkpoint to ONNX, then simplify/quantize/benchmark it
# and write a report. `pipeline` (the default) runs export -> simplify ->
# fp16 -> int8 -> report; `prune` is opt-in and not part of `pipeline`.
python optimize.py --config config.yaml --command pipeline
python optimize.py --config config.yaml --command prune --prune-amount 0.3
```

Artifact locations for a run are derived once from `config.yaml`
(`utils/paths.py`, resolved into a `Session` in `utils/session.py`) under
`<output_dir>/<framework>/<variant>/optimize/`: `model.onnx`,
`model_fp16.onnx`, `model_int8.onnx`, and an `optimization_report/` with
`report.md`, `report.html`, and `plots/`. The `benchmark`/`report`
commands skip any variant that hasn't been generated yet, benchmark on
CPU always and on GPU when onnxruntime reports a working CUDA provider,
and — for segmentation — compare accuracy across variants on the same
fixed test-split samples.

Switching task or backend is a config-only change:

```yaml
task: "classification"
framework: "yolo"
model:
  yolo:
    variant: "yolo11m-cls.pt"
```

## Benchmarking checkpoints

`SUMMARY.md` at the repo root is generated by `benchmark_checkpoints.py`, not
hand-written. It compares a list of trained checkpoints — one entry per row
in `benchmark_trials.yaml` (`name`, `framework`, `checkpoint`, `variant`) —
by cloning the base `--config` (so dataset/imgsz/conf-threshold/benchmark
warmup-iters stay identical across rows) and overriding only
`framework`/`model.checkpoint`/`model.<framework>.variant` per trial, then
running that exact config through `infer.py`'s own
`run_segmentation_eval`/`benchmark_fps` — the same accuracy metrics and FPS
benchmark a single `python infer.py --checkpoint ...` run would produce.

This is deliberately **not** the `optimize.py` ONNX pipeline: it drives each
backend's own library (`ultralytics` for YOLO, `rfdetr` for RF-DETR) through
the `ModelAdapter` interface, not `onnxruntime`, and none of the checkpoints
it benchmarks have been exported or optimized (no fp16/int8/pruning) — it's
the raw PyTorch checkpoint straight off training. Use it to compare
finetuning runs against each other; use `optimize.py`'s `report` command to
see what exporting/quantizing costs *within* one checkpoint.

VRAM is peak `torch.cuda.max_memory_allocated()` during the FPS benchmark
(`infer.py::benchmark_fps`), reset per trial — the number that determines
whether a given card OOMs, not just steady-state usage after warmup.

```bash
python benchmark_checkpoints.py --config config.yaml --trials benchmark_trials.yaml
# writes SUMMARY.md by default; override with --output
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover the metric math (segmentation IoU/Dice/Precision/Recall,
classification accuracy/precision/recall/F1), polygon rasterization,
FPS benchmark call counting, the plateau-detection/stage-advance state
machine, and the `optimize.py` pipeline: the optimizer transforms
(`test_optimizers.py`), the `optimize.py` command wiring
(`test_optimize.py`), the accuracy evaluator
(`test_evaluator.py`), and the `benchmark_trials.yaml` loader
(`test_trials.py`). These don't require a GPU, a trained checkpoint,
or the actual cloud dataset — they test the logic that's independent of
`ultralytics`/`rfdetr`/`onnxruntime` internals.

What's **not** covered by an automated test, because it genuinely needs
a GPU + real checkpoints + real data to exercise: the live training
loop inside each adapter's `run_stage`, the inference paths in
`infer.py`, and the actual ONNX export/simplify/fp16/int8 conversions
and benchmarking in `optimize.py`. Those were ported faithfully from
this project's original working training/inference scripts and are
flagged with "ponytail" comments wherever the implementation depends on
framework-internal behavior (e.g. `trainer.stop`) that's worth
re-checking against your installed `ultralytics`/`rfdetr`/`onnxruntime`
version before a long run.

## Known limitations

- `FPSBenchmark` measures single-stream, single-process throughput — it is a sanity check, not a production serving benchmark.