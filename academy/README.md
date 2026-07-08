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
├── infer.py                 # test-set evaluation + FPS benchmark
├── optimize.py               # ONNX export/simplify/fp16/int8/prune/benchmark/report CLI
├── utils/
│   ├── config.py             # typed Config dataclasses + YAML loader/validator
│   ├── commons.py            # dataset I/O, segmentation/classification metrics,
│   │                         # overlay rendering, FPS benchmarking
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
│   │   ├── profiler.py       # latency/FPS/RAM/VRAM measurement
│   │   ├── evaluator.py      # accuracy evaluation over a fixed sample set
│   │   └── onnx_decode.py    # onnxruntime output -> predicted mask per framework
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

## Usage

```bash
# Recommended — uv is significantly faster than pip
uv venv && uv pip install -r requirements.txt --index-strategy unsafe-best-match

# Alternative
pip install -r requirements.txt

# Train: runs all 3 stages back to back, writing checkpoints/plots under
# training.output_dir as configured in config.yaml.
python train.py --config config.yaml

# Evaluate the best checkpoint on the test split + benchmark FPS.
python infer.py --config config.yaml --checkpoint runs/cloudvision/finetune/weights/best.pt

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

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover the metric math (segmentation IoU/Dice/Precision/Recall,
classification accuracy/precision/recall/F1), polygon rasterization,
FPS benchmark call counting, the plateau-detection/stage-advance state
machine, and the `optimize.py` pipeline: the optimizer transforms
(`test_optimizers.py`), the `optimize.py` command wiring
(`test_optimize.py`), and the accuracy evaluator
(`test_evaluator.py`). These don't require a GPU, a trained checkpoint,
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