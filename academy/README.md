# Nimbus

A config-driven training/inference pipeline for cloud segmentation, built
on top of YOLO11/26-seg and RF-DETR-seg, with a custom 3-stage
progressive fine-tuning strategy and dual-task (segmentation /
classification) support.

## Architecture

```
nimbus/academy/
├── config.yaml              # single source of truth for train.py and infer.py
├── train.py                 # 3-stage training entry point
├── infer.py                 # test-set evaluation + FPS benchmark
├── utils/
│   ├── config.py             # typed Config dataclasses + YAML loader/validator
│   ├── commons.py            # dataset I/O, segmentation/classification metrics,
│   │                         # overlay rendering, FPS benchmarking
│   ├── plotter.py            # one push-based TrainingPlotter shared by every backend
│   ├── stages.py             # framework-agnostic plateau-detection / stage-advance logic
│   └── models/
│       ├── base.py           # ModelAdapter interface (5 methods)
│       ├── yolo_adapter.py   # Ultralytics backend (segmentation + classification)
│       └── rfdetr_adapter.py # RF-DETR backend (segmentation only)
└── tests/                    # unittest suite for everything that doesn't need a GPU/dataset
```

`train.py` and `infer.py` never import `ultralytics` or `rfdetr`
directly — they only talk to the `ModelAdapter` interface, so swapping
or adding a backend never touches the entry points.

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
```

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
FPS benchmark call counting, and the plateau-detection/stage-advance
state machine. These don't require a GPU, a
trained checkpoint, or the actual cloud dataset — they test the logic
that's independent of `ultralytics`/`rfdetr` internals.

What's **not** covered by an automated test, because it genuinely needs
a GPU + real checkpoints + real data to exercise: the live training
loop inside each adapter's `run_stage`, and the inference paths in
`infer.py`. Those were ported faithfully from this project's original
working training/inference scripts and are flagged with "ponytail"
comments wherever the implementation depends on framework-internal
behavior (e.g. `trainer.stop`) that's worth re-checking against your
installed `ultralytics`/`rfdetr` version before a long run.

## Known limitations

- `FPSBenchmark` measures single-stream, single-process throughput — it is a sanity check, not a production serving benchmark.