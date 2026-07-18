# CLAUDE.md — observatory

## What this repo is
Main C++ inference engine for Nimbus. Takes preprocessed frames in, runs AI
inference, hands masks/detections off to the weather intelligence layer.
No training here — that's `academy/`. No camera drivers here — that's `eyes/`.

## Core principle
Business logic never depends on a specific model. Everything goes through
interfaces; swapping models or backends is a config change, not a recompile.

## Interfaces (Strategy Pattern)
- `IInferenceModel` — `load()`, `infer()`, `warmup()`, `metadata()`.
  Implementations: `YOLO`, `RFDETR`, `SAM` (future).
- `IPreprocessor` — resize, normalize, padding, letterbox, color conversion,
  undistortion, ROI extraction. Model-independent.
- `IPostprocessor` — NMS, mask decoding, polygon extraction, cloud/sky %
  stats, connected components, confidence filtering, temporal smoothing.
- `ITracker` — temporal tracking / optical flow (future, not implemented yet).
- `IInferenceBackend` — abstracts ONNX Runtime / TensorRT / future OpenVINO.

# ponytail: `IExporter` shows up in the original spec next to these interfaces,
# but export happens in academy/ against .onnx, not here. Don't add it to this
# repo until there's an actual reason — YAGNI.

## Stack
- C++23, RAII, modern idioms.
- Patterns: Strategy, Factory, Builder, Dependency Injection.
- Eigen, OpenCV, ONNX Runtime (primary — CPU + CUDA EP now, TensorRT EP later).
- CUDA backend when available.

## Module layout
```
camera/          # consumes eyes/ through an interface, not owned here
preprocessing/
inference/
postprocessing/
tracking/
telemetry/
storage/
configuration/
logging/
communication/
```

## Benchmark module
Every model reports automatically: FPS, latency (avg/min/max), GPU memory,
CPU, RAM, warm-up time, accuracy. Model choice is benchmark-driven, not
assumed.

## Config
Nothing hardcoded: camera, model, thresholds, backend, sensor frequency,
LoRa params, storage, logging — all config-driven.

## Where it sits in the pipeline
```
academy  (train → export → ONNX → optimize)
    ↓
observatory  (load → infer → postprocess)   ← this repo
    ↓
weather intelligence / API / dashboard
```
Consumes `.onnx` artifacts produced by `academy/`. Does not train, does not
export.

## Open / not decided
- TensorRT integration timing.
- Whether `ITracker` gets implemented now or stays a stub interface until
  temporal tracking is actually on the roadmap (currently Stage 5, not now).