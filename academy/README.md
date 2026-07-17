# Nimbus

A DIY end-to-end cloud-observation system: an all-sky camera feeds ML cloud
segmentation on a Raspberry Pi 5, LoRa sensor nodes report environmental
telemetry, and a self-hosted dashboard serves it all. No cloud services —
everything runs on hardware you own. Training happens on a dev PC with a
GPU; the Pi only ever sees the final, optimized `.onnx` model.

## How it fits together

```
academy      (Python)  train → export → ONNX → optimize
    ↓  .onnx artifacts
observatory  (C++23)   camera → preprocess → infer → postprocess → observations
    ↓  POST /api/v1/observations   (HTTP/JSON; LoRa nodes POST telemetry too)
web          (Python)  FastAPI + SQLite store → REST API → Svelte dashboard
    ↓
browsers / mobile / e-paper / any HTTP client
```

One repository, three self-contained sub-projects. The only artifact that
crosses academy → observatory is a `.onnx` file; the only thing that crosses
observatory → web is HTTP/JSON (image files go straight to a shared
directory, never through the API). Swapping a model, backend, or camera is a
config change on one side of a seam — never a cross-project code change.

## Sub-projects

| Directory | What it is |
|---|---|
| [`academy/`](academy/README.md) | Config-driven training / evaluation / ONNX-optimization pipeline for cloud segmentation — YOLO11/26-seg and RF-DETR-seg backends, 3-stage progressive finetuning, export → simplify → fp16/int8 → benchmark → report |
| [`observatory/`](observatory/README.md) | C++23 inference engine — loads the ONNX model through ONNX Runtime (CPU / CUDA / TensorRT EPs behind an `IInferenceBackend` strategy), segments frames, post-processes masks into observations |
| [`web/`](web/README.md) | FastAPI + SQLite API and Svelte widget dashboard — stores observations and LoRa telemetry, serves them to any HTTP client, plugs external data in as cache-first providers |

Planned but not in the repo yet: `eyes/` (camera drivers & calibration),
`nest/` (ESP32 LoRa node firmware), `gateway/` (LoRa ↔ Pi bridge).

## Hardware

- **Raspberry Pi 5** — runs observatory and web side by side: inference,
  storage, API, dashboard, and (later) the LoRa gateway.
- **All-sky camera** — sensor under evaluation (IMX477 / IMX519 / IMX585 /
  …); observatory abstracts the camera behind an interface, so the choice
  stays swappable.
- **ESP32-C6 LoRa nodes (ESPRanger boards)** — temperature, humidity,
  pressure, wind, rain, UV over a private LoRa network. Firmware not
  started.
- **Dev PC with GPU** — training and frontend builds only; Node/npm and
  CUDA training never run on the Pi.

## Getting started

Each sub-project is fully self-contained — environment, dependencies, tests
and run instructions live in its own README. The short version:

- **academy** (dev PC, GPU) — `uv venv`, install `requirements.txt`, then
  `python train.py --config config.yaml` and
  `python optimize.py --config config.yaml --command pipeline`.
- **observatory** (Linux, GCC ≥ 14) —
  `cmake --preset debug && cmake --build --preset debug`.
- **web** — `uv venv`, install `requirements.txt`, then
  `uvicorn server.main:app --port 8080` (dashboard dev server:
  `cd frontend && npm run dev`).

## Status

- **academy** — functional end to end; first finetuning round benchmarked
  across nine checkpoints (`academy/SUMMARY.md` holds the current model
  comparison).
- **observatory** — CMake skeleton with all ten modules in place; the
  inference module (ONNX Runtime backend) is implemented and unit-tested,
  everything else is stubs. Wiring camera → pipeline is the current work.
- **web** — first working version: sectioned dashboard with 14 widgets,
  cache-first external providers, idempotent ingest for observations and
  auto-registering LoRa node telemetry.

## Roadmap (AI)

Sky/cloud segmentation *(models trained, deployment in progress)* → cloud
classification → cloud detection → solar irradiance estimation → temporal
cloud tracking → short-term forecasting.