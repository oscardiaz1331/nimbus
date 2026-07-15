# Nimbus Observatory

C++23 inference engine for the Nimbus all-sky cloud observation system.
Takes camera frames in, runs an ONNX segmentation model, and hands the
resulting masks off to the weather-intelligence layer. Training happens in
`academy/`, the dashboard lives in `web/` — this repo only loads, infers,
and post-processes.

Everything goes through interfaces (Strategy/Factory/DI): swapping a model,
backend, or camera is a config change, not a recompile. See
[`CLAUDE.md`](CLAUDE.md) for the full architecture and conventions.

```
observatory/
├── camera/          # consumes eyes/ through an interface — placeholder
├── preprocessing/    # resize/normalize/letterbox/undistort — placeholder
├── inference/         # ONNX Runtime backend — implemented
│   ├── include/observatory/inference/
│   │   ├── IInferenceModel.hpp    # load()/infer()/warmup()/metadata()
│   │   ├── ISegmenter.hpp         # segmentation strategy interface
│   │   ├── IInferenceBackend.hpp  # ONNX Runtime/TensorRT/OpenVINO strategy
│   │   ├── Tensor.hpp             # backend-agnostic named tensor
│   │   └── OnnxRuntimeBackend.hpp # pImpl - no ORT types leak into this header
│   └── src/
│       ├── Tensor.cpp
│       └── OnnxRuntimeBackend.cpp # EP selection, IoBinding, CPU<->device
│                                   #   sync-stream pipeline (upload/compute/
│                                   #   download), device-agnostic memory
├── postprocessing/   # NMS/masks/cloud-% stats — placeholder
├── tracking/          # temporal tracking (Stage 5, not started) — placeholder
├── telemetry/         # LoRa sensor node data — placeholder
├── storage/           # local persistence before the POST to web/ — placeholder
├── configuration/     # config-driven camera/model/backend/… — placeholder
├── logging/           # placeholder
├── communication/     # POST /api/v1/observations to web/ — placeholder
├── app/               # aggregate executable; currently a toolchain smoke test
├── cmake/             # CompilerWarnings, FetchOnnxRuntime, FetchGoogleTest
└── CMakePresets.json   # debug / release / debug-cuda
```

`inference/` is the only module with a real implementation so far
(`OnnxRuntimeBackend`: execution-provider registration and selection across
CPU/CUDA/TensorRT/OpenVINO/QNN/CANN, `IoBinding`, per-tensor CPU-vs-device
placement resolved once at construction, and an upload/compute/download
`OrtSyncStream` + notification pipeline for async CPU↔device transfers with
a synchronous fallback when an EP doesn't support streams). Every other
module is a `placeholder.cpp` stub that proves the CMake target links.

## Requirements

- Linux, GCC ≥ 14 (C++23; the project pins `g++-14`/`gcc-14` — see below), CMake ≥ 3.28, Ninja
- System packages: OpenCV, Eigen3 (≥ 3.4)
- Optional: CUDA Toolkit (auto-detected; enables the CUDA execution provider)

ONNX Runtime and GoogleTest are **not** system dependencies — CMake fetches
them automatically (see `cmake/FetchOnnxRuntime.cmake`,
`cmake/FetchGoogleTest.cmake`).

`CMakePresets.json` pins `CMAKE_CXX_COMPILER`/`CMAKE_C_COMPILER` to
`g++-14`/`gcc-14`, so `cmake --preset` will fail to configure until they're
installed. On Ubuntu/Debian:

```bash
sudo apt install gcc-14 g++-14
```

## Building

The project uses CMake presets, so there's nothing to configure by hand:

```bash
cmake --preset debug          # or: release, debug-cuda
cmake --build --preset debug
```

Each preset gets its own build directory (`build/debug`, `build/release`, …),
same idea as switching Debug/Release configurations in Visual Studio. Once
configured, day-to-day iteration can go straight through Ninja from inside
the build directory, which is faster when you only care about one module:

```bash
cd build/debug
ninja observatory_inference           # just this module
ninja observatory_inference_tests
ninja                                 # everything
```

Useful `cmake --preset` cache variables (`-D<VAR>=<value>` at configure
time, or edit `CMakePresets.json`):

| Variable | Default | Purpose |
|---|---|---|
| `OBSERVATORY_BUILD_TESTS` | `ON` | Build the GoogleTest suites per module |
| `OBSERVATORY_ENABLE_CUDA` | auto (`CUDAToolkit_FOUND`) | Fetch the CUDA-enabled ONNX Runtime build |
| `OBSERVATORY_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` on top of the warning set in `cmake/CompilerWarnings.cmake` |

## Testing

```bash
ctest --preset debug --output-on-failure
```

Every module has its own GoogleTest binary under `<module>/tests/` (e.g.
`build/debug/inference/tests/observatory_inference_tests`), so you can also
run just one directly.

## Running

`app/` links all ten modules into a single `observatory` executable. Right
now it's only a toolchain smoke test (prints OpenCV/Eigen/ONNX Runtime
versions and confirms everything links and runs) — the real capture → infer
→ postprocess → communicate pipeline isn't wired up yet.

```bash
./build/debug/app/observatory
```
