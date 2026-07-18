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

## Execution Providers

`OnnxRuntimeBackend` picks its execution provider (EP) from `InferenceBackendType`
(see `IInferenceBackend.hpp`). Only CPU, CUDA, and TensorRT-RTX are actually
wired up and test-covered today — the parameterized EP tests in
`inference/tests/OnnxRuntimeBackend_test.cpp` and `YoloModel_test.cpp`
construct a real backend for each and `GTEST_SKIP()` with the exact reason
(missing library, no matching device, …) when one isn't available on the
current machine, instead of failing. This section is what to install so a
given EP actually runs instead of skipping.

### CPU (`kOnnxRuntimeCPU`)

Nothing to install — ships inside every ONNX Runtime build. Always available.

### CUDA (`kOnnxRuntimeCUDA`)

On top of an NVIDIA GPU + driver:

1. **CUDA Toolkit**, major version 12 or 13. Install from
   [NVIDIA's own apt repo](https://developer.nvidia.com/cuda-downloads), not
   the distro's `nvidia-cuda-toolkit` package — see the WSL2 note below for
   why that matters. `cmake/FetchOnnxRuntime.cmake` auto-detects the major
   version via `find_package(CUDAToolkit)` and fetches the matching prebuilt
   ONNX Runtime GPU package.
2. **cuDNN 9.x** — required at runtime by ONNX Runtime's CUDA EP. Not fetched
   by this build; install it separately (NVIDIA's cuDNN apt repo/package).
3. Build with the CUDA-enabled preset:
   ```bash
   cmake --preset debug-cuda
   cmake --build --preset debug-cuda
   ```
   (or `-DOBSERVATORY_ENABLE_CUDA=ON` on any preset — it's auto-enabled
   whenever `find_package(CUDAToolkit)` succeeds).

### TensorRT-RTX (`kOnnxRuntimeTensorRT`, EP name `NvTensorRTRTXExecutionProvider`)

Not shipped by ONNX Runtime — it's a separate plugin library
(`libonnxruntime_providers_nv_tensorrt_rtx.so`) built from NVIDIA's
[TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) repo.
On top of CUDA's prerequisites above:

1. **TensorRT-RTX SDK** (tarball from NVIDIA — requires an Ampere-or-newer
   GPU) and **CUDA Toolkit 12.9+**.
2. Build the plugin:
   ```bash
   git clone https://github.com/NVIDIA/TensorRT-RTX-EP-ABI.git
   cd TensorRT-RTX-EP-ABI
   cmake -B build \
     -DONNXRUNTIME_ROOT=/path/to/observatory/build/debug/_deps/onnxruntime-src \
     -DTRT_RTX_ROOT=/path/to/TensorRT-RTX-<version>
   cmake --build build --config Release --parallel
   ```
   `ONNXRUNTIME_ROOT` can point straight at the ONNX Runtime SDK this repo
   already fetched (`build/<preset>/_deps/onnxruntime-src`) — no separate
   download needed. This also builds `protobuf`/`abseil`/`onnx` from source
   via `FetchContent`, so it takes a while the first time.
3. Copy the resulting `libonnxruntime_providers_nv_tensorrt_rtx.so` next to
   whichever executable will load it —
   `OnnxRuntimeBackend::Impl::RegisterExecutionProviders()` only looks in the
   running executable's own directory (e.g.
   `build/debug/inference/tests/` for the test binary,
   `build/debug/app/` for `observatory`).

### Verifying an EP actually runs

```bash
ctest --preset debug --output-on-failure
# or target one EP directly:
./build/debug/inference/tests/observatory_inference_tests --gtest_filter="*CUDA*:*TensorRT_RTX*"
```

### WSL2 caveat

CUDA and TensorRT-RTX both go through ONNX Runtime's V2 device-discovery API
(`AppendExecutionProvider_V2`/`GetEpDevices()`), which on Linux requires a
real PCI `sysfs` entry to match the GPU against. WSL2 doesn't provide one —
it exposes the GPU through paravirtualization (`/dev/dxg` + `dxgkrnl`), so
`/sys/bus/pci/devices` and `lspci` never show the real NVIDIA device even
though `nvidia-smi`/CUDA itself work fine. Both EPs will report "no devices
found" under WSL2 no matter how correctly everything above is installed —
that's a WSL2 architecture limitation (real PCI passthrough for WSL2 is an
open, unresolved feature request:
[microsoft/WSL#5492](https://github.com/microsoft/WSL/issues/5492)), not a
missing install step. Validate CUDA/TensorRT-RTX on bare-metal Linux instead.

Also watch out for apt's distro `nvidia-cuda-toolkit` package: it pulls in
`libnvidia-compute-<version>`, a real Linux GPU driver userspace package that
WSL2 should never have installed (its `libnvidia-ptxjitcompiler.so` shadows
the correct one WSL2 ships under `/usr/lib/wsl/drivers/.../`, and CUDA calls
that need it — e.g. `cudaGetDeviceCount()` — segfault). Use NVIDIA's own CUDA
apt repo instead, and if you've already got the distro package,
`sudo apt-get purge nvidia-cuda-toolkit libnvidia-compute-<version> && sudo apt-get autoremove --purge`.

### Raspberry Pi / ARM

No NVIDIA GPU on the Pi, so CUDA and TensorRT-RTX don't apply there — only
the CPU EP does. One thing to fix before even that builds, though:
`cmake/FetchOnnxRuntime.cmake` currently hardcodes the `linux-x64` ONNX
Runtime package name with no `aarch64` branch, so as-is it would fetch the
wrong architecture on a Pi. ONNX Runtime does publish an
`onnxruntime-linux-aarch64-<version>.tgz` release asset — that branch just
hasn't been added here yet.

## Running

`app/` links all ten modules into a single `observatory` executable. Right
now it's only a toolchain smoke test (prints OpenCV/Eigen/ONNX Runtime
versions and confirms everything links and runs) — the real capture → infer
→ postprocess → communicate pipeline isn't wired up yet.

```bash
./build/debug/app/observatory
```
