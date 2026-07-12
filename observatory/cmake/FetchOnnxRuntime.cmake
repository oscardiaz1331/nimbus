include(FetchContent)

set(OBSERVATORY_ONNXRUNTIME_VERSION "1.27.0" CACHE STRING "ONNX Runtime release to fetch")

if(OBSERVATORY_ENABLE_CUDA)
  # Only cuda12 and cuda13 GPU packages exist for this ORT release. Default to
  # whatever find_package(CUDAToolkit) actually detected (set by the root
  # CMakeLists.txt) instead of hardcoding a version, since the installed
  # toolkit on this machine is expected to change (12.x now, 13.x soon).
  if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION_MAJOR MATCHES "^(12|13)$")
    set(_observatory_cuda_major_default "${CUDAToolkit_VERSION_MAJOR}")
  else()
    if(CUDAToolkit_FOUND)
      message(WARNING "observatory: detected CUDA toolkit major version ${CUDAToolkit_VERSION_MAJOR} has no matching ONNX Runtime GPU package (only 12 and 13 exist); defaulting to 12")
    endif()
    set(_observatory_cuda_major_default "12")
  endif()

  set(OBSERVATORY_ONNXRUNTIME_CUDA_MAJOR "${_observatory_cuda_major_default}" CACHE STRING "ONNX Runtime GPU package CUDA major version (12 or 13)")
  set(_observatory_ort_pkg "onnxruntime-linux-x64-gpu_cuda${OBSERVATORY_ONNXRUNTIME_CUDA_MAJOR}-${OBSERVATORY_ONNXRUNTIME_VERSION}")
  message(STATUS "observatory: fetching CUDA-enabled ONNX Runtime (${_observatory_ort_pkg}). "
                  "Note: cuDNN is required at runtime for the CUDA execution provider and is not installed by this build.")
  unset(_observatory_cuda_major_default)
else()
  set(_observatory_ort_pkg "onnxruntime-linux-x64-${OBSERVATORY_ONNXRUNTIME_VERSION}")
endif()

FetchContent_Declare(onnxruntime
  URL "https://github.com/microsoft/onnxruntime/releases/download/v${OBSERVATORY_ONNXRUNTIME_VERSION}/${_observatory_ort_pkg}.tgz"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

if(NOT EXISTS "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
  message(FATAL_ERROR "observatory: expected ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so after fetching ${_observatory_ort_pkg}.tgz, not found")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
  INTERFACE_LINK_OPTIONS "-Wl,-rpath,${onnxruntime_SOURCE_DIR}/lib"
)

unset(_observatory_ort_pkg)
