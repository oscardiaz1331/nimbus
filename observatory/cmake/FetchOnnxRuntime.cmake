include(FetchContent)

set(OBSERVATORY_ONNXRUNTIME_VERSION "1.27.1" CACHE STRING "ONNX Runtime release to fetch")

if(WIN32)
  set(_observatory_ort_os "win")
  set(_observatory_ort_ext "zip")
else()
  set(_observatory_ort_os "linux")
  set(_observatory_ort_ext "tgz")
endif()

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
  set(_observatory_ort_pkg "onnxruntime-${_observatory_ort_os}-x64-gpu_cuda${OBSERVATORY_ONNXRUNTIME_CUDA_MAJOR}-${OBSERVATORY_ONNXRUNTIME_VERSION}")
  message(STATUS "observatory: fetching CUDA-enabled ONNX Runtime (${_observatory_ort_pkg}). "
                  "Note: cuDNN is required at runtime for the CUDA execution provider and is not installed by this build.")
  unset(_observatory_cuda_major_default)
else()
  set(_observatory_ort_pkg "onnxruntime-${_observatory_ort_os}-x64-${OBSERVATORY_ONNXRUNTIME_VERSION}")
endif()

FetchContent_Declare(onnxruntime
  URL "https://github.com/microsoft/onnxruntime/releases/download/v${OBSERVATORY_ONNXRUNTIME_VERSION}/${_observatory_ort_pkg}.${_observatory_ort_ext}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)

if(WIN32)
  # @todo(windows): fetch/link wiring only - nobody has built+run this on a
  #   real Windows machine yet. Windows has no rpath equivalent, so (unlike
  #   the Linux branch below, where INTERFACE_LINK_OPTIONS's rpath is enough
  #   on its own) executables also need the actual .dlls copied next to them;
  #   see observatory_copy_onnxruntime_dlls() below, called from app/ and
  #   inference/tests/.
  if(NOT EXISTS "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll" OR NOT EXISTS "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib")
    message(FATAL_ERROR "observatory: expected ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll and .lib after fetching ${_observatory_ort_pkg}.${_observatory_ort_ext}, not found")
  endif()
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
  )

  file(GLOB _observatory_ort_dlls "${onnxruntime_SOURCE_DIR}/lib/*.dll")

  # @todo(windows): untested. Copies every .dll ORT ships (onnxruntime.dll
  # plus onnxruntime_providers_shared.dll, which onnxruntime.dll loads
  # dynamically at runtime) next to `target`'s .exe - the Windows-side
  # equivalent of what the Linux branch's rpath gives for free.
  function(observatory_copy_onnxruntime_dlls target)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_observatory_ort_dlls} "$<TARGET_FILE_DIR:${target}>"
    )
  endfunction()
else()
  if(NOT EXISTS "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
    message(FATAL_ERROR "observatory: expected ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so after fetching ${_observatory_ort_pkg}.${_observatory_ort_ext}, not found")
  endif()
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
    INTERFACE_LINK_OPTIONS "-Wl,-rpath,${onnxruntime_SOURCE_DIR}/lib"
  )

  # rpath above already makes libonnxruntime.so's own dependencies (e.g.
  # libonnxruntime_providers_shared.so) resolvable, so nothing to do for
  # those. But OnnxRuntimeBackend::Impl::RegisterExecutionProviders() looks
  # for EP libraries (cuda, tensorrt, openvino, ...) next to the running
  # executable itself (see GetExecutablePath() in OnnxRuntimeBackend.cpp) -
  # rpath doesn't help there, so those still need copying, same as Windows'
  # DLLs below.
  file(GLOB _observatory_ort_provider_libs "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_*.so")
  function(observatory_copy_onnxruntime_dlls target)
    if(_observatory_ort_provider_libs)
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_observatory_ort_provider_libs} "$<TARGET_FILE_DIR:${target}>"
      )
    endif()
  endfunction()
endif()

unset(_observatory_ort_pkg)
unset(_observatory_ort_os)
unset(_observatory_ort_ext)
