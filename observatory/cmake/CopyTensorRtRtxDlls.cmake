# NVIDIA's TensorRT RTX execution provider plugin for ONNX Runtime
# (onnxruntime_providers_nv_tensorrt_rtx.dll) isn't published by ONNX
# Runtime's own release or any package manager - it's built separately
# (see TensorRT-RTX-EP-ABI) and its output directory is whatever the
# building dev's machine happens to have, so there's nothing to
# FetchContent here. Point the OBSERVATORY_TENSORRT_RTX_EP_DIR environment
# variable at that build directory and this copies every .dll in it
# alongside the target's .exe, the same way CopyOpenCVDlls.cmake does for
# OpenCV. Globbing *.dll rather than just the provider .dll itself is
# deliberate: the factory also needs companion libraries dropped in that
# same directory (e.g. tensorrt_plugins.dll) that land there as build
# output too, and this way a future one gets picked up without touching
# this file again. Missing env var (or missing .dll in it) is not fatal:
# RegisterExecutionProviders() in OnnxRuntimeBackend.cpp already treats a
# missing provider library as "skip that EP", same as any deployment that
# just doesn't have it.
#
# Separately, OBSERVATORY_TENSORRT_RTX_SDK_BIN_DIR should point at the
# installed TensorRT-RTX SDK's bin/ directory (e.g. "...\TensorRT-RTX-1.5.0.114\bin").
# This one is NOT optional in practice, despite being handled the same
# lenient way: tensorrt_rtx_provider_factory.cc delay-loads
# tensorrt_rtx_*.dll/tensorrt_onnxparser_rtx_*.dll, and having that
# directory merely on PATH is not equivalent to having the DLL next to the
# .exe - on this project having only PATH resolve it produced an
# unconditional hard crash (uncatchable, no exception, no log line) the
# instant nvinfer1::createInferRuntime() made its first delay-loaded call,
# even though the exact same file loads fine via a direct LoadLibrary by
# full path. Copying next to the .exe is what actually fixed it.
if(WIN32)
  function(observatory_copy_tensorrt_rtx_dlls target)
    if(DEFINED ENV{OBSERVATORY_TENSORRT_RTX_EP_DIR})
      file(GLOB _observatory_tensorrt_rtx_ep_dlls "$ENV{OBSERVATORY_TENSORRT_RTX_EP_DIR}/*.dll")
      if(_observatory_tensorrt_rtx_ep_dlls)
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_observatory_tensorrt_rtx_ep_dlls} "$<TARGET_FILE_DIR:${target}>"
        )
      else()
        message(WARNING "observatory: OBSERVATORY_TENSORRT_RTX_EP_DIR is set to \"$ENV{OBSERVATORY_TENSORRT_RTX_EP_DIR}\" but no .dll files were found there")
      endif()
    endif()

    if(DEFINED ENV{OBSERVATORY_TENSORRT_RTX_SDK_BIN_DIR})
      file(GLOB _observatory_tensorrt_rtx_sdk_dlls "$ENV{OBSERVATORY_TENSORRT_RTX_SDK_BIN_DIR}/tensorrt_rtx_*.dll" "$ENV{OBSERVATORY_TENSORRT_RTX_SDK_BIN_DIR}/tensorrt_onnxparser_rtx_*.dll")
      if(_observatory_tensorrt_rtx_sdk_dlls)
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_observatory_tensorrt_rtx_sdk_dlls} "$<TARGET_FILE_DIR:${target}>"
        )
      else()
        message(WARNING "observatory: OBSERVATORY_TENSORRT_RTX_SDK_BIN_DIR is set to \"$ENV{OBSERVATORY_TENSORRT_RTX_SDK_BIN_DIR}\" but no tensorrt_rtx_*.dll/tensorrt_onnxparser_rtx_*.dll files were found there")
      endif()
    endif()
  endfunction()
else()
  function(observatory_copy_tensorrt_rtx_dlls target)
  endfunction()
endif()
