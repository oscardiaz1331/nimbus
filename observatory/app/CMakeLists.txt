add_executable(observatory
  src/main.cpp
)
target_link_libraries(observatory PRIVATE
  observatory::camera
  observatory::preprocessing
  observatory::inference
  observatory::postprocessing
  observatory::tracking
  observatory::storage
  observatory::configuration
  observatory::logging
  observatory::communication
)

observatory_set_warnings(observatory)
observatory_copy_opencv_dlls(observatory)
observatory_copy_onnxruntime_dlls(observatory)
observatory_copy_tensorrt_rtx_dlls(observatory)

add_executable(observatory_benchmark
  src/benchmark.cpp
)
target_link_libraries(observatory_benchmark PRIVATE
  observatory::camera
  observatory::configuration
)

observatory_set_warnings(observatory_benchmark)
observatory_copy_opencv_dlls(observatory_benchmark)
observatory_copy_onnxruntime_dlls(observatory_benchmark)
observatory_copy_tensorrt_rtx_dlls(observatory_benchmark)
