add_executable(observatory
  src/main.cpp
)
target_link_libraries(observatory PRIVATE
  observatory::camera
  observatory::preprocessing
  observatory::inference
  observatory::postprocessing
  observatory::tracking
  observatory::telemetry
  observatory::storage
  observatory::configuration
  observatory::logging
  observatory::communication
)

observatory_set_warnings(observatory)
observatory_copy_onnxruntime_dlls(observatory)
