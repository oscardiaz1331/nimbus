add_library(observatory_inference STATIC
  src/OnnxRuntimeBackend.cpp
  src/OpenCVBackend.cpp
  src/Tensor.cpp
  src/YoloModel.cpp
)
add_library(observatory::inference ALIAS observatory_inference)

target_include_directories(observatory_inference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(observatory_inference PUBLIC
  onnxruntime::onnxruntime
  ${OpenCV_LIBS}
)

observatory_set_warnings(observatory_inference)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
