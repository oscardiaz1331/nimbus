add_executable(observatory_inference_tests
  IInferenceModel_test.cpp
  ISegmenter_test.cpp
  IInferenceBackend_test.cpp
  OnnxRuntimeBackend_test.cpp
  Tensor_test.cpp
)
target_link_libraries(observatory_inference_tests PRIVATE
  observatory::inference
  GTest::gtest_main
)
observatory_copy_onnxruntime_dlls(observatory_inference_tests)
gtest_discover_tests(observatory_inference_tests)
