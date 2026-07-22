add_executable(observatory_configuration_tests
  ConfigLoader_test.cpp
  PipelineFactory_test.cpp
)
target_compile_definitions(observatory_configuration_tests PRIVATE
  # Shares inference/'s fixture rather than duplicating the (large) .onnx file.
  OBSERVATORY_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../inference/tests/fixtures"
  # So ConfigLoader_test.cpp can load the real example config.yaml at the
  # repo root and catch it drifting out of sync with Config's schema.
  OBSERVATORY_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
target_link_libraries(observatory_configuration_tests PRIVATE
  observatory::configuration
  GTest::gtest_main
)
observatory_copy_opencv_dlls(observatory_configuration_tests)
observatory_copy_onnxruntime_dlls(observatory_configuration_tests)
observatory_copy_tensorrt_rtx_dlls(observatory_configuration_tests)
gtest_discover_tests(observatory_configuration_tests)
