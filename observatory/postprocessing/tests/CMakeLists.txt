add_executable(observatory_postprocessing_tests
  IPostprocessor_test.cpp
)
target_link_libraries(observatory_postprocessing_tests PRIVATE
  observatory::postprocessing
  GTest::gtest_main
)
gtest_discover_tests(observatory_postprocessing_tests)
