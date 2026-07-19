add_executable(observatory_preprocessing_tests
  IPreprocessor_test.cpp
  YoloSegPreprocessor_test.cpp
)
target_link_libraries(observatory_preprocessing_tests PRIVATE
  observatory::preprocessing
  GTest::gtest_main
)
gtest_discover_tests(observatory_preprocessing_tests)
