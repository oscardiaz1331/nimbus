add_executable(observatory_logging_tests
  LoggingModule_test.cpp
)
target_link_libraries(observatory_logging_tests PRIVATE
  observatory::logging
  GTest::gtest_main
)
gtest_discover_tests(observatory_logging_tests)
