add_executable(observatory_telemetry_tests
  TelemetryModule_test.cpp
)
target_link_libraries(observatory_telemetry_tests PRIVATE
  observatory::telemetry
  GTest::gtest_main
)
gtest_discover_tests(observatory_telemetry_tests)
