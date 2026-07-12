add_executable(observatory_tracking_tests
  ITracker_test.cpp
)
target_link_libraries(observatory_tracking_tests PRIVATE
  observatory::tracking
  GTest::gtest_main
)
gtest_discover_tests(observatory_tracking_tests)
