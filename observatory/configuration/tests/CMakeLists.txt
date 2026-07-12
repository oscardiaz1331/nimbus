add_executable(observatory_configuration_tests
  ConfigurationModule_test.cpp
)
target_link_libraries(observatory_configuration_tests PRIVATE
  observatory::configuration
  GTest::gtest_main
)
gtest_discover_tests(observatory_configuration_tests)
