add_executable(observatory_communication_tests
  CommunicationModule_test.cpp
)
target_link_libraries(observatory_communication_tests PRIVATE
  observatory::communication
  GTest::gtest_main
)
gtest_discover_tests(observatory_communication_tests)
