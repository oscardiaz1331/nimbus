add_executable(observatory_storage_tests
  StorageModule_test.cpp
)
target_link_libraries(observatory_storage_tests PRIVATE
  observatory::storage
  GTest::gtest_main
)
gtest_discover_tests(observatory_storage_tests)
