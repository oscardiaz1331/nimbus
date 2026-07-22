add_executable(observatory_camera_tests
  ICamera_test.cpp
  FileCamera_test.cpp
)
target_link_libraries(observatory_camera_tests PRIVATE
  observatory::camera
  GTest::gtest_main
)
observatory_copy_opencv_dlls(observatory_camera_tests)
gtest_discover_tests(observatory_camera_tests)
