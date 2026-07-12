add_executable(observatory_camera_tests
  CameraModule_test.cpp
)
target_link_libraries(observatory_camera_tests PRIVATE
  observatory::camera
  GTest::gtest_main
)
gtest_discover_tests(observatory_camera_tests)
