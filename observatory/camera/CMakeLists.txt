add_library(observatory_camera STATIC
  src/FileCamera.cpp
)
add_library(observatory::camera ALIAS observatory_camera)

target_include_directories(observatory_camera PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(observatory_camera PUBLIC
  ${OpenCV_LIBS}
)

observatory_set_warnings(observatory_camera)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
