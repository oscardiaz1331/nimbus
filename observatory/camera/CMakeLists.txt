add_library(observatory_camera STATIC
  src/placeholder.cpp
)
add_library(observatory::camera ALIAS observatory_camera)

observatory_set_warnings(observatory_camera)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
