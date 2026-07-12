add_library(observatory_tracking STATIC
  src/placeholder.cpp
)
add_library(observatory::tracking ALIAS observatory_tracking)

target_include_directories(observatory_tracking PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

observatory_set_warnings(observatory_tracking)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
