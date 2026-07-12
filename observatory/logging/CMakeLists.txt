add_library(observatory_logging STATIC
  src/placeholder.cpp
)
add_library(observatory::logging ALIAS observatory_logging)

observatory_set_warnings(observatory_logging)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
