add_library(observatory_configuration STATIC
  src/placeholder.cpp
)
add_library(observatory::configuration ALIAS observatory_configuration)

observatory_set_warnings(observatory_configuration)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
