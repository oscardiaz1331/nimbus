add_library(observatory_telemetry STATIC
  src/placeholder.cpp
)
add_library(observatory::telemetry ALIAS observatory_telemetry)

observatory_set_warnings(observatory_telemetry)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
