add_library(observatory_communication STATIC
  src/placeholder.cpp
)
add_library(observatory::communication ALIAS observatory_communication)

observatory_set_warnings(observatory_communication)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
