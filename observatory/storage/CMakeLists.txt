add_library(observatory_storage STATIC
  src/placeholder.cpp
)
add_library(observatory::storage ALIAS observatory_storage)

observatory_set_warnings(observatory_storage)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
