add_library(observatory_postprocessing STATIC
  src/placeholder.cpp
)
add_library(observatory::postprocessing ALIAS observatory_postprocessing)

target_include_directories(observatory_postprocessing PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

observatory_set_warnings(observatory_postprocessing)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
