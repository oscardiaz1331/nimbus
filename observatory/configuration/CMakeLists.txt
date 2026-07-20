add_library(observatory_configuration STATIC
  src/ConfigLoader.cpp
  src/PipelineFactory.cpp
)
add_library(observatory::configuration ALIAS observatory_configuration)

target_include_directories(observatory_configuration PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(observatory_configuration PUBLIC
  ${OpenCV_LIBS}
  observatory::inference
  observatory::preprocessing
  observatory::postprocessing
)

observatory_set_warnings(observatory_configuration)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
