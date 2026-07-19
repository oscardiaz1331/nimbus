add_library(observatory_preprocessing STATIC
  src/YoloSegPreprocessor.cpp
)
add_library(observatory::preprocessing ALIAS observatory_preprocessing)

target_include_directories(observatory_preprocessing PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(observatory_preprocessing PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  observatory::inference
)

observatory_set_warnings(observatory_preprocessing)

if(OBSERVATORY_BUILD_TESTS)
  add_subdirectory(tests)
endif()
