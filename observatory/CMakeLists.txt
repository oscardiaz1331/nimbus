cmake_minimum_required(VERSION 3.28)
project(observatory VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed; use cmake -B build (or a preset)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

find_package(CUDAToolkit QUIET)

option(OBSERVATORY_BUILD_TESTS "Build unit tests" ON)
option(OBSERVATORY_ENABLE_CUDA "Fetch/link the CUDA-enabled ONNX Runtime build" ${CUDAToolkit_FOUND})
option(OBSERVATORY_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include(FetchOnnxRuntime)

if(OBSERVATORY_BUILD_TESTS)
  include(FetchGoogleTest)
endif()

add_subdirectory(camera)
add_subdirectory(preprocessing)
add_subdirectory(inference)
add_subdirectory(postprocessing)
add_subdirectory(tracking)
add_subdirectory(telemetry)
add_subdirectory(storage)
add_subdirectory(configuration)
add_subdirectory(logging)
add_subdirectory(communication)
add_subdirectory(app)
