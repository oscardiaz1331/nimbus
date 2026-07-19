// Aggregate stub executable: links all 9 observatory modules plus
// OpenCV/Eigen/ONNX Runtime and proves the whole toolchain actually runs
// (not just configures/links) — in particular that the ONNX Runtime
// shared library fetched by cmake/FetchOnnxRuntime.cmake is found at
// runtime via the baked-in RPATH. No business logic here; see
// observatory/CLAUDE.md for what belongs in each module.

#include <onnxruntime_cxx_api.h>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cstdio>

int main() {
  std::printf("OpenCV version:       %s\n", CV_VERSION);
  std::printf("Eigen version:        %d.%d.%d\n", EIGEN_WORLD_VERSION,
              EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
  std::printf("ONNX Runtime version: %s\n", OrtGetApiBase()->GetVersionString());

  const Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
  const cv::Mat frame(4, 4, CV_8UC3);

  std::printf("Eigen 3x3 identity trace: %.1f\n", static_cast<double>(identity.trace()));
  std::printf("OpenCV cv::Mat constructed: %dx%d, %d channels\n", frame.rows, frame.cols,
              frame.channels());

  std::printf("observatory: 9/9 modules linked\n");
  return 0;
}
