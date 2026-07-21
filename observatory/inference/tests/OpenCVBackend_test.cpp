#include "observatory/inference/OpenCVBackend.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace observatory::inference {
namespace {

std::filesystem::path FixtureModelPath() {
  return std::filesystem::path(OBSERVATORY_TEST_FIXTURES_DIR) / "yolo11n-seg.onnx";
}

TEST(OpenCVBackend, ThrowsOnMissingFile) {
  EXPECT_THROW(OpenCVBackend backend("/nonexistent/path/model.onnx"), std::runtime_error);
}

TEST(OpenCVBackend, ThrowsOnNonOnnxExtension) {
  EXPECT_THROW(OpenCVBackend backend(__FILE__), std::runtime_error);
}

// Unlike OnnxRuntimeBackendRealModel.RunsYolo11nSeg, this fixture does NOT
// currently load through OpenCVBackend: it's an NMS-embedded export (see the
// comment on OnnxRuntimeBackendRealModel.RunsYolo11nSeg), and OpenCV 5's DNN
// module can't run that particular postprocessing subgraph on either engine
// as of this OpenCV build - ENGINE_CLASSIC (what OpenCVBackend uses) rejects
// one of its Transpose nodes at import time; the alternative, ENGINE_AUTO,
// imports fine but SIGFPEs inside GatherNDLayerImpl during forward() (a crash,
// not an exception - confirmed by hand, not something a test can assert on).
// This documents the current, real limitation rather than pretending the
// fixture works: OpenCVBackend is exercised end-to-end once academy/ exports
// a plain (non-NMS-embedded) graph.
TEST(OpenCVBackendRealModel, ThrowsOnNmsEmbeddedFixture) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path;
  }

  EXPECT_THROW(OpenCVBackend backend(model_path.string()), std::runtime_error);
}

}  // namespace
}  // namespace observatory::inference
