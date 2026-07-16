#include "observatory/inference/OnnxRuntimeBackend.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace observatory::inference {
namespace {

TEST(OnnxRuntimeBackend, ThrowsOnMissingFile) {
  EXPECT_THROW(OnnxRuntimeBackend backend("/nonexistent/path/model.onnx"), std::runtime_error);
}

TEST(OnnxRuntimeBackend, ThrowsOnNonOnnxExtension) {
  EXPECT_THROW(OnnxRuntimeBackend backend(__FILE__), std::runtime_error);
}

// Smoke test against a real academy-exported model - not committed to git
// (see fixtures/.gitignore). Generate it on the machine that has the
// trained checkpoint (config.yaml's model.checkpoint) with:
//   python optimize.py --config config.yaml --command export
// then copy the resulting runs/yolo/yolo26s-seg/optimize/model.onnx here as
// fixtures/yolo11n-seg.onnx. Skips itself if that file isn't present.
//
// Shapes below are hardcoded from this exact fixture (inspected once with
// onnxruntime, not queried at test time - OnnxRuntimeBackend deliberately
// keeps onnxruntime_cxx_api.h out of its public header so callers only ever
// deal in Tensor, and this test follows that same rule): input "images" is
// [1,3,608,608]; output0 [1,300,38] looks like an end-to-end NMS-baked
// export (4 bbox + conf + class_id + 32 mask coeffs per of up to 300
// detections); output1 [1,32,152,152] is the mask prototype tensor. Swap
// these if the fixture is ever re-exported with different settings.
TEST(OnnxRuntimeBackendRealModel, RunsYolo11nSeg) {
  const std::filesystem::path model_path = std::filesystem::path(OBSERVATORY_TEST_FIXTURES_DIR) / "yolo11n-seg.onnx";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path << " - see the comment above this test for how to generate it";
  }

  std::vector<Tensor> input;
  input.emplace_back("images", std::vector<int64_t>{1, 3, 608, 608}, TensorDataType::kFloat32);

  std::vector<Tensor> output;
  output.emplace_back("output0", std::vector<int64_t>{1, 300, 38}, TensorDataType::kFloat32);
  output.emplace_back("output1", std::vector<int64_t>{1, 32, 152, 152}, TensorDataType::kFloat32);

  OnnxRuntimeBackend backend(model_path.string(), InferenceBackendType::kOnnxRuntimeBest);
  EXPECT_NO_THROW(backend.run(input, output));
}

}  // namespace
}  // namespace observatory::inference
