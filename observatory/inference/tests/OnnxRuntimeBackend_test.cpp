#include "observatory/inference/OnnxRuntimeBackend.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace observatory::inference {
namespace {

std::filesystem::path FixtureModelPath() {
  return std::filesystem::path(OBSERVATORY_TEST_FIXTURES_DIR) / "yolo11n-seg.onnx";
}

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
  const std::filesystem::path model_path = FixtureModelPath();
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

// getMetadata() reads whatever custom metadata_props academy embedded (see
// academy/utils/optimizers/metadata.py) - this fixture may or may not have
// any, depending on when it was (re)generated, so this only checks the call
// itself is safe (no throw, a plain map back) rather than asserting specific
// keys. YoloModel_test.cpp's ParseMetadata tests cover the actual parsing
// against a hand-built map, independent of what this binary fixture has.
TEST(OnnxRuntimeBackendRealModel, GetMetadataDoesNotThrow) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path << " - see the comment above RunsYolo11nSeg for how to generate it";
  }

  OnnxRuntimeBackend backend(model_path.string(), InferenceBackendType::kOnnxRuntimeBest);
  std::unordered_map<std::string, std::string> metadata;
  EXPECT_NO_THROW(metadata = backend.getMetadata());
}

// Exercises OnnxRuntimeBackend pinned to a specific execution provider,
// unlike RunsYolo11nSeg above which lets ONNX Runtime auto-select. GTEST_SKIPs
// rather than failing when the requested EP has no matching device on this
// machine - e.g. CUDA without cuDNN installed, or TensorRT-RTX before its EP
// plugin (built from a separate repo - see EpRegistrationName() in
// OnnxRuntimeBackend.cpp) is present - since not every machine running this
// suite has every EP wired up. Once the EP becomes available, this same test
// starts exercising real hardware without any code change.
class OnnxRuntimeBackendEp : public ::testing::TestWithParam<InferenceBackendType> {};

TEST_P(OnnxRuntimeBackendEp, RunsYolo11nSeg) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path
                 << " - see the comment on OnnxRuntimeBackendRealModel.RunsYolo11nSeg for how to generate it";
  }

  std::unique_ptr<OnnxRuntimeBackend> backend;
  try {
    backend = std::make_unique<OnnxRuntimeBackend>(model_path.string(), GetParam());
  } catch (const std::runtime_error &ex) {
    GTEST_SKIP() << "execution provider unavailable on this machine: " << ex.what();
  }

  std::vector<Tensor> input;
  input.emplace_back("images", std::vector<int64_t>{1, 3, 608, 608}, TensorDataType::kFloat32);

  std::vector<Tensor> output;
  output.emplace_back("output0", std::vector<int64_t>{1, 300, 38}, TensorDataType::kFloat32);
  output.emplace_back("output1", std::vector<int64_t>{1, 32, 152, 152}, TensorDataType::kFloat32);

  EXPECT_NO_THROW(backend->run(input, output));
}

INSTANTIATE_TEST_SUITE_P(
    ExecutionProviders, OnnxRuntimeBackendEp,
    ::testing::Values(InferenceBackendType::kOnnxRuntimeCPU, InferenceBackendType::kOnnxRuntimeCUDA,
                       InferenceBackendType::kOnnxRuntimeTensorRT),
    [](const ::testing::TestParamInfo<InferenceBackendType> &info) -> std::string {
      switch (info.param) {
        case InferenceBackendType::kOnnxRuntimeCPU:
          return "CPU";
        case InferenceBackendType::kOnnxRuntimeCUDA:
          return "CUDA";
        case InferenceBackendType::kOnnxRuntimeTensorRT:
          return "TensorRT_RTX";
        default:
          return "Unknown";
      }
    });

}  // namespace
}  // namespace observatory::inference
