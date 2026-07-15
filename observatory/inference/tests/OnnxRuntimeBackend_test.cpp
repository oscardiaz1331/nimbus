#include "observatory/inference/OnnxRuntimeBackend.hpp"

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

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
// fixtures/yolo26s-seg.onnx. Skips itself if that file isn't present.
TEST(OnnxRuntimeBackendRealModel, RunsYolo26sSeg) {
  const std::filesystem::path model_path = std::filesystem::path(OBSERVATORY_TEST_FIXTURES_DIR) / "yolo26s-seg.onnx";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path << " - see the comment above this test for how to generate it";
  }

  // Query the model's real input/output shapes instead of hardcoding them:
  // a -seg export's channel counts depend on class count/imgsz, which
  // aren't fixed here. Uses ONNX Runtime directly since OnnxRuntimeBackend
  // doesn't expose shape metadata yet (that's IInferenceModel::metadata(),
  // still TODO(design) - not worth adding just for this test).
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntimeBackend_test");
  Ort::SessionOptions options;
  Ort::Session probe_session(env, model_path.c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;

  // Assumes float32, static (batch=1) shapes throughout - true for a plain
  // `--command export` with no dynamic-axes/half flags. Tensor's constructor
  // throws on any non-positive dimension, so a model exported with dynamic
  // axes will fail loudly here rather than corrupt a buffer silently.
  std::vector<Tensor> input;
  for (size_t i = 0; i < probe_session.GetInputCount(); ++i) {
    auto shape = probe_session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    input.emplace_back(probe_session.GetInputNameAllocated(i, allocator).get(), std::move(shape), TensorDataType::kFloat32);
  }
  std::vector<Tensor> output;
  for (size_t i = 0; i < probe_session.GetOutputCount(); ++i) {
    auto shape = probe_session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    output.emplace_back(probe_session.GetOutputNameAllocated(i, allocator).get(), std::move(shape), TensorDataType::kFloat32);
  }

  OnnxRuntimeBackend backend(model_path.string(), InferenceBackendType::kOnnxRuntimeCPU);
  EXPECT_NO_THROW(backend.run(input, output));
}

}  // namespace
}  // namespace observatory::inference
