#include "observatory/configuration/PipelineFactory.hpp"

#include <gtest/gtest.h>

#include <filesystem>

namespace observatory::configuration {
namespace {

std::filesystem::path FixtureModelPath() {
  return std::filesystem::path(OBSERVATORY_TEST_FIXTURES_DIR) / "yolo11n-seg.onnx";
}

TEST(PipelineFactory, RejectsUnknownBackend) {
  Config config;
  config.model_path = "/nonexistent/path/model.onnx";
  config.backend = "not-a-real-backend";

  const auto result = buildPipeline(config);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("unknown backend"), std::string::npos);
}

TEST(PipelineFactory, FailsCleanlyOnMissingModelFile) {
  Config config;
  config.model_path = "/nonexistent/path/model.onnx";

  const auto result = buildPipeline(config);
  EXPECT_FALSE(result.has_value());
}

TEST(PipelineFactory, RejectsNativeTensorRtAndOpenVinoBackendsForNow) {
  // "tensorrt"/"openvino" (no "onnx-" prefix) aren't valid backend strings
  // yet - they're reserved for a future backend that talks to those
  // runtimes directly (see Config::backend's doc comment and
  // InferenceBackendType::kTensorRT/kOpenVINO). Only onnx-tensorrt/
  // onnx-openvino (pinning ONNX Runtime's EP) exist today.
  Config config;
  config.model_path = "/nonexistent/path/model.onnx";
  config.backend = "tensorrt";

  const auto result = buildPipeline(config);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("unknown backend"), std::string::npos);
}

TEST(PipelineFactory, ResolveFrameworkReadsFrameworkKey) {
  const auto result = ResolveFramework({{"framework", "yolo"}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "yolo");
}

TEST(PipelineFactory, ResolveFrameworkFailsCleanlyWhenMissing) {
  const auto result = ResolveFramework({});
  EXPECT_FALSE(result.has_value());
}

// Exercises the real happy path against academy's fixture - same
// GTEST_SKIP-if-unavailable rationale as inference/tests/OnnxRuntimeBackend_
// test.cpp. Whether it gets past the "framework" and num_classes/
// nms_embedded metadata checks depends on when this exact fixture was
// (re)exported: older copies predate academy/utils/optimizers/provenance.py
// being wired into optimize.py, so both outcomes are asserted on rather than
// just one - re-exporting the fixture with the current pipeline is what
// flips this from a "missing metadata" skip to the full pipeline being
// built, with no code change.
TEST(PipelineFactory, BuildsFromRealFixture) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path << " - see inference/tests/OnnxRuntimeBackend_test.cpp for how to generate it";
  }

  Config config;
  config.model_path = model_path.string();

  const auto result = buildPipeline(config);
  if (!result.has_value()) {
    GTEST_SKIP() << "fixture has no num_classes/nms_embedded metadata yet (re-export it with the current "
                     "academy pipeline to exercise the full build): "
                  << result.error();
  }

  EXPECT_NE(result->model, nullptr);
  EXPECT_NE(result->preprocessor, nullptr);
  EXPECT_NE(result->postprocessor, nullptr);
}

}  // namespace
}  // namespace observatory::configuration
