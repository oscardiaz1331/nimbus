#include "observatory/inference/YoloModel.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "observatory/inference/OnnxRuntimeBackend.hpp"

namespace observatory::inference {
namespace {

// Same fixture as OnnxRuntimeBackend_test.cpp's RunsYolo11nSeg - see the
// comment there for how to (re)generate it. Shapes: input "images" is
// [1,3,608,608]; output0 [1,300,38]; output1 [1,32,152,152].
std::filesystem::path FixtureModelPath() {
  return std::filesystem::path(OBSERVATORY_TEST_FIXTURES_DIR) / "yolo11n-seg.onnx";
}

// YoloModel now takes an already-built backend (dependency injection - see
// its constructor doc comment), so every test needs one of these instead of
// a bare (path, ep_type) pair.
YoloModel MakeModel(const std::filesystem::path& model_path,
                     InferenceBackendType ep_type = InferenceBackendType::kOnnxRuntimeBest) {
  return YoloModel(std::make_unique<OnnxRuntimeBackend>(model_path.string(), ep_type));
}

TEST(YoloModelRealModel, WarmupAndInferSucceed) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path
                 << " - see OnnxRuntimeBackend_test.cpp for how to generate it";
  }

  YoloModel model = MakeModel(model_path);

  EXPECT_NO_THROW(model.warmup(2));

  std::vector<Tensor> input;
  input.emplace_back("images", std::vector<int64_t>{1, 3, 608, 608}, TensorDataType::kFloat32);

  const auto result = model.infer(input);
  ASSERT_TRUE(result.has_value()) << result.error();
  ASSERT_EQ(result->size(), 2U);
  EXPECT_EQ((*result)[0].shape(), (std::vector<int64_t>{1, 300, 38}));
  EXPECT_EQ((*result)[1].shape(), (std::vector<int64_t>{1, 32, 152, 152}));
}

TEST(YoloModelRealModel, MetadataReportsInputSizeFromBackendTensorShape) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path
                 << " - see OnnxRuntimeBackend_test.cpp for how to generate it";
  }

  YoloModel model = MakeModel(model_path);

  // Fixture's declared input shape is [1,3,608,608] (see the comment above).
  EXPECT_EQ(model.metadata().input_size, 608);
}

TEST(YoloModelParseMetadata, ReadsNumClassesAndNmsEmbeddedWhenPresent) {
  const YoloModelMetadata metadata = YoloModel::ParseMetadata({{"num_classes", "2"}, {"nms_embedded", "true"}});
  ASSERT_TRUE(metadata.num_classes.has_value());
  EXPECT_EQ(*metadata.num_classes, 2);
  ASSERT_TRUE(metadata.nms_embedded.has_value());
  EXPECT_TRUE(*metadata.nms_embedded);
}

TEST(YoloModelParseMetadata, ReadsNmsEmbeddedFalse) {
  const YoloModelMetadata metadata = YoloModel::ParseMetadata({{"nms_embedded", "false"}});
  ASSERT_TRUE(metadata.nms_embedded.has_value());
  EXPECT_FALSE(*metadata.nms_embedded);
}

TEST(YoloModelParseMetadata, MissingKeysLeaveFieldsNullopt) {
  const YoloModelMetadata metadata = YoloModel::ParseMetadata({});
  EXPECT_FALSE(metadata.num_classes.has_value());
  EXPECT_FALSE(metadata.nms_embedded.has_value());
}

TEST(YoloModelParseMetadata, MalformedValuesAreTreatedAsMissing) {
  const YoloModelMetadata metadata = YoloModel::ParseMetadata({{"num_classes", "not-a-number"}, {"nms_embedded", "yes"}});
  EXPECT_FALSE(metadata.num_classes.has_value());
  EXPECT_FALSE(metadata.nms_embedded.has_value());
}

TEST(YoloModelRealModel, InferRejectsWrongInputCount) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path;
  }

  YoloModel model = MakeModel(model_path);

  const auto result = model.infer({});
  EXPECT_FALSE(result.has_value());
}

TEST(YoloModelRealModel, InferRejectsWrongInputShape) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path;
  }

  YoloModel model = MakeModel(model_path);

  std::vector<Tensor> input;
  input.emplace_back("images", std::vector<int64_t>{1, 3, 224, 224}, TensorDataType::kFloat32);

  const auto result = model.infer(input);
  EXPECT_FALSE(result.has_value());
}

// Higher-level counterpart to OnnxRuntimeBackendEp in
// OnnxRuntimeBackend_test.cpp - same GTEST_SKIP-if-unavailable rationale
// applies (see the comment there), just exercised through YoloModel/
// IInferenceModel instead of the backend directly.
class YoloModelEp : public ::testing::TestWithParam<InferenceBackendType> {};

TEST_P(YoloModelEp, WarmupAndInferSucceed) {
  const std::filesystem::path model_path = FixtureModelPath();
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "fixture not found at " << model_path
                 << " - see OnnxRuntimeBackend_test.cpp for how to generate it";
  }

  std::unique_ptr<YoloModel> model;
  try {
    model = std::make_unique<YoloModel>(std::make_unique<OnnxRuntimeBackend>(model_path.string(), GetParam()));
  } catch (const std::runtime_error &ex) {
    GTEST_SKIP() << "execution provider unavailable on this machine: " << ex.what();
  }

  EXPECT_NO_THROW(model->warmup(2));

  std::vector<Tensor> input;
  input.emplace_back("images", std::vector<int64_t>{1, 3, 608, 608}, TensorDataType::kFloat32);

  const auto result = model->infer(input);
  ASSERT_TRUE(result.has_value()) << result.error();
  ASSERT_EQ(result->size(), 2U);
  EXPECT_EQ((*result)[0].shape(), (std::vector<int64_t>{1, 300, 38}));
  EXPECT_EQ((*result)[1].shape(), (std::vector<int64_t>{1, 32, 152, 152}));
}

INSTANTIATE_TEST_SUITE_P(
    ExecutionProviders, YoloModelEp,
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
