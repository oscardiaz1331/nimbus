#include "observatory/inference/IInferenceModel.hpp"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

namespace observatory::inference {
namespace {

class StubInferenceModel final : public IInferenceModel {
 public:
  StubInferenceModel(const std::string& model_path) {
    last_path_ = model_path;
  }
  void warmup(int iterations) override { warmup_calls_ += iterations; }
  std::expected<std::vector<Tensor>, std::string> infer(const std::vector<Tensor> &input) override {
    ++infer_calls_;
    return input;
  }
  const ModelMetadata& metadata() const override { return metadata_; }

  std::string last_path_;
  int warmup_calls_ = 0;
  int infer_calls_ = 0;
  ModelMetadata metadata_;
};

TEST(IInferenceModel, IsAbstract) {
  static_assert(std::is_abstract_v<IInferenceModel>,
                "IInferenceModel must stay a pure abstract interface");
}

TEST(IInferenceModel, StubSatisfiesFullContract) {
  StubInferenceModel model{"model.onnx"};
  EXPECT_EQ(model.last_path_, "model.onnx");
  model.warmup(3);
  EXPECT_EQ(model.warmup_calls_, 3);
  model.infer({});
  EXPECT_EQ(model.infer_calls_, 1);
  EXPECT_EQ(&model.metadata(), &model.metadata_);
}

}  // namespace
}  // namespace observatory::inference
