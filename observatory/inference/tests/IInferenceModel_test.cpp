#include "observatory/inference/IInferenceModel.hpp"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

namespace observatory::inference {
namespace {

class StubInferenceModel final : public IInferenceModel {
 public:
  bool load(const std::string& model_path) override {
    last_path_ = model_path;
    return true;
  }
  void warmup(int iterations) override { warmup_calls_ += iterations; }
  void infer() override { ++infer_calls_; }
  std::string metadata() const override { return "stub"; }

  std::string last_path_;
  int warmup_calls_ = 0;
  int infer_calls_ = 0;
};

TEST(IInferenceModel, IsAbstract) {
  static_assert(std::is_abstract_v<IInferenceModel>,
                "IInferenceModel must stay a pure abstract interface");
}

TEST(IInferenceModel, StubSatisfiesFullContract) {
  StubInferenceModel model;
  EXPECT_TRUE(model.load("model.onnx"));
  EXPECT_EQ(model.last_path_, "model.onnx");
  model.warmup(3);
  EXPECT_EQ(model.warmup_calls_, 3);
  model.infer();
  EXPECT_EQ(model.infer_calls_, 1);
  EXPECT_EQ(model.metadata(), "stub");
}

}  // namespace
}  // namespace observatory::inference
