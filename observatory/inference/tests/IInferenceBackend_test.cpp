#include "observatory/inference/IInferenceBackend.hpp"

#include <gtest/gtest.h>

#include <type_traits>
#include <vector>

#include "observatory/inference/Tensor.hpp"

namespace observatory::inference {
namespace {

class StubInferenceBackend final : public IInferenceBackend {
 public:
  void run(const std::vector<Tensor>& input, std::vector<Tensor>& output) override {
    ++run_calls_;
    output = input;
  }

  std::vector<Tensor> getInputTensorsDefault() override {
    return {Tensor("x", {4}, TensorDataType::kFloat32)};
  }

  std::vector<Tensor> getOutputTensorsDefault() override {
    return {Tensor("x", {4}, TensorDataType::kFloat32)};
  }

  std::unordered_map<std::string, std::string> getMetadata() override { return {{"key", "value"}}; }

  int run_calls_ = 0;
};

TEST(IInferenceBackend, IsAbstract) {
  static_assert(std::is_abstract_v<IInferenceBackend>,
                "IInferenceBackend must stay a pure abstract interface");
}

TEST(IInferenceBackend, StubSatisfiesContract) {
  StubInferenceBackend backend;
  const std::vector<Tensor> input{Tensor("x", {4}, TensorDataType::kFloat32)};
  std::vector<Tensor> output;
  backend.run(input, output);
  EXPECT_EQ(backend.run_calls_, 1);
  EXPECT_EQ(output.size(), 1U);
}

TEST(IInferenceBackend, GetMetadataReturnsBackendRawMap) {
  StubInferenceBackend backend;
  const auto metadata = backend.getMetadata();
  ASSERT_EQ(metadata.count("key"), 1U);
  EXPECT_EQ(metadata.at("key"), "value");
}

}  // namespace
}  // namespace observatory::inference
