#include "observatory/preprocessing/IPreprocessor.hpp"

#include <gtest/gtest.h>

#include <type_traits>

namespace observatory::preprocessing {
namespace {

class StubPreprocessor final : public IPreprocessor {
 public:
  std::expected<std::pair<std::vector<inference::Tensor>, std::vector<PreprocessContext>>, std::string>
  process(const std::vector<cv::Mat> &images) override {
    if (auto ok = validateInputs(images); !ok) return std::unexpected(ok.error());
    return std::make_pair(std::vector<inference::Tensor>{}, std::vector<PreprocessContext>{});
  }
};

TEST(IPreprocessor, HasVirtualDestructor) {
  static_assert(std::has_virtual_destructor_v<IPreprocessor>,
                "IPreprocessor must have a virtual destructor");
}

TEST(IPreprocessor, IsAbstract) {
  // process() is pure virtual: model-specific strategies must supply it.
  static_assert(std::is_abstract_v<IPreprocessor>,
                "IPreprocessor must stay abstract now that process() is pure virtual");
}

TEST(IPreprocessor, ConcreteSubclassOverridingProcessIsInstantiable) {
  StubPreprocessor preprocessor;
  (void)preprocessor;
  SUCCEED();
}

TEST(IPreprocessor, ValidateInputsRejectsEmptyBatch) {
  StubPreprocessor preprocessor;
  auto result = preprocessor.process({});
  EXPECT_FALSE(result.has_value());
}

TEST(IPreprocessor, ValidateInputsRejectsNonColorImage) {
  StubPreprocessor preprocessor;
  cv::Mat grayscale(4, 4, CV_8UC1, cv::Scalar(0));
  auto result = preprocessor.process({grayscale});
  EXPECT_FALSE(result.has_value());
}

TEST(IPreprocessor, ValidateInputsAcceptsWellFormedBatch) {
  StubPreprocessor preprocessor;
  cv::Mat image(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
  auto result = preprocessor.process({image});
  EXPECT_TRUE(result.has_value());
}

}  // namespace
}  // namespace observatory::preprocessing
