#include "observatory/postprocessing/IPostprocessor.hpp"

#include <gtest/gtest.h>

#include <type_traits>

namespace observatory::postprocessing {
namespace {

class StubPostprocessor final : public IPostprocessor {
 public:
  std::expected<std::vector<std::vector<Detection>>, std::string> process(
      const std::vector<inference::Tensor>& outputs) override {
    (void)outputs;
    return std::vector<std::vector<Detection>>{};
  }
};

TEST(IPostprocessor, HasVirtualDestructor) {
  static_assert(std::has_virtual_destructor_v<IPostprocessor>,
                "IPostprocessor must have a virtual destructor");
}

TEST(IPostprocessor, IsAbstract) {
  // process() is pure virtual: the interface itself must not be
  // instantiable, only concrete strategies like StubPostprocessor.
  static_assert(std::is_abstract_v<IPostprocessor>,
                "IPostprocessor must be abstract");
}

TEST(IPostprocessor, ConcreteSubclassIsInstantiable) {
  StubPostprocessor postprocessor;
  IPostprocessor& interface = postprocessor;
  auto result = interface.process({});
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value().empty());
}

}  // namespace
}  // namespace observatory::postprocessing
