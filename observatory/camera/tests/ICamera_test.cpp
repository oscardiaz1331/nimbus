#include "observatory/camera/ICamera.hpp"

#include <gtest/gtest.h>

#include <type_traits>

namespace observatory::camera {
namespace {

class StubCamera final : public ICamera {
 public:
  std::expected<cv::Mat, std::string> trigger() override {
    return cv::Mat(1, 1, CV_8UC3, cv::Scalar(0, 0, 0));
  }
};

TEST(ICamera, HasVirtualDestructor) {
  static_assert(std::has_virtual_destructor_v<ICamera>,
                "ICamera must have a virtual destructor");
}

TEST(ICamera, IsAbstract) {
  // trigger() is pure virtual: concrete strategies must supply it.
  static_assert(std::is_abstract_v<ICamera>,
                "ICamera must stay abstract now that trigger() is pure virtual");
}

TEST(ICamera, ConcreteSubclassIsInstantiable) {
  StubCamera camera;
  ICamera& interface = camera;
  auto result = interface.trigger();
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->empty());
}

}  // namespace
}  // namespace observatory::camera
