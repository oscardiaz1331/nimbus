#include "observatory/postprocessing/IPostprocessor.hpp"

#include <gtest/gtest.h>

#include <type_traits>

namespace observatory::postprocessing {
namespace {

class StubPostprocessor final : public IPostprocessor {};

TEST(IPostprocessor, HasVirtualDestructor) {
  // Deliberately not is_abstract_v: this is a minimal shell (no pure
  // virtual methods yet, per CLAUDE.md's undecided contract), so it's
  // concrete on purpose — just polymorphic and safely destructible
  // through a base pointer.
  static_assert(std::has_virtual_destructor_v<IPostprocessor>,
                "IPostprocessor must have a virtual destructor");
}

TEST(IPostprocessor, ConcreteSubclassIsInstantiable) {
  StubPostprocessor postprocessor;
  (void)postprocessor;
  SUCCEED();
}

}  // namespace
}  // namespace observatory::postprocessing
