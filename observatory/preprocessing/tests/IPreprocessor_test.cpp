#include "observatory/preprocessing/IPreprocessor.hpp"

#include <gtest/gtest.h>

#include <type_traits>

namespace observatory::preprocessing {
namespace {

class StubPreprocessor final : public IPreprocessor {};

TEST(IPreprocessor, HasVirtualDestructor) {
  // Deliberately not is_abstract_v: this is a minimal shell (no pure
  // virtual methods yet, per CLAUDE.md's undecided contract), so it's
  // concrete on purpose — just polymorphic and safely destructible
  // through a base pointer.
  static_assert(std::has_virtual_destructor_v<IPreprocessor>,
                "IPreprocessor must have a virtual destructor");
}

TEST(IPreprocessor, ConcreteSubclassIsInstantiable) {
  StubPreprocessor preprocessor;
  (void)preprocessor;
  SUCCEED();
}

}  // namespace
}  // namespace observatory::preprocessing
