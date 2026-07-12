#include "observatory/inference/ISegmenter.hpp"

#include <gtest/gtest.h>

#include <type_traits>

namespace observatory::inference {
namespace {

class StubSegmenter final : public ISegmenter {};

TEST(ISegmenter, HasVirtualDestructor) {
  // Deliberately not is_abstract_v: this is a minimal shell (no pure
  // virtual methods yet, per CLAUDE.md's undecided contract), so it's
  // concrete on purpose — just polymorphic and safely destructible
  // through a base pointer.
  static_assert(std::has_virtual_destructor_v<ISegmenter>,
                "ISegmenter must have a virtual destructor");
}

TEST(ISegmenter, ConcreteSubclassIsInstantiable) {
  StubSegmenter segmenter;
  (void)segmenter;
  SUCCEED();
}

}  // namespace
}  // namespace observatory::inference
