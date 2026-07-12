#include "observatory/tracking/ITracker.hpp"

#include <gtest/gtest.h>

#include <type_traits>

namespace observatory::tracking {
namespace {

class StubTracker final : public ITracker {};

TEST(ITracker, HasVirtualDestructor) {
  // Deliberately not is_abstract_v: this is a minimal shell (no pure
  // virtual methods yet, per CLAUDE.md's undecided contract), so it's
  // concrete on purpose — just polymorphic and safely destructible
  // through a base pointer.
  static_assert(std::has_virtual_destructor_v<ITracker>,
                "ITracker must have a virtual destructor");
}

TEST(ITracker, ConcreteSubclassIsInstantiable) {
  StubTracker tracker;
  (void)tracker;
  SUCCEED();
}

}  // namespace
}  // namespace observatory::tracking
