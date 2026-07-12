#pragma once

namespace observatory::tracking {

// Temporal tracking / optical flow strategy — future, not implemented yet
// per observatory/CLAUDE.md's "Open / not decided" section. Kept as a
// minimal shell so the module/build wiring exists ahead of the real
// design work.
class ITracker {
 public:
  virtual ~ITracker() = default;
};

}  // namespace observatory::tracking
