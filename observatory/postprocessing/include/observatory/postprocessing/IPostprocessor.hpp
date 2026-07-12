#pragma once

namespace observatory::postprocessing {

// Postprocessing strategy: NMS, mask decoding, polygon extraction,
// cloud/sky % stats, connected components, confidence filtering, temporal
// smoothing. See observatory/CLAUDE.md. The method contract isn't decided
// yet, so this stays a minimal shell rather than inventing signatures the
// spec doesn't give.
class IPostprocessor {
 public:
  virtual ~IPostprocessor() = default;
};

}  // namespace observatory::postprocessing
