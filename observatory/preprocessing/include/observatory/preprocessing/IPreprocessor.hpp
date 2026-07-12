#pragma once

namespace observatory::preprocessing {

// Model-independent preprocessing strategy: resize, normalize, padding,
// letterbox, color conversion, undistortion, ROI extraction. See
// observatory/CLAUDE.md. The method contract isn't decided yet, so this
// stays a minimal shell rather than inventing signatures the spec doesn't
// give.
class IPreprocessor {
 public:
  virtual ~IPreprocessor() = default;
};

}  // namespace observatory::preprocessing
