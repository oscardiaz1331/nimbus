#pragma once

namespace observatory::inference {

// Strategy interface for segmentation backends (YOLO26, RF-DETR, FastSAM,
// SAM2). See observatory/CLAUDE.md. The method contract isn't decided yet,
// so this stays a minimal shell rather than inventing signatures the spec
// doesn't give (same YAGNI stance as the IExporter "ponytail" note there).
class ISegmenter {
 public:
  virtual ~ISegmenter() = default;
};

}  // namespace observatory::inference
