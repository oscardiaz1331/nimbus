#pragma once

#include <expected>
#include <string>

#include <opencv5/opencv2/core.hpp>

namespace observatory::camera {

// Strategy interface for a triggered frame source. Real hardware capture
// lives in eyes/ (see observatory/CLAUDE.md) — this repo only ever consumes
// it through this interface, so implementations range from a future
// eyes/ bridge to FileCamera, which replays images from disk when there's
// no hardware attached or the pipeline is being exercised remotely.
class ICamera {
 public:
  virtual ~ICamera() = default;

  // Captures one frame in response to an acquisition trigger. What
  // "trigger" means is implementation-defined (a hardware signal, the next
  // file in a sequence, ...); callers just get a frame back or an error.
  virtual std::expected<cv::Mat, std::string> trigger() = 0;
};

}  // namespace observatory::camera
