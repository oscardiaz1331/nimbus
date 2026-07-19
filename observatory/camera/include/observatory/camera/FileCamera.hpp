#pragma once

#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>

#include "observatory/camera/ICamera.hpp"

namespace observatory::camera {

struct FileCameraConfig {
  // Frames returned in order, one per trigger() call.
  std::vector<std::filesystem::path> image_paths;
  // Wrap back to image_paths.front() after the last frame instead of
  // erroring out, so a short fixture sequence can still drive a
  // long-running test or a remote demo loop.
  bool loop = true;
};

// ICamera strategy that replays image files from disk instead of reading a
// real sensor. Stands in for eyes/ hardware until it's available, or for
// running the pipeline fully offline/remote (see observatory/CLAUDE.md).
class FileCamera final : public ICamera {
 public:
  explicit FileCamera(FileCameraConfig config) : config_(std::move(config)) {}

  std::expected<cv::Mat, std::string> trigger() override;

 private:
  FileCameraConfig config_;
  std::size_t next_index_ = 0;
};

}  // namespace observatory::camera
