#include "observatory/camera/FileCamera.hpp"

#include <format>

#include <opencv2/imgcodecs.hpp>

namespace observatory::camera {

std::expected<cv::Mat, std::string> FileCamera::trigger() {
  if (config_.image_paths.empty())
    return std::unexpected("FileCamera has no configured image paths.");

  if (next_index_ >= config_.image_paths.size()) {
    if (!config_.loop)
      return std::unexpected("FileCamera exhausted its configured image sequence.");
    next_index_ = 0;
  }

  const auto& path = config_.image_paths[next_index_++];
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty())
    return std::unexpected(std::format("FileCamera failed to read image file: {}", path.string()));

  return image;
}

}  // namespace observatory::camera
