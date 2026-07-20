#pragma once

#include <expected>
#include <filesystem>
#include <string>

#include "observatory/configuration/Config.hpp"

namespace observatory::configuration
{

    // Loads a Config from a YAML (or XML/JSON - cv::FileStorage's format is
    // detected from the file extension) config file. cv::FileStorage is used
    // instead of a new YAML/JSON dependency since OpenCV is already a hard
    // dependency of every module here and this is a handful of scalar fields.
    std::expected<Config, std::string> loadConfig(const std::filesystem::path &path);

} // namespace observatory::configuration
