#include "observatory/configuration/ConfigLoader.hpp"

#include <format>

#include <opencv5/opencv2/core/persistence.hpp>

namespace observatory::configuration
{

    std::expected<Config, std::string> loadConfig(const std::filesystem::path &path)
    {
        cv::FileStorage fs(path.string(), cv::FileStorage::READ);
        if (!fs.isOpened())
            return std::unexpected(std::format("Config: could not open \"{}\" as a YAML/XML/JSON config file.", path.string()));

        const cv::FileNode model_path_node = fs["model_path"];
        if (model_path_node.isNone())
            return std::unexpected(std::format("Config: \"{}\" is missing required key \"model_path\".", path.string()));

        Config config;
        config.model_path = static_cast<std::string>(model_path_node);

        if (const cv::FileNode node = fs["backend"]; !node.isNone())
            config.backend = static_cast<std::string>(node);
        if (const cv::FileNode node = fs["conf_threshold"]; !node.isNone())
            config.conf_threshold = static_cast<float>(node);
        if (const cv::FileNode node = fs["nms_threshold"]; !node.isNone())
            config.nms_threshold = static_cast<float>(node);

        return config;
    }

} // namespace observatory::configuration
