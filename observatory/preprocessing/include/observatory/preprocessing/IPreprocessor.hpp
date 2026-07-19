#pragma once

#include <expected>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include <opencv5/opencv2/core.hpp>

#include "observatory/inference/Tensor.hpp"

namespace observatory::preprocessing
{
    struct PreprocessContext
    {
        cv::Size original_size;
        float scale;      // uniform scale applied to map back to original_size
        cv::Point2f pad;  // (left, top) padding in pixels; {0,0} if the preprocessor doesn't pad
    };

    class IPreprocessor
    {
    public:
        virtual ~IPreprocessor() = default;

        virtual std::expected<std::pair<std::vector<inference::Tensor>, std::vector<PreprocessContext>>, std::string>
            process(const std::vector<cv::Mat> &images) = 0;

    protected:
        // Identical for every architecture: no per-model knowledge needed here.
        static inline std::expected<void, std::string> validateInputs(const std::vector<cv::Mat> &images)
        {
            if (images.empty())
                return std::unexpected("Empty vector images to preprocessor.");
            for (const auto &image : images)
            {
                if (image.empty())
                    return std::unexpected("Empty image in preprocessor input.");
                if (image.channels() != 3 || image.depth() != CV_8U)
                    return std::unexpected(std::format(
                        "Expected 8-bit 3-channel BGR image, got {} channels, depth {}.",
                        image.channels(), image.depth()));
            }
            return {};
        }
    };
} // namespace observatory::preprocessing