#pragma once

#include "observatory/preprocessing/IPreprocessor.hpp"

namespace observatory::preprocessing
{

    struct YoloSegPreprocessorConfig
    {
        int stride = 32;;
        int target_size;
    };

    class YoloSegPreprocessor final : public IPreprocessor
    {
    public:
        // target_size / stride should come from the exported ONNX metadata
        // (imgsz, stride), not be hardcoded by the caller.
        explicit YoloSegPreprocessor(const YoloSegPreprocessorConfig& config)
            : target_size_(config.target_size), stride_(config.stride) {}

        std::expected<std::pair<std::vector<inference::Tensor>, std::vector<PreprocessContext>>, std::string>
        process(const std::vector<cv::Mat> &images) override;

    private:
        cv::Mat letterbox(const cv::Mat &image, PreprocessContext &ctx) const;

        int target_size_;
        int stride_;
        static constexpr uchar kPadValue = 114;
    };

} // namespace observatory::preprocessing