#include "observatory/preprocessing/YoloSegPreprocessor.hpp"

#include <cmath>
#include <cstring>

#include <opencv5/opencv2/dnn.hpp>
#include <opencv5/opencv2/imgproc.hpp>

namespace observatory::preprocessing
{
    cv::Mat YoloSegPreprocessor::letterbox(const cv::Mat &image, PreprocessContext &ctx) const
    {
        const float scale = std::min(static_cast<float>(target_size_) / image.cols,
                                     static_cast<float>(target_size_) / image.rows);
        const int unpadded_w = static_cast<int>(std::round(image.cols * scale));
        const int unpadded_h = static_cast<int>(std::round(image.rows * scale));

        cv::Mat resized;
        cv::resize(image, resized, {unpadded_w, unpadded_h}, 0, 0, cv::INTER_LINEAR);

        const float pad_w = static_cast<float>(target_size_ - unpadded_w) / 2.0f;
        const float pad_h = static_cast<float>(target_size_ - unpadded_h) / 2.0f;
        const int top = static_cast<int>(std::round(pad_h - 0.1f));
        const int bottom = target_size_ - unpadded_h - top;
        const int left = static_cast<int>(std::round(pad_w - 0.1f));
        const int right = target_size_ - unpadded_w - left;

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(kPadValue, kPadValue, kPadValue));

        ctx = PreprocessContext{
            .original_size = image.size(),
            .scale = scale,
            .pad = {static_cast<float>(left), static_cast<float>(top)},
        };
        return padded;
    }

    std::expected<std::pair<std::vector<inference::Tensor>, std::vector<PreprocessContext>>, std::string>
    YoloSegPreprocessor::process(const std::vector<cv::Mat> &images)
    {
        if (auto ok = validateInputs(images); !ok)
            return std::unexpected(ok.error());

        const auto batch_size = static_cast<int64_t>(images.size());
        inference::Tensor tensor("images", {batch_size, 3, target_size_, target_size_},
                                 inference::TensorDataType::kFloat32);
        auto view = tensor.as_mdspan<float, 4>();
        float *dst = view.data_handle(); // assumes layout_right/contiguous allocation true for a freshly-built Tensor.
        const std::size_t image_stride = 3ull * target_size_ * target_size_;

        std::vector<PreprocessContext> contexts;
        contexts.reserve(images.size());

        for (std::size_t n = 0; n < images.size(); ++n)
        {
            PreprocessContext ctx;
            const cv::Mat letterboxed = letterbox(images[n], ctx);
            contexts.push_back(ctx);

            // BGR uint8 -> RGB float32 [0,1], HWC -> CHW, in one OpenCV-optimized
            // call instead of a hand-rolled per-pixel loop.
            const cv::Mat blob = cv::dnn::blobFromImage(
                letterboxed, 1.0 / 255.0, {target_size_, target_size_},
                cv::Scalar(), /*swapRB=*/true, /*crop=*/false, CV_32F);
            std::memcpy(dst + n * image_stride, blob.ptr<float>(), image_stride * sizeof(float));
        }

        std::vector<inference::Tensor> tensors;
        tensors.push_back(std::move(tensor));
        return std::make_pair(std::move(tensors), std::move(contexts));
    }
} // namespace observatory::preprocessing