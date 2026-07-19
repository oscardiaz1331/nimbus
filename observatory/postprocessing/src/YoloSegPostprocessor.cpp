#include "observatory/postprocessing/YoloSegPostprocessor.hpp"

#include <opencv5/opencv2/imgproc.hpp>

#include <format>
#include <ranges>
#include <span>

namespace observatory::postprocessing
{

    std::expected<std::vector<std::vector<Detection>>, std::string> YoloSegPostprocessor::process(const std::vector<inference::Tensor> &outputs)
    {
        if (outputs.size() != 2)
        {
            return std::unexpected(std::format("Expected 2 output tensors in postprocessor, got {}.", outputs.size()));
        }
        // The YOLO-seg ONNX graph names its outputs "output0" (detections)
        // and "output1" (mask prototypes). Resolve them by name instead of
        // by position, so a backend that reorders outputs can't silently
        // swap them.
        const inference::Tensor *dets = nullptr;
        const inference::Tensor *protos = nullptr;
        for (const inference::Tensor &tensor : outputs)
        {
            if (tensor.name() == "output0")
            {
                dets = &tensor;
            }
            else if (tensor.name() == "output1")
            {
                protos = &tensor;
            }
        }
        if (dets == nullptr || protos == nullptr)
        {
            return std::unexpected("Expected output tensors named \"output0\" (dets) and \"output1\" (protos) in postprocessor.");
        }
        if (dets->shape().size() != 3 || protos->shape().size() != 4)
        {
            return std::unexpected(std::format("Expected dets rank 3 and protos rank 4 in postprocessor, got {} and {}.", dets->shape().size(), protos->shape().size()));
        }
        if (dets->shape()[0] != protos->shape()[0])
        {
            return std::unexpected(std::format("Batch mismatch in postprocessor: dets {} vs protos {}.", dets->shape()[0], protos->shape()[0]));
        }
        std::vector<std::vector<Detection>> detections = (this->*process_dets_)(*dets);
        decodeMasks(detections, *protos);
        return detections;
    }

    std::vector<std::vector<Detection>> YoloSegPostprocessor::processDetsWithNMS(const inference::Tensor &dets)
    {
        const int64_t batch_size = dets.shape()[0];
        const int64_t max_detections = dets.shape()[1];
        const int nm = static_cast<int>(dets.shape()[2]) - 6;

        // dets has no operator[]; as_mdspan<T, Rank>() gives a typed,
        // shape-checked view instead (layout documented in the .hpp).
        const auto dets_view = dets.as_mdspan<float, 3>();
        const cv::Rect2f frame(0.0f, 0.0f, static_cast<float>(image_size_), static_cast<float>(image_size_));

        // shape() dims are always > 0 (enforced by Tensor's constructor),
        // so this is a real invariant, safe to hand to the optimizer.
        [[assume(max_detections > 0)]];

        std::vector<std::vector<Detection>> detections;
        detections.reserve(static_cast<std::size_t>(batch_size));

        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            std::vector<Detection> batch_detections;

            const auto above_threshold = [&](int64_t detection_idx)
            {
                return dets_view[batch_idx, detection_idx, 4] >= threshold_;
            };

            for (const int64_t detection_idx : std::views::iota(int64_t{0}, max_detections) | std::views::filter(above_threshold))
            {
                //@todo check if they are ordered by threshold, so this can be a break
                const cv::Rect2f raw_box(cv::Point2f(dets_view[batch_idx, detection_idx, 0], dets_view[batch_idx, detection_idx, 1]),
                                          cv::Point2f(dets_view[batch_idx, detection_idx, 2], dets_view[batch_idx, detection_idx, 3]));
                const cv::Rect box_i(raw_box & frame);
                if (box_i.width <= 0 || box_i.height <= 0)
                {
                    continue;
                }

                Detection detection;
                detection.box = box_i;
                detection.score = dets_view[batch_idx, detection_idx, 4];
                detection.class_id = static_cast<int>(dets_view[batch_idx, detection_idx, 5]);
                // Coeffs are contiguous in dets_view's last dimension.
                const float *coeff_ptr = &dets_view[batch_idx, detection_idx, 6];
                detection.mask_coeffs.assign(coeff_ptr, coeff_ptr + nm);
                batch_detections.push_back(std::move(detection));
            }

            detections.push_back(std::move(batch_detections));
        }

        return detections;
    }

    std::vector<std::vector<Detection>> YoloSegPostprocessor::processDetsNoNMS([[maybe_unused]] const inference::Tensor &dets)
    {
        //@todo manual per-class NMS over the channel-first (cx,cy,w,h,
        // per-class-scores,nm) layout documented in the .hpp: argmax the nc
        // per-class scores to get class_id+score per anchor, convert
        // cx,cy,w,h -> xyxy, clamp against the frame like processDetsWithNMS
        // does, then suppress by IoU per class. The result should be the same
        // box/class_id/score/mask_coeffs records processDetsWithNMS builds,
        // so decodeMasks() stays oblivious to which export mode ran.
        throw std::logic_error("YoloSegPostprocessor::processDetsNoNMS: not implemented yet");
    }

    void YoloSegPostprocessor::decodeMasks(std::vector<std::vector<Detection>> &detections, const inference::Tensor &protos) const
    {
        const std::vector<int64_t> &protos_shape = protos.shape(); // (batch, nm, mask_h, mask_w)
        const int nm = static_cast<int>(protos_shape[1]);
        const int mask_h = static_cast<int>(protos_shape[2]);
        const int mask_w = static_cast<int>(protos_shape[3]);
        const float mask_scale_x = static_cast<float>(mask_w) / static_cast<float>(image_size_);
        const float mask_scale_y = static_cast<float>(mask_h) / static_cast<float>(image_size_);
        const std::span<const float> protos_data = protos.as_span<float>();

        for (std::size_t batch_idx = 0; batch_idx < detections.size(); ++batch_idx)
        {
            if (detections[batch_idx].empty())
            {
                continue;
            }

            // OpenCV has no const-data Mat view; proto_mat is only ever read
            // through (GEMM), so wrapping the tensor's buffer this way (no
            // copy) is safe despite the const_cast.
            float *proto_batch_ptr = const_cast<float *>(protos_data.data()) +
                                      batch_idx * static_cast<std::size_t>(nm) * static_cast<std::size_t>(mask_h) * static_cast<std::size_t>(mask_w);
            const cv::Mat proto_mat(nm, mask_h * mask_w, CV_32F, proto_batch_ptr);

            for (Detection &detection : detections[batch_idx])
            {
                const cv::Mat coeffs(1, nm, CV_32F, detection.mask_coeffs.data());
                const cv::Mat logits = cv::Mat(coeffs * proto_mat).reshape(1, mask_h); // raw logits, sigmoid not applied yet

                // Crop in mask space before upsampling, so the resize cost
                // is proportional to the box area, not the whole frame.
                const cv::Rect2f box_in_mask_space(static_cast<float>(detection.box.x) * mask_scale_x, static_cast<float>(detection.box.y) * mask_scale_y,
                                                    static_cast<float>(detection.box.width) * mask_scale_x, static_cast<float>(detection.box.height) * mask_scale_y);
                const cv::Rect mask_crop = cv::Rect(box_in_mask_space) & cv::Rect(0, 0, mask_w, mask_h);
                if (mask_crop.empty())
                {
                    continue;
                }
                cv::Mat upsampled;
                cv::resize(logits(mask_crop), upsampled, detection.box.size(), 0, 0, cv::INTER_LINEAR);

                // sigmoid(x) >= 0.5 iff x >= 0: threshold the raw logits
                // directly instead of computing exp() over every pixel.
                cv::Mat detection_mask;
                cv::threshold(upsampled, detection_mask, 0.0, 255, cv::THRESH_BINARY);
                detection_mask.convertTo(detection_mask, CV_8U);

                detection.seg_masks.push_back(std::move(detection_mask));
            }
        }
    }

} // namespace observatory::postprocessing
