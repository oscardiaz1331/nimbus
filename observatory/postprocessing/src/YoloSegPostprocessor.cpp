#include "observatory/postprocessing/YoloSegPostprocessor.hpp"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

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

        for (int64_t batch_id = 0; batch_id < batch_size; ++batch_id)
        {
            std::vector<Detection> batch_detections;

            const auto above_threshold = [&](int64_t detection_idx)
            {
                return dets_view[batch_id, detection_idx, 4] >= threshold_;
            };
            // Using ::take_while instead of ::filter because the detections are ordered by threshold, so we can break the loop
            for (const int64_t detection_idx : std::views::iota(int64_t{0}, max_detections) | std::views::take_while(above_threshold))
            {
                //@todo check if they are ordered by threshold, so this can be a break
                const cv::Rect2f raw_box(cv::Point2f(dets_view[batch_id, detection_idx, 0], dets_view[batch_id, detection_idx, 1]),
                                         cv::Point2f(dets_view[batch_id, detection_idx, 2], dets_view[batch_id, detection_idx, 3]));
                const cv::Rect box_i(raw_box & frame);
                if (box_i.width <= 0 || box_i.height <= 0)
                {
                    continue;
                }

                Detection detection;
                detection.box = box_i;
                detection.score = dets_view[batch_id, detection_idx, 4];
                detection.class_id = static_cast<int>(dets_view[batch_id, detection_idx, 5]);
                // Coeffs are contiguous in dets_view's last dimension.
                const float *coeff_ptr = &dets_view[batch_id, detection_idx, 6];
                detection.mask_coeffs.assign(coeff_ptr, coeff_ptr + nm);
                batch_detections.push_back(std::move(detection));
            }

            detections.push_back(std::move(batch_detections));
        }

        return detections;
    }

    std::vector<std::vector<Detection>> YoloSegPostprocessor::processDetsNoNMS(const inference::Tensor &dets)
    {
        const int64_t batch_size = dets.shape()[0];
        const int64_t num_channels = dets.shape()[1];
        const int64_t num_anchors = dets.shape()[2];
        const int64_t num_classes = static_cast<int64_t>(num_classes_);
        const int64_t coeff_channel = 4 + num_classes;
        const int nm = static_cast<int>(num_channels - coeff_channel);

        // dets has no operator[]; as_mdspan<T, Rank>() gives a typed,
        // shape-checked view instead (layout documented in the .hpp). Used
        // below for the few per-anchor scalar reads (box, mask coeffs) that
        // don't benefit from a bulk cv::Mat view the way the score block
        // does further down.
        const auto dets_view = dets.as_mdspan<float, 3>();
        const std::span<const float> dets_data = dets.as_span<float>();
        const cv::Rect2f frame(0.0f, 0.0f, static_cast<float>(image_size_), static_cast<float>(image_size_));

        // shape() dims are always > 0 (enforced by Tensor's constructor),
        // so this is a real invariant, safe to hand to the optimizer.
        [[assume(num_anchors > 0)]];

        std::vector<std::vector<Detection>> detections;
        detections.reserve(static_cast<std::size_t>(batch_size));

        for (int64_t batch_id = 0; batch_id < batch_size; ++batch_id)
        {
            // Per-class scores occupy channels [4, 4+num_classes), and in
            // this channel-first layout each channel is one contiguous row
            // of num_anchors floats. That makes the whole score block
            // exactly a (num_classes, num_anchors) matrix in memory already,
            // so it can be wrapped as a cv::Mat with no copy and reduced
            // with two vectorized calls, instead of a per-anchor loop that
            // scans per-class scores with a stride of num_anchors floats
            // between reads.
            float *scores_ptr = const_cast<float *>(dets_data.data()) +
                                 static_cast<std::size_t>(batch_id) * static_cast<std::size_t>(num_channels) * static_cast<std::size_t>(num_anchors) +
                                 4 * static_cast<std::size_t>(num_anchors);
            const cv::Mat scores(num_classes_, static_cast<int>(num_anchors), CV_32F, scores_ptr);

            cv::Mat best_class_mat;
            cv::Mat best_score_mat;
            cv::reduceArgMax(scores, best_class_mat, 0); // per-anchor argmax over classes
            cv::reduce(scores, best_score_mat, 0, cv::REDUCE_MAX); // per-anchor max score
            const int32_t *best_class = best_class_mat.ptr<int32_t>(0);
            const float *best_score = best_score_mat.ptr<float>(0);

            // Candidates before NMS: box/score/class_id per anchor that
            // clears the threshold, plus the anchor index so mask_coeffs
            // only get copied for anchors NMS actually keeps below (most
            // candidates are duplicates of the same object and get
            // suppressed).
            std::vector<cv::Rect> candidate_boxes;
            std::vector<float> candidate_scores;
            std::vector<int> candidate_class_ids;
            std::vector<int64_t> candidate_anchor_ids;

            for (int64_t anchor_id = 0; anchor_id < num_anchors; ++anchor_id)
            {
                if (best_score[anchor_id] < threshold_)
                {
                    continue;
                }
                const cv::Point2f center{dets_view[batch_id, 0, anchor_id], dets_view[batch_id, 1, anchor_id]};
                // Channels 2/3 are full width/height, not half-extents, so
                // the box corners are center +/- half of that, not +/- the
                // raw width/height.
                const cv::Point2f half_size{dets_view[batch_id, 2, anchor_id] * 0.5f, dets_view[batch_id, 3, anchor_id] * 0.5f};
                const cv::Rect box_i(cv::Rect2f(center - half_size, center + half_size) & frame);
                if (box_i.width <= 0 || box_i.height <= 0)
                {
                    continue;
                }
                candidate_boxes.push_back(box_i);
                candidate_scores.push_back(best_score[anchor_id]);
                candidate_class_ids.push_back(best_class[anchor_id]);
                candidate_anchor_ids.push_back(anchor_id);
            }

            // Candidates are already threshold-filtered above, so the score
            // threshold passed here is 0: it only disables NMSBoxesBatched's
            // own (otherwise redundant) pre-filter. Suppression is grouped
            // by class_id, so overlapping boxes of different classes don't
            // suppress each other.
            std::vector<int> kept_indices;
            cv::dnn::NMSBoxesBatched(candidate_boxes, candidate_scores, candidate_class_ids, 0.0f, nms_threshold_, kept_indices);

            std::vector<Detection> batch_detections;
            batch_detections.reserve(kept_indices.size());
            for (const int i : kept_indices)
            {
                const std::size_t candidate_idx = static_cast<std::size_t>(i);
                const int64_t anchor_id = candidate_anchor_ids[candidate_idx];

                Detection detection;
                detection.box = candidate_boxes[candidate_idx];
                detection.score = candidate_scores[candidate_idx];
                detection.class_id = candidate_class_ids[candidate_idx];
                // Channel-first layout: coeffs for one anchor are NOT
                // contiguous (that's only true of processDetsWithNMS's
                // detection-first layout). Each coeff channel is num_anchors
                // floats apart, so it has to be gathered one channel at a
                // time instead of read as a contiguous run.
                detection.mask_coeffs.resize(static_cast<std::size_t>(nm));
                for (int c = 0; c < nm; ++c)
                {
                    detection.mask_coeffs[static_cast<std::size_t>(c)] = dets_view[batch_id, coeff_channel + c, anchor_id];
                }
                batch_detections.push_back(std::move(detection));
            }

            detections.push_back(std::move(batch_detections));
        }

        return detections;
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

        for (std::size_t batch_id = 0; batch_id < detections.size(); ++batch_id)
        {
            if (detections[batch_id].empty())
            {
                continue;
            }

            // OpenCV has no const-data Mat view; proto_mat is only ever read
            // through (GEMM), so wrapping the tensor's buffer this way (no
            // copy) is safe despite the const_cast.
            float *proto_batch_ptr = const_cast<float *>(protos_data.data()) +
                                     batch_id * static_cast<std::size_t>(nm) * static_cast<std::size_t>(mask_h) * static_cast<std::size_t>(mask_w);
            const cv::Mat proto_mat(nm, mask_h * mask_w, CV_32F, proto_batch_ptr);

            //@todo if profiling ever shows this loop as a hot path with high
            // detection counts: stack every detection's mask_coeffs into one
            // (num_dets x nm) Mat and do a single coeffs_batch * proto_mat
            // GEMM instead of one per detection, then batch_logits.row(i) per
            // detection below. Crop/resize/threshold stay per-detection
            // either way (each box is a different size) -- only the GEMM
            // batches. Not worth the bookkeeping below a handful of
            // detections per frame; needs real numbers first.
            for (Detection &detection : detections[batch_id])
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
