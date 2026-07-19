#pragma once

#include "observatory/postprocessing/IPostprocessor.hpp"

namespace observatory::postprocessing
{
    struct YoloSegPostprocessorConfig
    {
        size_t num_classes;
        bool nms_embedded;
        float threshold;
        size_t image_size;
    };

    class YoloSegPostprocessor final : public IPostprocessor
    {
    public:
        explicit YoloSegPostprocessor(const YoloSegPostprocessorConfig &config)
            : process_dets_(config.nms_embedded ? &YoloSegPostprocessor::processDetsWithNMS : &YoloSegPostprocessor::processDetsNoNMS),
              num_classes_(config.num_classes), nms_embedded_(config.nms_embedded), threshold_(config.threshold), image_size_(config.image_size) {}
        std::expected<std::vector<std::vector<Detection>>, std::string> process(const std::vector<inference::Tensor> &outputs) override;

    private:
        // Detections tensor (the ONNX graph exports it as "output0"), export
        // WITHOUT embedded NMS. Shape (batch, 4+nc+nm, num_anchors).
        // CHANNEL-FIRST layout: each channel is contiguous across all anchors
        // (NOT an array-of-structs per anchor). To read channel c of anchor a:
        // data[c * num_anchors + a], not data[a * C + c].
        // Channels, in order: [0:4)          = cx, cy, w, h (center-width-height,
        //                                       ABSOLUTE pixels in network input
        //                                       space, already decoded)
        //                     [4:4+nc)       = per-class score, sigmoid ALREADY
        //                                       applied (independent multi-label,
        //                                       not softmax — more than one class
        //                                       can be high at once)
        //                     [4+nc:4+nc+nm) = nm mask coefficients
        // Requires manual NMS downstream (not deduplicated). Once that NMS
        // is written, this should build the same per-detection records
        // (box, class_id, score, mask_coeffs) as processDetsWithNMS —
        // decodeMasks() below doesn't care which export mode produced them.
        std::vector<std::vector<Detection>> processDetsNoNMS(const inference::Tensor &dets);

        // Detections tensor ("output0"), export WITH embedded NMS (nms=True).
        // Shape (batch, max_det, 6+nm).
        // DETECTION-FIRST layout (opposite of the no-NMS variant): each
        // 6+nm-value detection is contiguous in memory.
        // Fields, in order: x1, y1, x2, y2 (xyxy, absolute pixels — note: xyxy
        //                   here, not cx,cy,w,h like the no-NMS variant),
        //                   score (already a single confidence value post-NMS,
        //                   not a per-class array), class_id (float, cast to
        //                   int), then nm mask coefficients.
        // Already deduplicated (NMS ran inside the ONNX graph). Rows with no
        // real detection come back as zero — check score > 0 (or > threshold)
        // before using a row, padded up to max_det.
        std::vector<std::vector<Detection>> processDetsWithNMS(const inference::Tensor &dets);

        // Fills in each Detection's seg_masks by combining its mask_coeffs
        // (from processDets*) with the prototype planes in `protos` (the
        // ONNX graph exports it as "output1") — shape (batch, nm, mask_h,
        // mask_w), channel-first, IDENTICAL in both export modes.
        // mask_h/mask_w ≈ input_w/4, input_h/4 of the NETWORK INPUT tensor
        // after preprocessing (e.g. 640x640 letterboxed), NOT the original
        // image. Needs both sides together — a prototype plane alone has no
        // meaning, and mask_coeffs alone has nothing to combine with — so
        // this isn't a standalone per-tensor pass: dets and protos were
        // never independent.
        void decodeMasks(std::vector<std::vector<Detection>> &detections, const inference::Tensor &protos) const;

        // Which of the two processDets* variants applies is fixed by
        // config.nms_embedded and never changes for this instance, so it's
        // resolved once here instead of re-branching on nms_embedded_ on
        // every process() call. A plain member-function pointer (not
        // std::function) since there are exactly two fixed candidates: no
        // type erasure, no possible heap allocation.
        using ProcessDetsFn = std::vector<std::vector<Detection>> (YoloSegPostprocessor::*)(const inference::Tensor &);
        ProcessDetsFn process_dets_;

        size_t num_classes_;
        bool nms_embedded_;
        float threshold_;
        size_t image_size_;
    };
} // namespace observatory::postprocessing