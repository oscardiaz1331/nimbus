#pragma once

#include <vector>
#include <opencv5/opencv2/core.hpp>

namespace observatory::postprocessing
{

    struct Detection
    {
        cv::Rect box;
        int class_id = 0;
        float score = 0.0f;
        // Copied out of output_0 rather than kept as a view into it: output_0
        // isn't guaranteed to outlive the Detection this ends up in, and nm
        // is small enough (tens of floats) that the copy is noise next to
        // the GEMM/resize work decodeMasks() does with it.
        std::vector<float> mask_coeffs;

        float cloud_sky_percentage = 0;
        std::vector<cv::Mat> seg_masks;
    };

} // namespace observatory::postprocessing