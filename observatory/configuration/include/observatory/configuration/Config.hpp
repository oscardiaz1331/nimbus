#pragma once

#include <string>

namespace observatory::configuration
{

    // Top-level, config-driven settings for the observatory pipeline. 
    // Starts minimal - just what the pipeline needs today - and grows as camera/
    // backend/threshold selection actually needs to be config-driven too.
    struct Config
    {
        // Path to the .onnx model file to load (see inference/IInferenceModel).
        std::string model_path;

        // Which IInferenceBackend to run on: "onnx" (auto EP selection,
        // default), "onnx-cpu", "onnx-cuda", "onnx-tensorrt", or
        // "onnx-openvino" pin a specific ONNX Runtime execution provider.
        // The "onnx-" prefix leaves "tensorrt"/"openvino" free for future
        // backends that talk to those runtimes directly instead of through
        // ONNX Runtime's EP mechanism (see InferenceBackendType::kTensorRT/
        // kOpenVINO).
        std::string backend = "onnx";

        // Score/IoU thresholds for postprocessing (see
        // YoloSegPostprocessorConfig). Always user config - unlike
        // num_classes/nms_embedded, these aren't something the model itself
        // can tell us (see PipelineFactory.cpp).
        float conf_threshold = 0.25f;
        float nms_threshold = 0.45f;
    };

} // namespace observatory::configuration
