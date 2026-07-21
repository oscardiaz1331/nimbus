#include "observatory/configuration/PipelineFactory.hpp"

#include <format>
#include <stdexcept>
#include <utility>

#include "observatory/inference/OnnxRuntimeBackend.hpp"
#include "observatory/inference/OpenCVBackend.hpp"
#include "observatory/inference/YoloModel.hpp"
#include "observatory/postprocessing/YoloSegPostprocessor.hpp"
#include "observatory/preprocessing/YoloSegPreprocessor.hpp"

namespace observatory::configuration
{
    namespace
    {
        std::expected<inference::InferenceBackendType, std::string> ParseBackend(const std::string &backend)
        {
            if (backend == "onnx")
                return inference::InferenceBackendType::kOnnxRuntimeBest;
            if (backend == "onnx-cpu")
                return inference::InferenceBackendType::kOnnxRuntimeCPU;
            if (backend == "onnx-cuda")
                return inference::InferenceBackendType::kOnnxRuntimeCUDA;
            if (backend == "onnx-tensorrt")
                return inference::InferenceBackendType::kOnnxRuntimeTensorRT;
            if (backend == "onnx-openvino")
                return inference::InferenceBackendType::kOnnxRuntimeOpenVINO;
            if (backend == "opencv")
                return inference::InferenceBackendType::kOpenCV;
            return std::unexpected(std::format(
                "PipelineFactory: unknown backend \"{}\" (expected one of onnx/onnx-cpu/onnx-cuda/onnx-tensorrt/onnx-openvino/opencv).", backend));
        }

        // Builds the YOLO triplet once ResolveFramework() has already said
        // "yolo". Takes ownership of `backend` - YoloModel wraps it directly
        // (dependency injection, see its constructor doc comment) rather
        // than opening the model file a second time.
        std::expected<Pipeline, std::string> BuildYoloPipeline(std::unique_ptr<inference::IInferenceBackend> backend,
                                                               const Config &config)
        {
            auto model = std::make_unique<inference::YoloModel>(std::move(backend));

            const inference::YoloModelMetadata &metadata = model->metadata();
            if (!metadata.num_classes.has_value() || !metadata.nms_embedded.has_value())
                return std::unexpected(std::format(
                    "PipelineFactory: \"{}\" is missing num_classes/nms_embedded metadata - re-export it with "
                    "academy's current optimize.py pipeline (see academy/utils/optimizers/metadata.py).",
                    config.model_path));

            auto preprocessor = std::make_unique<preprocessing::YoloSegPreprocessor>(
                preprocessing::YoloSegPreprocessorConfig{.target_size = metadata.input_size});

            auto postprocessor = std::make_unique<postprocessing::YoloSegPostprocessor>(postprocessing::YoloSegPostprocessorConfig{
                .num_classes = *metadata.num_classes,
                .nms_embedded = *metadata.nms_embedded,
                .threshold = config.conf_threshold,
                .image_size = static_cast<size_t>(metadata.input_size),
                .nms_threshold = config.nms_threshold,
            });

            return Pipeline{
                .model = std::move(model),
                .preprocessor = std::move(preprocessor),
                .postprocessor = std::move(postprocessor),
            };
        }
    } // namespace

    std::expected<std::string, std::string> ResolveFramework(const std::unordered_map<std::string, std::string> &raw_metadata)
    {
        const auto it = raw_metadata.find("framework");
        if (it == raw_metadata.end())
            return std::unexpected(
                "PipelineFactory: model is missing \"framework\" metadata - re-export it with academy's current "
                "optimize.py pipeline (see academy/utils/optimizers/provenance.py) so the model can identify itself.");
        return it->second;
    }

    std::expected<Pipeline, std::string> buildPipeline(const Config &config)
    {
        const auto backend_type = ParseBackend(config.backend);
        if (!backend_type)
            return std::unexpected(backend_type.error());

        // Built once, as the backend-agnostic interface: which concrete
        // model wrapper owns it is decided below, from the model's own metadata, not hardcoded
        // here - every family exports to plain ONNX and runs through
        // whichever IInferenceBackend was requested, they only disagree on
        // how to interpret the tensors.
        std::unique_ptr<inference::IInferenceBackend> backend;
        switch (*backend_type)
        {
        case inference::InferenceBackendType::kOnnxRuntimeBest:
        case inference::InferenceBackendType::kOnnxRuntimeCPU:
        case inference::InferenceBackendType::kOnnxRuntimeCUDA:
        case inference::InferenceBackendType::kOnnxRuntimeTensorRT:
        case inference::InferenceBackendType::kOnnxRuntimeOpenVINO:
        {
            try
            {
                backend = std::make_unique<inference::OnnxRuntimeBackend>(config.model_path, *backend_type);
            }
            catch (const std::exception &ex)
            {
                return std::unexpected(std::format("PipelineFactory: failed to load model \"{}\": {}", config.model_path, ex.what()));
            }
            break;
        }
        case inference::InferenceBackendType::kOpenCV:
        {
            try
            {
                backend = std::make_unique<inference::OpenCVBackend>(config.model_path);
            }
            catch (const std::exception &ex)
            {
                return std::unexpected(std::format("PipelineFactory: failed to load model \"{}\": {}", config.model_path, ex.what()));
            }
            break;
        }
        case inference::InferenceBackendType::kTensorRT:
        case inference::InferenceBackendType::kOpenVINO:
            // Not ONNX Runtime EPs at all (see IsOnnxRuntimeBackendType() in
            // OnnxRuntimeBackend.cpp) - ParseBackend() can't produce these
            // today (nothing maps to them yet), but handled explicitly
            // rather than left to -Wswitch/a null `backend` so a future
            // native TensorRT/OpenVINO backend has an obvious seam here.
            return std::unexpected(std::format(
                "PipelineFactory: backend \"{}\" is not implemented yet (only the ONNX Runtime backends are).", config.backend));
        }

        const auto framework = ResolveFramework(backend->getMetadata());
        if (!framework)
            return std::unexpected(framework.error());

        if (*framework == "yolo")
            return BuildYoloPipeline(std::move(backend), config);

        return std::unexpected(std::format(
            "PipelineFactory: framework \"{}\" (from \"{}\") is not supported yet",
            *framework, config.model_path));
    }

} // namespace observatory::configuration
