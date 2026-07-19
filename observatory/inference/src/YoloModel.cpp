#include "observatory/inference/YoloModel.hpp"
#include "observatory/inference/OnnxRuntimeBackend.hpp" // todo change the ORT construction to a proper IInference-Backend builder.
#include <format>

namespace observatory::inference
{
    // todo change the ORT construction to a proper IInference-Backend builder.
    YoloModel::YoloModel(const std::string &model_path, const InferenceBackendType ep_type)
    {
        backend_ = std::make_unique<OnnxRuntimeBackend>(model_path, ep_type);
        default_input_ = backend_->getInputTensorsDefault();
        default_output_ = backend_->getOutputTensorsDefault();

        // Input shape is (batch, channels, H, W); the last dim (W) is the
        // network's input side length (YOLO exports are square).
        metadata_.input_size = static_cast<int>(default_input_.front().shape().back());
    }

    void YoloModel::warmup(int iterations)
    {
        std::vector<Tensor> input = default_input_;
        std::vector<Tensor> output = default_output_;
        for (int i = 0; i < iterations; ++i)
            backend_->run(input, output);
    }

    std::expected<std::vector<Tensor>, std::string> YoloModel::infer(const std::vector<Tensor> &input_tensors)
    {
        std::vector<Tensor> output = default_output_;
        if (input_tensors.size() != default_input_.size() || output.size() != 2)
        {
            return std::unexpected(std::format("Incorrect size input: given {}, expected {}; or output: given {}, expected {}", input_tensors.size(), default_input_.size(), 2, output.size()));
        }
        if(input_tensors[0].shape() != default_input_[0].shape())
        {
            return std::unexpected("Incorrect tensor input shape");
        }
        backend_->run(input_tensors, output);
        return output;
    }

    const YoloModelMetadata &YoloModel::metadata() const
    {
        return metadata_;
    }

} // namespace observatory::inference