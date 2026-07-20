#include "observatory/inference/YoloModel.hpp"
#include <charconv>
#include <format>
#include <utility>

namespace observatory::inference
{
    YoloModel::YoloModel(std::unique_ptr<IInferenceBackend> backend)
    {
        backend_ = std::move(backend);
        default_input_ = backend_->getInputTensorsDefault();
        default_output_ = backend_->getOutputTensorsDefault();

        metadata_ = ParseMetadata(backend_->getMetadata());
        // Input shape is (batch, channels, H, W); the last dim (W) is the
        // network's input side length (YOLO exports are square). Derived
        // straight from the graph itself, not from embedded metadata - always
        // present, unlike num_classes/nms_embedded above.
        metadata_.input_size = static_cast<int>(default_input_.front().shape().back());
    }

    YoloModelMetadata YoloModel::ParseMetadata(const std::unordered_map<std::string, std::string> &raw)
    {
        YoloModelMetadata metadata;

        if (const auto it = raw.find("num_classes"); it != raw.end())
        {
            int value = 0;
            const auto [ptr, ec] = std::from_chars(it->second.data(), it->second.data() + it->second.size(), value);
            if (ec == std::errc{} && ptr == it->second.data() + it->second.size())
                metadata.num_classes = value;
        }

        if (const auto it = raw.find("nms_embedded"); it != raw.end())
        {
            if (it->second == "true")
                metadata.nms_embedded = true;
            else if (it->second == "false")
                metadata.nms_embedded = false;
        }

        return metadata;
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