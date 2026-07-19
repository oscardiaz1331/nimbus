#pragma once

#include <vector>
#include <expected>

#include "observatory/inference/IInferenceBackend.hpp"
#include "observatory/inference/IInferenceModel.hpp"
#include "observatory/inference/Tensor.hpp"

namespace observatory::inference {

// Derived from the loaded model itself, not user config - see the
// IInferenceModel::metadata() doc comment for what belongs here vs. in a
// preprocessor/postprocessor config.
struct YoloModelMetadata : ModelMetadata {
  // Network input side length (YOLO exports are square, e.g. 640), read
  // straight off the backend's declared input tensor shape. Feeds
  // YoloSegPreprocessorConfig::target_size / YoloSegPostprocessorConfig::
  // image_size.
  int input_size = 0;
};

class YoloModel final : public IInferenceModel
{
    public:
    YoloModel(const std::string &model_path, const InferenceBackendType ep_type);
    ~YoloModel() = default;

    void warmup(int iterations) override;

    std::expected<std::vector<Tensor>, std::string> infer(const std::vector<Tensor>& input_tensors) override;

    const YoloModelMetadata &metadata() const override;

    private:
    std::vector<Tensor> default_input_, default_output_;
    YoloModelMetadata metadata_;
};

} // namespace observatory::inference