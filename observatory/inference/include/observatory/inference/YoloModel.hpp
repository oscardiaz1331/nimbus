#pragma once

#include <vector>
#include <expected>

#include "observatory/inference/IInferenceBackend.hpp"
#include "observatory/inference/IInferenceModel.hpp"
#include "observatory/inference/Tensor.hpp"

namespace observatory::inference {

class YoloModel final : public IInferenceModel
{
    public:
    YoloModel(const std::string &model_path, const InferenceBackendType ep_type);
    ~YoloModel() = default;

    void warmup(int iterations) override;

    std::expected<std::vector<Tensor>, std::string> infer(const std::vector<Tensor>& input_tensors) override;

    std::string metadata() const override;

    private:
    std::vector<Tensor> default_input_, default_output_;
};

} // namespace observatory::inference