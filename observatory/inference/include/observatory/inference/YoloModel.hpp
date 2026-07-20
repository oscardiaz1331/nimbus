#pragma once

#include <vector>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

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

  // The rest comes from the .onnx's custom metadata (see
  // academy/utils/optimizers/metadata.py), which not every model file has
  // (older exports, or one produced before that pipeline stage existed) -
  // nullopt means "not present", not "false"/"zero", so a configuration/
  // factory can tell "derive from the model" apart from "fall back to
  // user config" instead of silently trusting a guessed default.
  std::optional<int> num_classes;
  std::optional<bool> nms_embedded;
};

class YoloModel final : public IInferenceModel
{
    public:
    // Takes ownership of an already-built backend rather than constructing
    // one itself: which concrete IInferenceBackend to build (and from what
    // model path/execution provider) is a decision that has to happen
    // *before* we even know this is a YOLO model - a configuration/ factory
    // builds the backend first, reads its raw getMetadata()["framework"] to
    // decide YoloModel is the right wrapper, then hands that same backend
    // here instead of opening the model file a second time.
    explicit YoloModel(std::unique_ptr<IInferenceBackend> backend);
    ~YoloModel() = default;

    void warmup(int iterations) override;

    std::expected<std::vector<Tensor>, std::string> infer(const std::vector<Tensor>& input_tensors) override;

    const YoloModelMetadata &metadata() const override;

    // Parses the handful of keys YoloModelMetadata cares about out of a
    // backend's raw getMetadata() map (see IInferenceBackend::getMetadata).
    // Public and static so it's testable directly against a hand-built map,
    // without needing a real backend/model file. A key that's missing or
    // fails to parse is left as std::nullopt rather than defaulted, since a
    // malformed value is exactly as "not there" as a missing one here.
    static YoloModelMetadata ParseMetadata(const std::unordered_map<std::string, std::string> &raw);

    private:
    std::vector<Tensor> default_input_, default_output_;
    YoloModelMetadata metadata_;
};

} // namespace observatory::inference