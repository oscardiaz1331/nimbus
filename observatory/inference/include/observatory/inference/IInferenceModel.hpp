#pragma once

#include <expected>
#include <string>
#include "observatory/inference/ModelMetadata.hpp"
#include "observatory/inference/OnnxRuntimeBackend.hpp"

namespace observatory::inference {

// Strategy interface for a loadable, runnable inference model
// (YOLO, RF-DETR, SAM...). See observatory/CLAUDE.md.
class IInferenceModel {
 public:
  virtual ~IInferenceModel() = default;

  // Runs one or more dummy inference passes to warm up the backend
  // (allocator caches, kernel autotuning, ...) before real traffic.
  virtual void warmup(int iterations) = 0;

  virtual std::expected<std::vector<Tensor>, std::string> infer(const std::vector<Tensor>& input_tensors) = 0;

  // What this model needs to build a matching IPreprocessor/IPostprocessor
  // (input size, ...). Only fields derivable from the model itself (its
  // declared tensor shapes today; .onnx-embedded custom metadata later, once
  // academy/utils/optimizers/metadata.py is actually wired into the export
  // pipeline) belong here - things that can't be derived (class names, an
  // nms_embedded flag, thresholds) stay user-supplied config instead of being
  // guessed. Concrete models override this with a covariant return type
  // (e.g. YoloModel::metadata() -> const YoloModelMetadata&); see
  // ModelMetadata.hpp.
  virtual const ModelMetadata& metadata() const = 0;

  protected:
  std::unique_ptr<IInferenceBackend> backend_;
};

}  // namespace observatory::inference
