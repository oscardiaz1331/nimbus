#pragma once

#include <expected>
#include <string>
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

  // Metadata embedded in the .onnx file by
  // academy/utils/optimizers/metadata.py (framework, task, input size,
  // class names, ...).
  // TODO(design): return a typed struct once that schema is finalized; a
  // string placeholder avoids inventing it here.
  virtual std::string metadata() const = 0;

  protected: 
  std::unique_ptr<IInferenceBackend> backend_;
};

}  // namespace observatory::inference
