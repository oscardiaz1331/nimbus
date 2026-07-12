#pragma once

#include <string>

namespace observatory::inference {

// Strategy interface for a loadable, runnable inference model
// (YOLOSegmentation, RF-DETR, SAM...). See observatory/CLAUDE.md.
class IInferenceModel {
 public:
  virtual ~IInferenceModel() = default;

  // Loads model weights/graph from the given path. Returns true on success.
  virtual bool load(const std::string& model_path) = 0;

  // Runs one or more dummy inference passes to warm up the backend
  // (allocator caches, kernel autotuning, ...) before real traffic.
  virtual void warmup(int iterations) = 0;

  // TODO(design): takes/returns real Frame/Detection types once those are
  // defined; left parameterless here deliberately rather than guessing.
  virtual void infer() = 0;

  // Metadata embedded in the .onnx file by
  // academy/utils/optimizers/metadata.py (framework, task, input size,
  // class names, ...).
  // TODO(design): return a typed struct once that schema is finalized; a
  // string placeholder avoids inventing it here.
  virtual std::string metadata() const = 0;
};

}  // namespace observatory::inference
