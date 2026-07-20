#pragma once

#include <string>
#include <unordered_map>

#include <opencv2/core.hpp>

#include "observatory/inference/Tensor.hpp"

namespace observatory::inference {

enum class InferenceBackendType {
  kOnnxRuntimeBest,
  kOnnxRuntimeCPU,
  kOnnxRuntimeCUDA,
  kOnnxRuntimeOpenVINO,
  kOnnxRuntimeTensorRT,
  kTensorRT,
  kOpenVINO
};

// Strategy interface abstracting the inference backend (ONNX Runtime,
// TensorRT, future OpenVINO). See observatory/CLAUDE.md.
class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  // Runs a forward pass on a single preprocessed input blob, writing the
  // raw output blob.
  virtual void run(const std::vector<Tensor>& input, std::vector<Tensor>& output) = 0;

  // Describes the model's expected inputs/outputs (name, shape, dtype), each
  // pre-allocated to the right size for its shape. Callers (IInferenceModel::
  // warmup(), mainly) use these directly as the input/output vectors passed
  // to run(), instead of hardcoding shapes per model.
  virtual std::vector<Tensor> getInputTensorsDefault() = 0;
  virtual std::vector<Tensor> getOutputTensorsDefault() = 0;

  // Raw custom key/value metadata embedded in the model file by the
  // exporting pipeline (see academy/utils/optimizers/metadata.py) - exactly
  // as written, un-decoded (non-string values there are JSON-encoded
  // strings; it's up to the caller, e.g. YoloModel, to parse the handful of
  // keys it actually cares about). Empty if the backend found nothing (an
  // older export predating this, or a backend that doesn't support reading
  // it at all) - never a hard failure, this is descriptive metadata.
  virtual std::unordered_map<std::string, std::string> getMetadata() = 0;
};

}  // namespace observatory::inference
