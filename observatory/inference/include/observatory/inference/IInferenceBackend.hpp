#pragma once

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
};

}  // namespace observatory::inference
