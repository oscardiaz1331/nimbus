#pragma once

#include <memory>
#include <string>
#include <vector>

#include "observatory/inference/IInferenceBackend.hpp"
#include "observatory/inference/Tensor.hpp"

namespace observatory::inference
{

  // ONNX Runtime backend: loads a model at construction (RAII, throws on an
  // invalid path). `ep_type` picks the execution provider: kOnnxRuntime lets
  // ONNX Runtime auto-select the best-performing registered EP, while
  // kTensorRT/kOpenVINO pin the session to that specific provider.
  //
  // pImpl: keeps onnxruntime_cxx_api.h (and everything it pulls in) out of
  // this header, so consumers that only call run() don't pay ONNX Runtime's
  // compile-time cost transitively.
  class OnnxRuntimeBackend final : public IInferenceBackend
  {
  public:
    explicit OnnxRuntimeBackend(const std::string &model_path, InferenceBackendType ep_type = InferenceBackendType::kOnnxRuntimeBest);
    ~OnnxRuntimeBackend() override;

    void run(const std::vector<Tensor> &input, std::vector<Tensor> &output) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> p_impl_;
  };

} // namespace observatory::inference
