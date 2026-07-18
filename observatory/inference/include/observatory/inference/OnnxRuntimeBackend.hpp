#pragma once

#include <memory>
#include <string>
#include <vector>

#include "observatory/inference/IInferenceBackend.hpp"
#include "observatory/inference/Tensor.hpp"

namespace observatory::inference
{

  /// @brief ONNX Runtime implementation of IInferenceBackend.
  ///
  /// Loads a model at construction (RAII: an invalid path or backend type
  /// throws). `ep_type` selects the execution provider: kOnnxRuntimeBest lets
  /// ONNX Runtime auto-select the best-performing registered EP, while
  /// kOnnxRuntimeCUDA / kOnnxRuntimeOpenVINO / kOnnxRuntimeTensorRT pin the
  /// session to that specific provider.
  ///
  /// Uses the pImpl idiom: every ONNX Runtime type lives in Impl, defined
  /// only in OnnxRuntimeBackend.cpp. This header intentionally does not
  /// include onnxruntime_cxx_api.h, so callers that only need run() don't
  /// pay ONNX Runtime's compile-time cost transitively.
  class OnnxRuntimeBackend final : public IInferenceBackend
  {
  public:
    /// @brief Loads `model_path` and prepares a session for `ep_type`.
    /// @param[in] model_path Path to a .onnx model file.
    /// @param[in] ep_type Execution provider to run the model on.
    /// @throws std::runtime_error if `model_path` is not a valid .onnx file,
    ///   `ep_type` is not an ONNX Runtime backend type, or session creation
    ///   fails (e.g. no device found for the requested execution provider).
    explicit OnnxRuntimeBackend(const std::string &model_path, const InferenceBackendType ep_type = InferenceBackendType::kOnnxRuntimeBest);

    /// @brief Defined out-of-line in the .cpp: with the pImpl idiom, Impl
    ///   must be a complete type wherever std::unique_ptr<Impl> is
    ///   destroyed, which is not possible in this header.
    ~OnnxRuntimeBackend() override;

    /// @brief Runs a forward pass on `input`, writing results into `output`.
    /// @details Kept lowercase to match the IInferenceBackend::run() override
    ///   it implements - see observatory/inference/IInferenceBackend.hpp.
    /// @param[in] input Preprocessed input tensors, one per model input.
    /// @param[in,out] output Pre-shaped output tensors; filled in place.
    void run(const std::vector<Tensor> &input, std::vector<Tensor> &output) override;

    /// @brief Builds one Tensor per model input, named and shaped from the
    ///   session's declared input metadata (dynamic dimensions, if any, are
    ///   pinned to 1) and pre-allocated to match.
    std::vector<Tensor> getInputTensorsDefault() override;

    /// @copydoc getInputTensorsDefault
    std::vector<Tensor> getOutputTensorsDefault() override;

  private:
    struct Impl;
    std::unique_ptr<Impl> p_impl_;
  };

} // namespace observatory::inference
