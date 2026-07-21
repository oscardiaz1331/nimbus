#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "observatory/inference/IInferenceBackend.hpp"
#include "observatory/inference/Tensor.hpp"

namespace observatory::inference
{

  /// @brief OpenCV `dnn` implementation of IInferenceBackend.
  ///
  /// Runs the forward pass through cv::dnn::Net (CPU, DNN_BACKEND_OPENCV) -
  /// a dependency-light alternative to OnnxRuntimeBackend for deployments
  /// that don't carry ONNX Runtime's execution-provider libraries (e.g. a
  /// CPU-only edge box). cv::dnn::Net has no API to describe a model's
  /// declared input/output tensors (name/shape/dtype) without already
  /// knowing an input shape, and no API to read .onnx custom metadata_props
  /// at all - so getInputTensorsDefault()/getOutputTensorsDefault()/
  /// getMetadata() are answered by a second, introspection-only
  /// Ort::Session instead (never Run(), so it skips graph optimization and
  /// EP registration - see the .cpp). This also keeps the Tensor
  /// descriptions identical to what OnnxRuntimeBackend reports for the same
  /// model, which matters for the benchmark module's apples-to-apples
  /// backend comparison (see observatory/CLAUDE.md).
  ///
  /// Uses the pImpl idiom: every OpenCV/ONNX Runtime type lives in Impl,
  /// defined only in OpenCVBackend.cpp. This header intentionally does not
  /// include opencv2/dnn.hpp or onnxruntime_cxx_api.h, so callers that only
  /// need run() don't pay either library's compile-time cost transitively.
  class OpenCVBackend final : public IInferenceBackend
  {
  public:
    /// @brief Loads `model_path` on CPU (cv::dnn::DNN_BACKEND_OPENCV /
    ///   DNN_TARGET_CPU) and prepares an ONNX Runtime session used only for
    ///   metadata/shape introspection.
    /// @param[in] model_path Path to a .onnx model file.
    /// @throws std::runtime_error if `model_path` is not a valid .onnx file,
    ///   or either engine fails to load it.
    explicit OpenCVBackend(const std::string &model_path);

    /// @brief Defined out-of-line in the .cpp: with the pImpl idiom, Impl
    ///   must be a complete type wherever std::unique_ptr<Impl> is
    ///   destroyed, which is not possible in this header.
    ~OpenCVBackend() override;

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

    /// @brief Reads the .onnx file's custom metadata_props (written by
    ///   academy/utils/optimizers/metadata.py) via ONNX Runtime's
    ///   Ort::ModelMetadata API. Empty if the model has none.
    std::unordered_map<std::string, std::string> getMetadata() override;

  private:
    struct Impl;
    std::unique_ptr<Impl> p_impl_;
  };

} // namespace observatory::inference
