#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "observatory/inference/IInferenceBackend.hpp"
#include "observatory/inference/Tensor.hpp"

namespace observatory::inference
{
  using OrtFileString = std::basic_string<ORTCHAR_T>;
  // ONNX Runtime backend: loads a model at construction (RAII, throws on an
  // invalid path). `ep_type` picks the execution provider: kOnnxRuntime lets
  // ONNX Runtime auto-select the best-performing registered EP, while
  // kTensorRT/kOpenVINO pin the session to that specific provider.
  class OnnxRuntimeBackend final : public IInferenceBackend
  {
  public:
    explicit OnnxRuntimeBackend(const std::string &model_path, InferenceBackendType ep_type = InferenceBackendType::kOnnxRuntimeBest);
    ~OnnxRuntimeBackend() override = default;

    void run(const std::vector<Tensor> &input, std::vector<Tensor> &output) override;

  private:
    void register_execution_providers();
    void select_execution_provider(Ort::SessionOptions &session_options, InferenceBackendType ep_type);

    void preprocess(const std::vector<Tensor> &input, std::vector<Tensor> &output);
    void postprocess( std::vector<Tensor> &output);

    /// @brief  Checks if the given model file is a valid ONNX model file.
    /// @details  Checks if the given model file exists and has a .onnx extension.
    /// @param[in] model_file The path to the model file to check.
    /// @return True if the file is a valid ONNX model file, false otherwise.
    static bool is_model_file_valid(const std::filesystem::path &model_file);

    static ONNXTensorElementDataType to_onnx_element_type(const TensorDataType dtype);

    static Ort::Value create_ort_value(const OrtMemoryInfo *mem_info, Tensor &tensor);

    static inline OrtFileString toOrtFileString(const std::filesystem::path &path)
    {
      const std::string string(path.string());
      return {string.begin(), string.end()};
    }
    static inline bool needs_staged_copy(const Ort::ConstMemoryInfo &mem_info)
    {
      return mem_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU &&
             mem_info.GetDeviceMemoryType() == OrtDeviceMemoryType_DEFAULT;
    }

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::IoBinding> io_binding_;
    std::vector<Ort::AllocatedStringPtr> input_names_;
    std::vector<Ort::AllocatedStringPtr> output_names_;
    std::vector<Ort::ConstMemoryInfo> input_memory_info_;
    std::vector<Ort::ConstMemoryInfo> output_memory_info_;
    // Own the Ort::Value objects bound into io_binding_ so they outlive the
    // preprocess() call that creates them, through session_->Run() in run().
    std::vector<Ort::Value> input_tensors_;
    std::vector<Ort::Value> output_tensors_;
    static inline const Ort::AllocatorWithDefaultOptions kCpuAllocator;
  };

} // namespace observatory::inference
