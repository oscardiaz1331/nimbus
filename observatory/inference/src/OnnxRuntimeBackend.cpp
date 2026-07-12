#include "observatory/inference/OnnxRuntimeBackend.hpp"

#include <array>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <iostream>

namespace observatory::inference
{

  namespace
  {

#if defined(_WIN32)
    inline static constexpr std::string_view kDllPrefix = "";
    inline static constexpr std::string_view kDllSuffix = ".dll";
#else
    inline static constexpr std::string_view kDllPrefix = "lib";
    inline static constexpr std::string_view kDllSuffix = ".so";
#endif

    std::string dll_name(const std::string_view base_name)
    {
      return std::format("{}{}{}", kDllPrefix, base_name, kDllSuffix);
    }

    /// @brief  Maps an InferenceBackendType to the registration name used
    ///   in register_execution_providers(). Empty for the values that don't
    ///   pin a specific registered library (kOnnxRuntimeBest picks
    ///   automatically, kOnnxRuntimeCPU needs no registration, and
    ///   kTensorRT/kOpenVINO aren't ONNX Runtime EPs at all).
    constexpr std::string_view ep_registration_name(const InferenceBackendType ep_type)
    {
      switch (ep_type)
      {
      case InferenceBackendType::kOnnxRuntimeCUDA:
        return "cuda";
      case InferenceBackendType::kOnnxRuntimeOpenVINO:
        return "openvino";
      case InferenceBackendType::kOnnxRuntimeTensorRT:
        return "nv_tensorrt_rtx";
      default:
        return "";
      }
    }

    /// @brief  Whether `ep_type` is one this ONNX Runtime backend can honor.
    ///   kTensorRT/kOpenVINO name backends that don't go through ONNX
    ///   Runtime's EP mechanism at all (a future TensorRTBackend/
    ///   OpenVINOBackend would handle those directly).
    constexpr bool is_onnx_runtime_backend_type(const InferenceBackendType ep_type)
    {
      switch (ep_type)
      {
      case InferenceBackendType::kOnnxRuntimeBest:
      case InferenceBackendType::kOnnxRuntimeCPU:
      case InferenceBackendType::kOnnxRuntimeCUDA:
      case InferenceBackendType::kOnnxRuntimeOpenVINO:
      case InferenceBackendType::kOnnxRuntimeTensorRT:
        return true;
      default:
        return false;
      }
    }

    /// @brief  Returns the absolute path to the currently running executable.
    /// @details  Windows uses GetModuleFileNameW; everything else reads the
    ///   /proc/self/exe symlink (Linux). Throws std::runtime_error if the path
    ///   cannot be determined.
    std::filesystem::path get_executable_path()
    {
#if defined(_WIN32)
      std::wstring buffer(MAX_PATH, L'\0');
      for (;;)
      {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
        {
          throw std::runtime_error("get_executable_path: GetModuleFileNameW failed");
        }
        if (length < buffer.size())
        {
          buffer.resize(length);
          return std::filesystem::path(buffer);
        }
        // Buffer was too small; grow and retry.
        buffer.resize(buffer.size() * 2);
      }
#else
      std::error_code ec;
      auto path = std::filesystem::read_symlink("/proc/self/exe", ec);
      if (ec)
      {
        throw std::runtime_error("get_executable_path: failed to read /proc/self/exe: " + ec.message());
      }
      return path;
#endif
    }

  } // namespace

  OnnxRuntimeBackend::OnnxRuntimeBackend(const std::string &model_path, InferenceBackendType ep_type)
      : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "observatory"))
  {
    if (!env_)
      throw std::runtime_error("OnnxRuntimeBackend: failed to create Ort::Env");
    if (!is_onnx_runtime_backend_type(ep_type))
      throw std::runtime_error("OnnxRuntimeBackend: InferenceBackendType is not an ONNX Runtime backend type");
    if (!is_model_file_valid(model_path))
      throw std::runtime_error("OnnxRuntimeBackend: \"" + model_path + "\" is not a valid .onnx file");

    register_execution_providers();
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    select_execution_provider(session_options, ep_type);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    if (session_->GetInputCount() == 0 || session_->GetOutputCount() == 0)
      throw std::runtime_error("OnnxRuntimeBackend: session has no input or output tensors");

    io_binding_ = std::make_unique<Ort::IoBinding>(*session_);

    const size_t input_count = session_->GetInputCount();
    input_names_.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i)
      input_names_.push_back(session_->GetInputNameAllocated(i, kCpuAllocator));

    const size_t output_count = session_->GetOutputCount();
    output_names_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i)
      output_names_.push_back(session_->GetOutputNameAllocated(i, kCpuAllocator));

    input_memory_info_ = session_->GetMemoryInfoForInputs();
    output_memory_info_ = session_->GetMemoryInfoForOutputs();
  }

  void OnnxRuntimeBackend::run(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    preprocess(input, output);

    Ort::RunOptions run_options;
    session_->Run(run_options, *io_binding_);
    io_binding_->SynchronizeOutputs();
  }

  void OnnxRuntimeBackend::preprocess(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    if (env_ == nullptr || session_ == nullptr)
      throw std::runtime_error("OnnxRuntimeBackend::preprocess: env_ or session_ is null");
    if (input.empty() || output.empty())
      throw std::invalid_argument("OnnxRuntimeBackend::preprocess: input or output vector is empty");

    input_tensors_.clear();
    input_tensors_.reserve(input.size());
    output_tensors_.clear();
    output_tensors_.reserve(output.size());

    // Inputs: the caller's buffer already holds this frame's real data, so
    // anything going to device memory needs a CPU->device copy *before*
    // Run(). Anything staying on CPU is bound straight to the caller's
    // buffer - no copy needed.
    for (size_t i = 0; i < input.size(); ++i)
    {
      Tensor &tensor = const_cast<Tensor &>(input[i]);
      const auto &mem_info = input_memory_info_.at(i);
      Ort::Value cpu_value = create_ort_value(kCpuAllocator.GetInfo(), tensor);

      if (!needs_staged_copy(mem_info))
      {
        input_tensors_.push_back(std::move(cpu_value));
        continue;
      }

      auto device_allocator = env_->GetSharedAllocator(mem_info);
      Ort::Value device_value = Ort::Value::CreateTensor(
          device_allocator, tensor.shape().data(), tensor.shape().size(), to_onnx_element_type(tensor.dtype()));
      // Synchronous (stream == nullptr): completes before CopyTensor
      // returns, so cpu_value can be dropped right after.
      env_->CopyTensor(cpu_value, device_value, /*stream=*/nullptr);
      input_tensors_.push_back(std::move(device_value));
    }

    // Outputs: there's nothing to copy in yet - the model hasn't run. Device
    // outputs just need an empty buffer of the right shape; the device->host
    // copy happens after Run(), in run().
    for (size_t i = 0; i < output.size(); ++i)
    {
      Tensor &tensor = output[i];
      const auto &mem_info = output_memory_info_.at(i);

      if (!needs_staged_copy(mem_info))
      {
        output_tensors_.push_back(create_ort_value(kCpuAllocator.GetInfo(), tensor));
        continue;
      }

      auto device_allocator = env_->GetSharedAllocator(mem_info);
      output_tensors_.push_back(Ort::Value::CreateTensor(
          device_allocator, tensor.shape().data(), tensor.shape().size(), to_onnx_element_type(tensor.dtype())));
    }

    for (size_t i = 0; i < input_tensors_.size(); ++i)
      io_binding_->BindInput(input_names_.at(i).get(), input_tensors_[i]);
    for (size_t i = 0; i < output_tensors_.size(); ++i)
      io_binding_->BindOutput(output_names_.at(i).get(), output_tensors_[i]);
  }

  void OnnxRuntimeBackend::postprocess(std::vector<Tensor> &output)
  {
    // Outputs that landed on device memory aren't readable by the caller yet:
    // bring each one back into the caller's (CPU) Tensor buffer. Outputs that
    // were already bound directly to the caller's buffer need nothing further
    // - the data is already there.
    for (size_t i = 0; i < output.size(); ++i)
    {
      if (!needs_staged_copy(output_memory_info_.at(i)))
        continue;
      Ort::Value cpu_value = create_ort_value(kCpuAllocator.GetInfo(), output[i]);
      env_->CopyTensor(output_tensors_[i], cpu_value, /*stream=*/nullptr);
    }
  }

  void OnnxRuntimeBackend::register_execution_providers()
  {
    if (!env_)
    {
      throw std::runtime_error("OnnxRuntimeBackend::register_execution_providers: env_ is null");
    }

    static const auto get_ep_dll_name = [](const std::string_view ep_name)
    {
      return std::pair<std::string_view, std::string>(ep_name, dll_name(std::format("onnxruntime_providers_{}", ep_name)));
    };
    static const std::array<std::pair<std::string_view, std::string>, 5> provider_libraries{{
        get_ep_dll_name("nv_tensorrt_rtx"),
        get_ep_dll_name("cuda"),
        get_ep_dll_name("openvino"),
        get_ep_dll_name("qnn"),
        get_ep_dll_name("cann"),
    }};
    for (auto &[registration_name, dll] : provider_libraries)
    {
      const auto providers_library = get_executable_path().parent_path() / dll;
      if (!std::filesystem::is_regular_file(providers_library))
      {
        std::cerr << std::format("Provider library {} does not exist! Skipping execution provider", providers_library.string());
        continue;
      }
      try
      {
        env_->RegisterExecutionProviderLibrary(registration_name.data(), toOrtFileString(providers_library));
      }
      catch (std::exception &ex)
      {
        std::cerr << std::format("Failed to register {}! Skipping execution provider", providers_library.string());
      }
    }
  }

  void OnnxRuntimeBackend::select_execution_provider(Ort::SessionOptions &session_options, InferenceBackendType ep_type)
  {
    if (!env_)
    {
      throw std::runtime_error("OnnxRuntimeBackend::select_execution_provider: env_ is null");
    }

    if (!is_onnx_runtime_backend_type(ep_type))
    {
      throw std::runtime_error("OnnxRuntimeBackend::select_execution_provider: InferenceBackendType is not an ONNX Runtime backend type");
    }

    if (ep_type == InferenceBackendType::kOnnxRuntimeBest)
    {
      session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE);
      return;
    }

    if (ep_type == InferenceBackendType::kOnnxRuntimeCPU)
    {
      session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);
      return;
    }

    const auto registration_name = ep_registration_name(ep_type);
    if (registration_name.empty())
    {
      throw std::runtime_error("OnnxRuntimeBackend::select_execution_provider: unsupported InferenceBackendType");
    }

    std::vector<Ort::ConstEpDevice> selected_devices;
    for (const auto &ep_device : env_->GetEpDevices())
    {
      if (registration_name == ep_device.EpName())
      {
        selected_devices.push_back(ep_device);
      }
    }

    if (selected_devices.empty())
    {
      throw std::runtime_error(std::format(
          "OnnxRuntimeBackend::select_execution_provider: no devices found for execution provider \"{}\"",
          registration_name));
    }
    bool has_npu_mem = false, has_gpu_mem = false;
    for (const auto &device : selected_devices)
    {
      switch (device.Device().Type())
      {
      case OrtHardwareDeviceType_GPU:
        has_gpu_mem = true;
        break;
      case OrtHardwareDeviceType_NPU:
        has_npu_mem = true;
        break;
      case OrtHardwareDeviceType_CPU:
        break;
      }
    }
    OrtExecutionProviderDevicePolicy policy = has_npu_mem ? OrtExecutionProviderDevicePolicy_PREFER_NPU : has_gpu_mem ? OrtExecutionProviderDevicePolicy_PREFER_GPU
                                                                                                                      : OrtExecutionProviderDevicePolicy_PREFER_CPU;
    session_options.SetEpSelectionPolicy(policy);
    session_options.AppendExecutionProvider_V2(*env_, selected_devices, std::unordered_map<std::string, std::string>{});
  }

  bool OnnxRuntimeBackend::is_model_file_valid(const std::filesystem::path &model_file)
  {
    return std::filesystem::is_regular_file(model_file) && model_file.extension() == ".onnx";
  }

  ONNXTensorElementDataType OnnxRuntimeBackend::to_onnx_element_type(const TensorDataType dtype)
  {
    switch (dtype)
    {
    case TensorDataType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TensorDataType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TensorDataType::kUInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    throw std::logic_error("to_onnx_element_type: unhandled TensorDataType");
  }

  Ort::Value OnnxRuntimeBackend::create_ort_value(const OrtMemoryInfo *mem_info, Tensor &tensor)
  {
    const std::vector<int64_t> &shape = tensor.shape();
    return Ort::Value::CreateTensor(mem_info, tensor.data(), tensor.byte_size(), shape.data(), shape.size(), to_onnx_element_type(tensor.dtype()));
  }

} // namespace observatory::inference
