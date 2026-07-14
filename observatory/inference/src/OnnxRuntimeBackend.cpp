#include "observatory/inference/OnnxRuntimeBackend.hpp"

#include <onnxruntime_cxx_api.h>

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

  using OrtFileString = std::basic_string<ORTCHAR_T>;

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
          throw std::runtime_error("get_executable_path: GetModuleFileNameW failed");

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
        throw std::runtime_error("get_executable_path: failed to read /proc/self/exe: " + ec.message());
      return path;
#endif
    }

  } // namespace

  // Holds every ONNX Runtime type. Defined only here so OnnxRuntimeBackend.hpp
  // never has to include onnxruntime_cxx_api.h.
  struct OnnxRuntimeBackend::Impl
  {
    Impl(const std::string &model_path, InferenceBackendType ep_type);

    void run(const std::vector<Tensor> &input, std::vector<Tensor> &output);

  private:
    void register_execution_providers();
    void select_execution_provider(Ort::SessionOptions &session_options, InferenceBackendType ep_type);

    void preprocess(const std::vector<Tensor> &input, std::vector<Tensor> &output);
    void postprocess(std::vector<Tensor> &output);

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

    static inline OrtSyncStreamImpl *sync_stream_impl(const Ort::SyncStream &stream)
    {
      const OrtSyncStreamImpl *impl = Ort::GetEpApi().SyncStream_GetImpl(stream);
      if (impl == nullptr)
        throw std::runtime_error("sync_stream_impl: EP does not expose an OrtSyncStreamImpl for this stream");
      return const_cast<OrtSyncStreamImpl *>(impl);
    }

    struct SyncNotificationDeleter
    {
      void operator()(OrtSyncNotificationImpl *notification) const noexcept
      {
        if (notification != nullptr)
          notification->Release(notification);
      }
    };
    using SyncNotificationPtr = std::unique_ptr<OrtSyncNotificationImpl, SyncNotificationDeleter>;

    static inline SyncNotificationPtr create_notification(OrtSyncStreamImpl *stream_impl)
    {
      OrtSyncNotificationImpl *raw = nullptr;
      Ort::ThrowOnError(stream_impl->CreateNotification(stream_impl, &raw));
      return SyncNotificationPtr(raw);
    }

    // TODO: convert to N-buffered ring for cross-frame pipelining once camera/
    // exists and the benchmark module shows upload/download is the bottleneck.
    struct FrameSlot
    {
      std::unique_ptr<Ort::IoBinding> io_binding{nullptr};
      std::vector<Ort::Value> input_tensors{}, output_tensors{};
    };

    // Precomputed once per input/output index at construction time (mem_info
    // and its allocator never change between frames) instead of re-resolved
    // on every preprocess()/postprocess() call.
    struct TensorPlan
    {
      bool staged;                    // true: needs a CPU<->device copy through a staged Ort::Value.
      OrtAllocator *device_allocator;  // only valid if staged == true.
    };

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::RunOptions> run_options_;
    std::vector<Ort::AllocatedStringPtr> input_names_;
    std::vector<Ort::AllocatedStringPtr> output_names_;
    std::vector<Ort::ConstMemoryInfo> input_memory_info_;
    std::vector<Ort::ConstMemoryInfo> output_memory_info_;
    std::vector<TensorPlan> input_plan_;
    std::vector<TensorPlan> output_plan_;
    std::unique_ptr<Ort::SyncStream> upload_stream_;
    SyncNotificationPtr upload_notification_;
    std::unique_ptr<Ort::SyncStream> compute_stream_;
    SyncNotificationPtr compute_notification_;
    std::unique_ptr<Ort::SyncStream> download_stream_;
    SyncNotificationPtr download_notification_;
    FrameSlot frame_slot_;
    static inline const Ort::AllocatorWithDefaultOptions kCpuAllocator;
  };

  OnnxRuntimeBackend::Impl::Impl(const std::string &model_path, InferenceBackendType ep_type)
      : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "observatory")), run_options_(std::make_unique<Ort::RunOptions>())
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

    frame_slot_ = FrameSlot{.io_binding{std::make_unique<Ort::IoBinding>(*session_)}, .input_tensors{}, .output_tensors{}};

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

    const auto build_plan = [this](const std::vector<Ort::ConstMemoryInfo> &mem_infos)
    {
      std::vector<TensorPlan> plan;
      plan.reserve(mem_infos.size());
      for (const auto &mem_info : mem_infos)
      {
        OrtAllocator *allocator = env_->GetSharedAllocator(mem_info);
        bool staged = needs_staged_copy(mem_info) && allocator != nullptr;
        plan.push_back({staged, staged ? allocator : nullptr});
      }
      return plan;
    };
    input_plan_ = build_plan(input_memory_info_);
    output_plan_ = build_plan(output_memory_info_);

    const auto create_stream_and_notification = [](const Ort::ConstEpDevice &device, std::unique_ptr<Ort::SyncStream> &stream, SyncNotificationPtr &notification)
    {
      if (device == nullptr || device.Device().Type() == OrtHardwareDeviceType_CPU)
      {
        stream = nullptr;
        notification = nullptr;
        return;
      }
      try
      {
        stream = std::make_unique<Ort::SyncStream>(device.CreateSyncStream());
        notification = create_notification(sync_stream_impl(*stream));
      }
      catch (const std::exception &)
      {
        std::cerr << std::format("Device/EP type {} does not support SyncStream, falling back to synchronous copies",
                                 static_cast<int>(device.Device().Type()))
                  << std::endl;
        stream = nullptr;
        notification = nullptr;
      }
    };

    // @todo check how to work with different devices
    const auto input_ep_device = session_->GetEpDeviceForInputs().at(0);
    create_stream_and_notification(input_ep_device, upload_stream_, upload_notification_);
    create_stream_and_notification(input_ep_device, compute_stream_, compute_notification_);
    const auto output_ep_device = session_->GetEpDeviceForOutputs().at(0);
    create_stream_and_notification(output_ep_device, download_stream_, download_notification_);
  }

  void OnnxRuntimeBackend::Impl::run(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    preprocess(input, output);
    session_->Run(*run_options_, *frame_slot_.io_binding);
    frame_slot_.io_binding->SynchronizeOutputs();
    if (compute_notification_ != nullptr && download_stream_ != nullptr)
    {
      Ort::ThrowOnError(compute_notification_->Activate(compute_notification_.get()));
      Ort::ThrowOnError(compute_notification_->WaitOnDevice(compute_notification_.get(), *download_stream_.get()));
    }
    postprocess(output);
  }

  void OnnxRuntimeBackend::Impl::preprocess(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    if (env_ == nullptr || session_ == nullptr)
      throw std::runtime_error("OnnxRuntimeBackend::preprocess: env_ or session_ is null");
    if (input.empty() || output.empty())
      throw std::invalid_argument("OnnxRuntimeBackend::preprocess: input or output vector is empty");
    if (input.size() != input_plan_.size())
      throw std::invalid_argument("OnnxRuntimeBackend::preprocess: input size does not match model input count");
    if (output.size() != output_plan_.size())
      throw std::invalid_argument("OnnxRuntimeBackend::preprocess: output size does not match model output count");

    frame_slot_.input_tensors.clear();
    frame_slot_.input_tensors.reserve(input.size());
    frame_slot_.output_tensors.clear();
    frame_slot_.output_tensors.reserve(output.size());

    // Inputs: the caller's buffer already holds this frame's real data, so
    // anything going to device memory needs a CPU->device copy *before*
    // Run(). Anything staying on CPU is bound straight to the caller's
    // buffer - no copy needed.
    bool any_input_staged = false;
    for (size_t i = 0; i < input.size(); ++i)
    {
      Tensor &tensor = const_cast<Tensor &>(input[i]);
      Ort::Value cpu_value = create_ort_value(kCpuAllocator.GetInfo(), tensor);

      const auto &plan = input_plan_[i];
      if (!plan.staged)
        frame_slot_.input_tensors.push_back(std::move(cpu_value));
      else
      {
        Ort::Value device_value = Ort::Value::CreateTensor(
            plan.device_allocator, tensor.shape().data(), tensor.shape().size(), to_onnx_element_type(tensor.dtype()));
        // Synchronous (stream == nullptr): completes before CopyTensor
        // returns, so cpu_value can be dropped right after.
        // Note that if upload_stream_ == nullptr, it will synchronously work in CPU
        OrtSyncStream *upload = upload_stream_ ? static_cast<OrtSyncStream *>(*upload_stream_) : nullptr;
        env_->CopyTensor(cpu_value, device_value, /*stream=*/upload);
        any_input_staged = true;
        frame_slot_.input_tensors.push_back(std::move(device_value));
      }
      frame_slot_.io_binding->BindInput(input_names_[i].get(), frame_slot_.input_tensors[i]);
    }
    if (any_input_staged && upload_notification_ != nullptr && compute_stream_ != nullptr)
    {
      Ort::ThrowOnError(upload_notification_->Activate(upload_notification_.get()));
      Ort::ThrowOnError(upload_notification_->WaitOnDevice(upload_notification_.get(), *compute_stream_.get()));
    }
    // Outputs: there's nothing to copy in yet - the model hasn't run. Device
    // outputs just need an empty buffer of the right shape; the device->host
    // copy happens after Run(), in run().
    for (size_t i = 0; i < output.size(); ++i)
    {
      Tensor &tensor = output[i];
      const auto &plan = output_plan_[i];
      if (!plan.staged)
        frame_slot_.output_tensors.push_back(create_ort_value(kCpuAllocator.GetInfo(), tensor));
      else
        frame_slot_.output_tensors.push_back(Ort::Value::CreateTensor(
            plan.device_allocator, tensor.shape().data(), tensor.shape().size(), to_onnx_element_type(tensor.dtype())));

      frame_slot_.io_binding->BindOutput(output_names_[i].get(), frame_slot_.output_tensors[i]);
    }
  }

  void OnnxRuntimeBackend::Impl::postprocess(std::vector<Tensor> &output)
  {
    // Outputs that landed on device memory aren't readable by the caller yet:
    // bring each one back into the caller's (CPU) Tensor buffer. Outputs that
    // were already bound directly to the caller's buffer need nothing further
    // - the data is already there.
    bool any_output_staged = false;
    for (size_t i = 0; i < output.size(); ++i)
    {
      const auto &plan = output_plan_[i];
      if (!plan.staged)
        continue;
      Ort::Value cpu_value = create_ort_value(kCpuAllocator.GetInfo(), output[i]);
      OrtSyncStream *download = download_stream_ ? static_cast<OrtSyncStream *>(*download_stream_) : nullptr;
      env_->CopyTensor(frame_slot_.output_tensors[i], cpu_value, /*stream=*/download);
      any_output_staged = true;
    }
    if (any_output_staged && download_notification_ != nullptr)
    {
      Ort::ThrowOnError(download_notification_->Activate(download_notification_.get()));
      Ort::ThrowOnError(download_notification_->WaitOnHost(download_notification_.get()));
    }
  }

  void OnnxRuntimeBackend::Impl::register_execution_providers()
  {
    if (!env_)
      throw std::runtime_error("OnnxRuntimeBackend::register_execution_providers: env_ is null");

    static const auto get_ep_dll_name = [](const std::string_view ep_name)
    {
      return std::pair<std::string_view, std::string>(ep_name, dll_name(std::format("onnxruntime_providers_{}", ep_name)));
    };
    
    static const std::array<std::pair<std::string_view, std::string>, 5> kProviderLibraries{{
        get_ep_dll_name("nv_tensorrt_rtx"),
        get_ep_dll_name("cuda"),
        get_ep_dll_name("openvino"),
        get_ep_dll_name("qnn"),
        get_ep_dll_name("cann"),
    }};
    const auto kFolderParentPath = get_executable_path().parent_path();
    for (auto &[registration_name, dll] : kProviderLibraries)
    {
      const auto providers_library = kFolderParentPath / dll;
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
        std::cerr << std::format("Failed to register {}: {}! Skipping execution provider", providers_library.string(), ex.what());
      }
    }
  }

  void OnnxRuntimeBackend::Impl::select_execution_provider(Ort::SessionOptions &session_options, InferenceBackendType ep_type)
  {
    if (!env_)
      throw std::runtime_error("OnnxRuntimeBackend::select_execution_provider: env_ is null");

    if (!is_onnx_runtime_backend_type(ep_type))
      throw std::runtime_error("OnnxRuntimeBackend::select_execution_provider: InferenceBackendType is not an ONNX Runtime backend type");

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
      throw std::runtime_error("OnnxRuntimeBackend::select_execution_provider: unsupported InferenceBackendType");

    std::vector<Ort::ConstEpDevice> selected_devices;
    for (const auto &ep_device : env_->GetEpDevices())
    {
      if (registration_name == ep_device.EpName())
        selected_devices.push_back(ep_device);
    }

    if (selected_devices.empty())
      throw std::runtime_error(std::format(
          "OnnxRuntimeBackend::select_execution_provider: no devices found for execution provider \"{}\"",
          registration_name));

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

  bool OnnxRuntimeBackend::Impl::is_model_file_valid(const std::filesystem::path &model_file)
  {
    return std::filesystem::is_regular_file(model_file) && model_file.extension() == ".onnx";
  }

  ONNXTensorElementDataType OnnxRuntimeBackend::Impl::to_onnx_element_type(const TensorDataType dtype)
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

  Ort::Value OnnxRuntimeBackend::Impl::create_ort_value(const OrtMemoryInfo *mem_info, Tensor &tensor)
  {
    const std::vector<int64_t> &shape = tensor.shape();
    return Ort::Value::CreateTensor(mem_info, tensor.data(), tensor.byte_size(), shape.data(), shape.size(), to_onnx_element_type(tensor.dtype()));
  }

  // --- OnnxRuntimeBackend: thin forwarding to Impl ---------------------------

  OnnxRuntimeBackend::OnnxRuntimeBackend(const std::string &model_path, InferenceBackendType ep_type)
      : p_impl_(std::make_unique<Impl>(model_path, ep_type))
  {
  }

  OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

  void OnnxRuntimeBackend::run(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    p_impl_->run(input, output);
  }

} // namespace observatory::inference
