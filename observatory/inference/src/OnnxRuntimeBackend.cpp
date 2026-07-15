#include "observatory/inference/OnnxRuntimeBackend.hpp"

#include <array>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <onnxruntime_cxx_api.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace observatory::inference
{

  using OrtFileString = std::basic_string<ORTCHAR_T>;

  namespace
  {

    /// @brief Maps an InferenceBackendType to the registration name used in
    ///   Impl::RegisterExecutionProviders(). Empty for the values that don't
    ///   pin a specific registered library (kOnnxRuntimeBest picks
    ///   automatically, kOnnxRuntimeCPU needs no registration, and
    ///   kTensorRT / kOpenVINO aren't ONNX Runtime EPs at all).
    constexpr std::string_view EpRegistrationName(const InferenceBackendType ep_type)
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

    /// @brief Whether `ep_type` is one this ONNX Runtime backend can honor.
    /// @details kTensorRT / kOpenVINO name backends that don't go through
    ///   ONNX Runtime's EP mechanism at all (a future TensorRTBackend /
    ///   OpenVINOBackend would handle those directly).
    constexpr bool IsOnnxRuntimeBackendType(const InferenceBackendType ep_type)
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

    /// @brief Returns the absolute path to the currently running executable.
    /// @details Windows uses GetModuleFileNameW; everything else reads the
    ///   /proc/self/exe symlink (Linux).
    /// @throws std::runtime_error if the path cannot be determined.
    std::filesystem::path GetExecutablePath()
    {
#if defined(_WIN32)
      std::wstring buffer(MAX_PATH, L'\0');
      for (;;)
      {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
          throw std::runtime_error("GetExecutablePath: GetModuleFileNameW failed");

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
        throw std::runtime_error("GetExecutablePath: failed to read /proc/self/exe: " + ec.message());
      return path;
#endif
    }

  } // namespace

  /// @brief Holds every ONNX Runtime type used by OnnxRuntimeBackend.
  /// @details Defined only here (pImpl idiom) so OnnxRuntimeBackend.hpp never
  ///   has to include onnxruntime_cxx_api.h.
  struct OnnxRuntimeBackend::Impl
  {
    /// @copydoc OnnxRuntimeBackend::OnnxRuntimeBackend
    Impl(const std::string &model_path, InferenceBackendType ep_type);

    /// @copydoc OnnxRuntimeBackend::run
    void Run(const std::vector<Tensor> &input, std::vector<Tensor> &output);

  private:
    /// @brief Registers every execution provider library shipped next to the
    ///   executable with `env_`. Missing or unloadable libraries are logged
    ///   and skipped rather than treated as fatal, since not every deployment
    ///   ships every EP (e.g. a CPU-only Pi image won't have CUDA's .so).
    void RegisterExecutionProviders();

    /// @brief Configures `session_options` for `ep_type`: either an ONNX
    ///   Runtime auto EP-selection policy, or a session pinned to the
    ///   specific device(s) registered for `ep_type`.
    /// @throws std::runtime_error if `ep_type` is not an ONNX Runtime backend
    ///   type, or no device is found for its execution provider.
    void SelectExecutionProvider(Ort::SessionOptions &session_options, InferenceBackendType ep_type);

    /// @brief Builds this frame's input/output Ort::Value tensors from
    ///   `input`/`output`, staging CPU->device copies where needed, and
    ///   binds them all into frame_slot_.io_binding ahead of Run().
    /// @param[in] input Preprocessed input tensors, one per model input.
    /// @param[in,out] output Pre-shaped output tensors; not written yet here
    ///   (that happens in Postprocess(), after the model has actually run).
    /// @throws std::runtime_error if the backend isn't initialized.
    /// @throws std::invalid_argument if `input`/`output` are empty or don't
    ///   match the model's input/output count.
    void Preprocess(const std::vector<Tensor> &input, std::vector<Tensor> &output);

    /// @brief Copies device-resident outputs back into the caller's `output`
    ///   Tensors after Run() has produced them. Outputs already bound
    ///   directly to the caller's buffer need nothing further.
    void Postprocess(std::vector<Tensor> &output);

    /// @brief Checks if the given model file is a valid ONNX model file.
    /// @details Checks if the given model file exists and has a .onnx extension.
    /// @param[in] model_file The path to the model file to check.
    /// @return True if the file is a valid ONNX model file, false otherwise.
    static bool IsModelFileValid(const std::filesystem::path &model_file);

    /// @brief Maps a backend-agnostic TensorDataType to its ONNX Runtime
    ///   equivalent.
    /// @throws std::logic_error if `dtype` has no known mapping.
    static ONNXTensorElementDataType ToOnnxElementType(const TensorDataType dtype);

    /// @brief Wraps `tensor`'s existing buffer as a non-owning Ort::Value at
    ///   `mem_info`, without copying or allocating.
    static Ort::Value CreateOrtValue(const OrtMemoryInfo *mem_info, Tensor &tensor);

    static inline OrtFileString ToOrtFileString(const std::filesystem::path &path)
    {
      const std::string string(path.string());
      return {string.begin(), string.end()};
    }

    /// @brief Whether a tensor at `mem_info` lives in memory the caller can't
    ///   write to/read from directly, and therefore needs a staged
    ///   CPU<->device Ort::Value plus an explicit CopyTensor.
    static inline bool NeedsStagedCopy(const Ort::ConstMemoryInfo &mem_info)
    {
      return mem_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU &&
             mem_info.GetDeviceMemoryType() == OrtDeviceMemoryType_DEFAULT;
    }

    /// @brief Retrieves the OrtSyncStreamImpl backing `stream`, needed to
    ///   create OrtSyncNotificationImpl objects for it.
    /// @return The OrtSyncStreamImpl, or an error message if the EP doesn't
    ///   expose one for `stream`. Not every EP implements OrtSyncStream (see
    ///   create_stream_and_notification's caller), so "not found" is a
    ///   routine, expected outcome here, not a programming error - hence
    ///   std::expected instead of throwing.
    static inline std::expected<OrtSyncStreamImpl *, std::string> GetSyncStreamImpl(const Ort::SyncStream &stream)
    {
      const OrtSyncStreamImpl *impl = Ort::GetEpApi().SyncStream_GetImpl(stream);
      if (impl == nullptr)
        return std::unexpected("EP does not expose an OrtSyncStreamImpl for this stream");
      return const_cast<OrtSyncStreamImpl *>(impl);
    }

    /// @brief Deleter for OrtSyncNotificationImpl, so it can be owned by a
    ///   std::unique_ptr instead of manually calling Release().
    struct SyncNotificationDeleter
    {
      void operator()(OrtSyncNotificationImpl *notification) const noexcept
      {
        if (notification != nullptr)
          notification->Release(notification);
      }
    };
    using SyncNotificationPtr = std::unique_ptr<OrtSyncNotificationImpl, SyncNotificationDeleter>;

    /// @brief Creates a reusable OrtSyncNotificationImpl for `stream_impl`,
    ///   used to synchronize between upload/compute/download streams (see
    ///   upload_notification_, compute_notification_, download_notification_).
    static inline SyncNotificationPtr CreateNotification(OrtSyncStreamImpl *stream_impl)
    {
      OrtSyncNotificationImpl *raw = nullptr;
      Ort::ThrowOnError(stream_impl->CreateNotification(stream_impl, &raw));
      return SyncNotificationPtr(raw);
    }

    /// @brief The Ort::IoBinding and Ort::Value objects for one in-flight
    ///   frame. Owns the Ort::Value objects bound into io_binding, so they
    ///   outlive the Preprocess() call that creates them, through
    ///   session_->Run() in Run().
    /// @todo Convert to an N-buffered ring for cross-frame pipelining once
    ///   camera/ exists and the benchmark module shows upload/download is the
    ///   bottleneck.
    struct FrameSlot
    {
      std::unique_ptr<Ort::IoBinding> io_binding{nullptr};
      std::vector<Ort::Value> input_tensors{}, output_tensors{};
    };

    /// @brief Precomputed once per input/output index at construction time
    ///   (a tensor's mem_info and allocator never change between frames),
    ///   instead of re-resolved on every Preprocess()/Postprocess() call.
    struct TensorPlan
    {
      bool staged;                     ///< True: needs a CPU<->device copy through a staged Ort::Value.
      OrtAllocator *device_allocator;  ///< Only valid if staged == true.
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
    // Separate streams for the CPU->device upload, the compute (Run()) step,
    // and the device->CPU download, so each can be synchronized against the
    // others independently. See upload_notification_/compute_notification_/
    // download_notification_ below for how they're stitched together.
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
    if (!IsOnnxRuntimeBackendType(ep_type))
      throw std::runtime_error("OnnxRuntimeBackend: InferenceBackendType is not an ONNX Runtime backend type");
    if (!IsModelFileValid(model_path))
      throw std::runtime_error("OnnxRuntimeBackend: \"" + model_path + "\" is not a valid .onnx file");
    RegisterExecutionProviders();
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    SelectExecutionProvider(session_options, ep_type);
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
        bool staged = NeedsStagedCopy(mem_info) && allocator != nullptr;
        plan.push_back({staged, staged ? allocator : nullptr});
      }
      return plan;
    };
    input_plan_ = build_plan(input_memory_info_);
    output_plan_ = build_plan(output_memory_info_);

    const auto create_stream_and_notification = [](const Ort::ConstEpDevice &device, std::unique_ptr<Ort::SyncStream> &stream, SyncNotificationPtr &notification) static
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
        // GetSyncStreamImpl returns std::expected rather than throwing: "no
        // impl for this stream" is a routine outcome (not every EP
        // implements OrtSyncStream), so surface it as a value here. It's
        // turned into an exception right after purely to join the single
        // recovery path below, shared with CreateSyncStream()/
        // CreateNotification() above/below, which are ONNX Runtime calls we
        // don't control and that do throw on failure.
        const std::expected<OrtSyncStreamImpl *, std::string> stream_impl = GetSyncStreamImpl(*stream);
        if (!stream_impl.has_value())
          throw std::runtime_error(stream_impl.error());
        notification = CreateNotification(stream_impl.value());
      }
      catch (const std::exception &ex)
      {
        // std::to_underlying (C++23): explicit about using the enum's real
        // underlying type instead of assuming it's int via static_cast<int>.
        std::println(stderr, "Device/EP type {} does not support SyncStream ({}), falling back to synchronous copies",
                     std::to_underlying(device.Device().Type()), ex.what());
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

  void OnnxRuntimeBackend::Impl::Run(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    Preprocess(input, output);
    session_->Run(*run_options_, *frame_slot_.io_binding);
    frame_slot_.io_binding->SynchronizeOutputs();
    if (compute_notification_ != nullptr && download_stream_ != nullptr)
    {
      // Let download_stream_ wait (on the device, without blocking the CPU)
      // for every op enqueued on compute_stream_ - including this Run() - to
      // finish, before Postprocess() issues the device->host copy below.
      Ort::ThrowOnError(compute_notification_->Activate(compute_notification_.get()));
      Ort::ThrowOnError(compute_notification_->WaitOnDevice(compute_notification_.get(), *download_stream_.get()));
    }
    Postprocess(output);
  }

  void OnnxRuntimeBackend::Impl::Preprocess(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    if (env_ == nullptr || session_ == nullptr)
      throw std::runtime_error("OnnxRuntimeBackend::Preprocess: env_ or session_ is null");
    if (input.empty() || output.empty())
      throw std::invalid_argument("OnnxRuntimeBackend::Preprocess: input or output vector is empty");
    if (input.size() != input_plan_.size())
      throw std::invalid_argument("OnnxRuntimeBackend::Preprocess: input size does not match model input count");
    if (output.size() != output_plan_.size())
      throw std::invalid_argument("OnnxRuntimeBackend::Preprocess: output size does not match model output count");

    frame_slot_.input_tensors.clear();
    frame_slot_.input_tensors.reserve(input.size());
    frame_slot_.output_tensors.clear();
    frame_slot_.output_tensors.reserve(output.size());

    // Inputs: the caller's buffer already holds this frame's real data, so
    // anything going to device memory needs a CPU->device copy *before*
    // Run(). Anything staying on CPU is bound straight to the caller's
    // buffer - no copy needed.
    // std::views::zip (C++23): walks input/input_plan_/input_names_ together
    // instead of indexing all three by a shared `i` - one less moving part to
    // keep in sync (this exact loop has been the site of index-related bugs
    // before).
    bool any_input_staged = false;
    for (auto &&[input_tensor, plan, name] : std::views::zip(input, input_plan_, input_names_))
    {
      Tensor &tensor = const_cast<Tensor &>(input_tensor);
      Ort::Value cpu_value = CreateOrtValue(kCpuAllocator.GetInfo(), tensor);

      if (!plan.staged)
        frame_slot_.input_tensors.push_back(std::move(cpu_value));
      else
      {
        Ort::Value device_value = Ort::Value::CreateTensor(
            plan.device_allocator, tensor.shape().data(), tensor.shape().size(), ToOnnxElementType(tensor.dtype()));
        // Synchronous (stream == nullptr): completes before CopyTensor
        // returns, so cpu_value can be dropped right after.
        // Note that if upload_stream_ == nullptr, it will synchronously work in CPU
        OrtSyncStream *upload = upload_stream_ ? static_cast<OrtSyncStream *>(*upload_stream_) : nullptr;
        env_->CopyTensor(cpu_value, device_value, /*stream=*/upload);
        any_input_staged = true;
        frame_slot_.input_tensors.push_back(std::move(device_value));
      }
      frame_slot_.io_binding->BindInput(name.get(), frame_slot_.input_tensors.back());
    }
    if (any_input_staged && upload_notification_ != nullptr && compute_stream_ != nullptr)
    {
      // One barrier for the whole batch: every CopyTensor above shares
      // upload_stream_, and a stream executes in FIFO order, so a single
      // Activate() here already covers every copy enqueued in the loop.
      Ort::ThrowOnError(upload_notification_->Activate(upload_notification_.get()));
      Ort::ThrowOnError(upload_notification_->WaitOnDevice(upload_notification_.get(), *compute_stream_.get()));
    }
    // Outputs: there's nothing to copy in yet - the model hasn't run. Device
    // outputs just need an empty buffer of the right shape; the device->host
    // copy happens after Run(), in Postprocess().
    for (auto &&[tensor, plan, name] : std::views::zip(output, output_plan_, output_names_))
    {
      if (!plan.staged)
        frame_slot_.output_tensors.push_back(CreateOrtValue(kCpuAllocator.GetInfo(), tensor));
      else
        frame_slot_.output_tensors.push_back(Ort::Value::CreateTensor(
            plan.device_allocator, tensor.shape().data(), tensor.shape().size(), ToOnnxElementType(tensor.dtype())));

      frame_slot_.io_binding->BindOutput(name.get(), frame_slot_.output_tensors.back());
    }
  }

  void OnnxRuntimeBackend::Impl::Postprocess(std::vector<Tensor> &output)
  {
    // Outputs that landed on device memory aren't readable by the caller yet:
    // bring each one back into the caller's (CPU) Tensor buffer. Outputs that
    // were already bound directly to the caller's buffer need nothing further
    // - the data is already there.
    bool any_output_staged = false;
    for (auto &&[plan, tensor, device_value] : std::views::zip(output_plan_, output, frame_slot_.output_tensors))
    {
      if (!plan.staged)
        continue;
      Ort::Value cpu_value = CreateOrtValue(kCpuAllocator.GetInfo(), tensor);
      OrtSyncStream *download = download_stream_ ? static_cast<OrtSyncStream *>(*download_stream_) : nullptr;
      env_->CopyTensor(device_value, cpu_value, /*stream=*/download);
      any_output_staged = true;
    }
    if (any_output_staged && download_notification_ != nullptr)
    {
      // Unlike the upload/compute handoff, the consumer here is the CPU
      // itself (the caller is about to read `output`), so this waits on the
      // host rather than on another device stream.
      Ort::ThrowOnError(download_notification_->Activate(download_notification_.get()));
      Ort::ThrowOnError(download_notification_->WaitOnHost(download_notification_.get()));
    }
  }

  void OnnxRuntimeBackend::Impl::RegisterExecutionProviders()
  {
    if (!env_)
      throw std::runtime_error("OnnxRuntimeBackend::RegisterExecutionProviders: env_ is null");

    // Every name here is known at compile time, so this is a plain constexpr
    // table instead of building strings with std::format at runtime.
#if defined(_WIN32)
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 5> kProviderLibraries{{
        {"nv_tensorrt_rtx", "onnxruntime_providers_nv_tensorrt_rtx.dll"},
        {"cuda", "onnxruntime_providers_cuda.dll"},
        {"openvino", "onnxruntime_providers_openvino.dll"},
        {"qnn", "onnxruntime_providers_qnn.dll"},
        {"cann", "onnxruntime_providers_cann.dll"},
    }};
#else
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 5> kProviderLibraries{{
        {"nv_tensorrt_rtx", "libonnxruntime_providers_nv_tensorrt_rtx.so"},
        {"cuda", "libonnxruntime_providers_cuda.so"},
        {"openvino", "libonnxruntime_providers_openvino.so"},
        {"qnn", "libonnxruntime_providers_qnn.so"},
        {"cann", "libonnxruntime_providers_cann.so"},
    }};
#endif
    const auto executable_parent_path = GetExecutablePath().parent_path();
    for (auto &[registration_name, dll] : kProviderLibraries)
    {
      const auto providers_library = executable_parent_path / dll;
      if (!std::filesystem::is_regular_file(providers_library))
      {
        std::println(stderr, "Provider library {} does not exist! Skipping execution provider", providers_library.string());
        continue;
      }
      try
      {
        env_->RegisterExecutionProviderLibrary(registration_name.data(), ToOrtFileString(providers_library));
      }
      catch (std::exception &ex)
      {
        std::println(stderr, "Failed to register {}: {}! Skipping execution provider", providers_library.string(), ex.what());
      }
    }
  }

  void OnnxRuntimeBackend::Impl::SelectExecutionProvider(Ort::SessionOptions &session_options, InferenceBackendType ep_type)
  {
    if (!env_)
      throw std::runtime_error("OnnxRuntimeBackend::SelectExecutionProvider: env_ is null");

    if (!IsOnnxRuntimeBackendType(ep_type))
      throw std::runtime_error("OnnxRuntimeBackend::SelectExecutionProvider: InferenceBackendType is not an ONNX Runtime backend type");

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

    const auto registration_name = EpRegistrationName(ep_type);
    if (registration_name.empty())
      throw std::runtime_error("OnnxRuntimeBackend::SelectExecutionProvider: unsupported InferenceBackendType");

    std::vector<Ort::ConstEpDevice> selected_devices;
    for (const auto &ep_device : env_->GetEpDevices())
    {
      if (registration_name == ep_device.EpName())
        selected_devices.push_back(ep_device);
    }

    if (selected_devices.empty())
      throw std::runtime_error(std::format(
          "OnnxRuntimeBackend::SelectExecutionProvider: no devices found for execution provider \"{}\"",
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

  bool OnnxRuntimeBackend::Impl::IsModelFileValid(const std::filesystem::path &model_file)
  {
    return std::filesystem::is_regular_file(model_file) && model_file.extension() == ".onnx";
  }

  ONNXTensorElementDataType OnnxRuntimeBackend::Impl::ToOnnxElementType(const TensorDataType dtype)
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
    throw std::logic_error("ToOnnxElementType: unhandled TensorDataType");
  }

  Ort::Value OnnxRuntimeBackend::Impl::CreateOrtValue(const OrtMemoryInfo *mem_info, Tensor &tensor)
  {
    const std::vector<int64_t> &shape = tensor.shape();
    return Ort::Value::CreateTensor(mem_info, tensor.data(), tensor.byte_size(), shape.data(), shape.size(), ToOnnxElementType(tensor.dtype()));
  }

  // --- OnnxRuntimeBackend: thin forwarding to Impl ---------------------------

  OnnxRuntimeBackend::OnnxRuntimeBackend(const std::string &model_path, InferenceBackendType ep_type)
      : p_impl_(std::make_unique<Impl>(model_path, ep_type))
  {
  }

  OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

  void OnnxRuntimeBackend::run(const std::vector<Tensor> &input, std::vector<Tensor> &output)
  {
    p_impl_->Run(input, output);
  }

} // namespace observatory::inference
