#include "observatory/inference/OpenCVBackend.hpp"

#include <cstring>
#include <filesystem>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>

#include <onnxruntime_cxx_api.h>

namespace observatory::inference
{

    /// @brief Holds every OpenCV/ONNX Runtime type used by OpenCVBackend.
    /// @details Defined only here (pImpl idiom) so OpenCVBackend.hpp never has
    ///   to include opencv2/dnn.hpp or onnxruntime_cxx_api.h.
    struct OpenCVBackend::Impl
    {
        /// @copydoc OpenCVBackend::OpenCVBackend
        explicit Impl(const std::string &model_path);

        /// @copydoc OpenCVBackend::run
        void Run(const std::vector<Tensor> &input, std::vector<Tensor> &output);

        /// @copydoc OpenCVBackend::getInputTensorsDefault
        std::vector<Tensor> getInputTensorsDefault();

        /// @copydoc OpenCVBackend::getOutputTensorsDefault
        std::vector<Tensor> getOutputTensorsDefault();

        /// @copydoc OpenCVBackend::getMetadata
        std::unordered_map<std::string, std::string> getMetadata();

    private:
        /// @brief Checks if the given model file is a valid ONNX model file.
        static bool IsModelFileValid(const std::filesystem::path &model_file);

        /// @brief Widens `path` to Ort::Session's required ORTCHAR_T (wchar_t
        ///   on Windows, char elsewhere) - mirrors OnnxRuntimeBackend.cpp's
        ///   ToOrtFileString.
        static std::basic_string<ORTCHAR_T> ToOrtFileString(const std::filesystem::path &path);

        /// @brief Maps a backend-agnostic TensorDataType to its cv::Mat depth
        ///   equivalent (CV_32F / CV_64S / CV_8U).
        static int ToCvDepth(TensorDataType dtype);

        /// @brief Maps an ONNX Runtime element type to its backend-agnostic
        ///   equivalent.
        /// @throws std::logic_error if `type` has no known mapping.
        static TensorDataType FromOnnxElementType(ONNXTensorElementDataType type);

        /// @brief Non-owning cv::Mat view over `tensor`'s own buffer - no copy,
        ///   no allocation. Only safe as an input to net_.setInput(), which
        ///   reads the blob rather than retaining a handle to it past the call.
        static cv::Mat WrapTensor(Tensor &tensor);

        /// @brief Narrows `shape` (int64_t, Tensor's convention) to the `int`
        ///   dims cv::Mat/MatShape expect. Tensor guarantees every dimension is
        ///   positive and small (pixel/feature counts), so this narrowing is
        ///   value-preserving in practice.
        static std::vector<int> ToCvSizes(const std::vector<int64_t> &shape);

        std::string model_path_;
        cv::dnn::Net net_;

        // Never Run() - exists purely so getInputTensorsDefault()/
        // getOutputTensorsDefault()/getMetadata() can answer from the same
        // declared-graph-metadata source OnnxRuntimeBackend uses (see class doc
        // comment in the header), so it skips graph optimization and execution
        // provider registration entirely.
        std::unique_ptr<Ort::Env> ort_env_;
        std::unique_ptr<Ort::Session> ort_session_;
        std::vector<Ort::AllocatedStringPtr> input_names_;
        std::vector<Ort::AllocatedStringPtr> output_names_;
        static inline const Ort::AllocatorWithDefaultOptions kCpuAllocator;
    };

    OpenCVBackend::Impl::Impl(const std::string &model_path) : model_path_(model_path)
    {
        if (!IsModelFileValid(model_path))
            throw std::runtime_error("OpenCVBackend: \"" + model_path + "\" is not a valid .onnx file");

        // ENGINE_AUTO (the default) prefers OpenCV 5's new graph engine, which -
        // at least as of this OpenCV build - divides by zero *inside a running
        // forward() call* (GatherNDLayerImpl::forward_impl(), a SIGFPE that no
        // C++ try/catch can intercept, let alone one wrapped around just the
        // constructor) on NMS-embedded exports like this one (GatherND is a
        // typical op in a baked-in NMS postprocessing subgraph). ENGINE_CLASSIC
        // fails on the *same* subgraph too (a Transpose/Permute shape-inference
        // assertion), but it fails at import time with a catchable exception -
        // a load-time std::runtime_error a caller can handle beats an
        // unrecoverable crash three calls later, even though neither engine can
        // actually run this particular graph today.
        try
        {
            net_ = cv::dnn::readNetFromONNX(model_path, cv::dnn::ENGINE_AUTO);
            if (net_.empty())
                throw std::runtime_error("OpenCVBackend: \"" + model_path + "\" produced an empty cv::dnn::Net");
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        catch (const cv::Exception &ex)
        {
            // cv::Exception derives from plain std::exception, not
            // std::runtime_error, so left uncaught it would silently break this
            // class's documented @throws contract (and EXPECT_THROW(...,
            // std::runtime_error) in tests).
            throw std::runtime_error("OpenCVBackend: cv::dnn failed to load \"" + model_path + "\": " + ex.what());
        }

        try
        {
            ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "observatory_opencv_backend");
            Ort::SessionOptions session_options;
            // This session only ever answers GetInputTypeInfo/GetOutputTypeInfo/
            // GetModelMetadata - it never runs, so optimizing the graph or
            // spinning up an intra-op thread pool would just be wasted work at
            // construction time.
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            session_options.SetIntraOpNumThreads(1);
            ort_session_ = std::make_unique<Ort::Session>(*ort_env_, ToOrtFileString(model_path).c_str(), session_options);
        }
        catch (const Ort::Exception &ex)
        {
            // Same reasoning as the cv::Exception catch above: Ort::Exception is
            // also a plain std::exception, not a std::runtime_error.
            throw std::runtime_error("OpenCVBackend: ONNX Runtime failed to open \"" + model_path + "\" for introspection: " + ex.what());
        }
        if (ort_session_->GetInputCount() == 0 || ort_session_->GetOutputCount() == 0)
            throw std::runtime_error("OpenCVBackend: session has no input or output tensors");

        const size_t input_count = ort_session_->GetInputCount();
        input_names_.reserve(input_count);
        for (size_t i = 0; i < input_count; ++i)
            input_names_.push_back(ort_session_->GetInputNameAllocated(i, kCpuAllocator));

        const size_t output_count = ort_session_->GetOutputCount();
        output_names_.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i)
            output_names_.push_back(ort_session_->GetOutputNameAllocated(i, kCpuAllocator));
    }

    void OpenCVBackend::Impl::Run(const std::vector<Tensor> &input, std::vector<Tensor> &output)
    {
        if (net_.empty())
            throw std::runtime_error("OpenCVBackend::Run: net_ is empty");
        if (input.empty() || output.empty())
            throw std::invalid_argument("OpenCVBackend::Run: input or output vector is empty");

        for (const Tensor &input_tensor : input)
        {
            // cv::dnn::Net::setInput() only accepts CV_32F or CV_8U blobs; an
            // int64 input (token ids, not pixel data) can't go through it.
            if (input_tensor.dtype() == TensorDataType::kInt64)
                throw std::invalid_argument("OpenCVBackend::Run: input \"" + input_tensor.name() +
                                            "\" is int64; cv::dnn::Net::setInput only accepts CV_32F or CV_8U blobs");
            Tensor &mutable_tensor = const_cast<Tensor &>(input_tensor);
            net_.setInput(WrapTensor(mutable_tensor), input_tensor.name());
        }

        std::vector<std::string> output_names;
        output_names.reserve(output.size());
        for (const Tensor &output_tensor : output)
            output_names.push_back(output_tensor.name());

        // net_.forward() always allocates its own Mats for the outputs - unlike
        // OnnxRuntimeBackend's IoBinding, cv::dnn::Net gives no documented way
        // to bind a caller-owned buffer as the destination, so the results are
        // copied into the caller's pre-shaped output Tensors below rather than
        // risking an unverified zero-copy assumption about forward()'s internals.
        std::vector<cv::Mat> raw_outputs;
        net_.forward(raw_outputs, output_names);

        if (raw_outputs.size() != output.size())
            throw std::runtime_error("OpenCVBackend::Run: forward() returned a different number of outputs than requested");

        for (auto &&[output_tensor, mat] : std::views::zip(output, raw_outputs))
        {
            const std::size_t mat_byte_size = mat.total() * mat.elemSize();
            if (mat_byte_size != output_tensor.byte_size())
                throw std::runtime_error("OpenCVBackend::Run: output \"" + output_tensor.name() +
                                         "\" came back a different size than its pre-shaped Tensor");
            std::memcpy(output_tensor.data(), mat.data, output_tensor.byte_size());
        }
    }

    std::vector<Tensor> OpenCVBackend::Impl::getInputTensorsDefault()
    {
        std::vector<Tensor> tensors;
        const size_t count = ort_session_->GetInputCount();
        tensors.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            const Ort::TypeInfo type_info = ort_session_->GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensor_info.GetShape();
            for (int64_t &dim : shape)
                if (dim <= 0)
                    dim = 1;
            tensors.emplace_back(input_names_[i].get(), std::move(shape), FromOnnxElementType(tensor_info.GetElementType()));
        }
        return tensors;
    }

    std::vector<Tensor> OpenCVBackend::Impl::getOutputTensorsDefault()
    {
        std::vector<Tensor> tensors;
        const size_t count = ort_session_->GetOutputCount();
        tensors.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            const Ort::TypeInfo type_info = ort_session_->GetOutputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensor_info.GetShape();
            for (int64_t &dim : shape)
                if (dim <= 0)
                    dim = 1;
            tensors.emplace_back(output_names_[i].get(), std::move(shape), FromOnnxElementType(tensor_info.GetElementType()));
        }
        return tensors;
    }

    std::unordered_map<std::string, std::string> OpenCVBackend::Impl::getMetadata()
    {
        std::unordered_map<std::string, std::string> metadata;
        const Ort::ModelMetadata model_metadata = ort_session_->GetModelMetadata();
        for (const auto &key : model_metadata.GetCustomMetadataMapKeysAllocated(kCpuAllocator))
        {
            Ort::AllocatedStringPtr value = model_metadata.LookupCustomMetadataMapAllocated(key.get(), kCpuAllocator);
            if (value)
                metadata.emplace(key.get(), value.get());
        }
        return metadata;
    }

    bool OpenCVBackend::Impl::IsModelFileValid(const std::filesystem::path &model_file)
    {
        return std::filesystem::is_regular_file(model_file) && model_file.extension() == ".onnx";
    }

    std::basic_string<ORTCHAR_T> OpenCVBackend::Impl::ToOrtFileString(const std::filesystem::path &path)
    {
        const std::string string(path.string());
        return {string.begin(), string.end()};
    }

    int OpenCVBackend::Impl::ToCvDepth(const TensorDataType dtype)
    {
        switch (dtype)
        {
        case TensorDataType::kFloat32:
            return CV_32F;
        case TensorDataType::kInt64:
            return CV_64S;
        case TensorDataType::kUInt8:
            return CV_8U;
        }
        throw std::logic_error("OpenCVBackend::ToCvDepth: unhandled TensorDataType");
    }

    TensorDataType OpenCVBackend::Impl::FromOnnxElementType(const ONNXTensorElementDataType type)
    {
        switch (type)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return TensorDataType::kFloat32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return TensorDataType::kInt64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return TensorDataType::kUInt8;
        default:
            throw std::logic_error(std::format("OpenCVBackend::FromOnnxElementType: unsupported ONNXTensorElementDataType {}", std::to_underlying(type)));
        }
    }

    cv::Mat OpenCVBackend::Impl::WrapTensor(Tensor &tensor)
    {
        const std::vector<int> sizes = ToCvSizes(tensor.shape());
        return cv::Mat(static_cast<int>(sizes.size()), sizes.data(), CV_MAKETYPE(ToCvDepth(tensor.dtype()), 1), tensor.data());
    }

    std::vector<int> OpenCVBackend::Impl::ToCvSizes(const std::vector<int64_t> &shape)
    {
        std::vector<int> sizes;
        sizes.reserve(shape.size());
        for (const int64_t dim : shape)
            sizes.push_back(static_cast<int>(dim));
        return sizes;
    }

    // --- OpenCVBackend: thin forwarding to Impl --------------------------------

    OpenCVBackend::OpenCVBackend(const std::string &model_path) : p_impl_(std::make_unique<Impl>(model_path))
    {
    }

    OpenCVBackend::~OpenCVBackend() = default;

    void OpenCVBackend::run(const std::vector<Tensor> &input, std::vector<Tensor> &output)
    {
        p_impl_->Run(input, output);
    }

    std::vector<Tensor> OpenCVBackend::getInputTensorsDefault()
    {
        return p_impl_->getInputTensorsDefault();
    }

    std::vector<Tensor> OpenCVBackend::getOutputTensorsDefault()
    {
        return p_impl_->getOutputTensorsDefault();
    }

    std::unordered_map<std::string, std::string> OpenCVBackend::getMetadata()
    {
        return p_impl_->getMetadata();
    }

} // namespace observatory::inference
