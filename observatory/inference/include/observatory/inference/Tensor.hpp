#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace observatory::inference {

enum class TensorDataType {
  kFloat32,
  kInt64,
  kUInt8,
};

std::size_t element_byte_width(TensorDataType dtype);

// Backend-agnostic named tensor: a contiguous, type-erased buffer plus the
// shape/dtype/name needed to bind it to a model input or output. Mirrors
// how Ort::Session::Run() takes named inputs/outputs, so IInferenceBackend
// implementations (and preprocessing/postprocessing, once they need it)
// don't have to carry OpenCV- or ORT-specific types in their contract.
class Tensor {
 public:
  Tensor(std::string name, std::vector<int64_t> shape, TensorDataType dtype);

  const std::string& name() const { return name_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  TensorDataType dtype() const { return dtype_; }

  // Number of elements implied by shape() (product of dimensions; 1 for a
  // scalar/empty shape).
  std::size_t element_count() const;
  // Size in bytes of the underlying buffer.
  std::size_t byte_size() const { return data_.size(); }

  std::byte* data() { return data_.data(); }
  const std::byte* data() const { return data_.data(); }

  // Reinterprets the buffer as a typed span. Throws std::logic_error if
  // sizeof(T) doesn't match dtype()'s element width.
  template <typename T>
  std::span<T> as_span() {
    if (sizeof(T) != element_byte_width(dtype_)) {
      throw std::logic_error("Tensor::as_span<T>: T's size does not match this tensor's dtype");
    }
    return std::span<T>(reinterpret_cast<T*>(data_.data()), element_count());
  }
  template <typename T>
  std::span<const T> as_span() const {
    if (sizeof(T) != element_byte_width(dtype_)) {
      throw std::logic_error("Tensor::as_span<T>: T's size does not match this tensor's dtype");
    }
    return std::span<const T>(reinterpret_cast<const T*>(data_.data()), element_count());
  }

 private:
  std::string name_;
  std::vector<int64_t> shape_;
  TensorDataType dtype_;
  std::vector<std::byte> data_;
};

}  // namespace observatory::inference
