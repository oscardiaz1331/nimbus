#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <mdspan>
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

// Restricts as_span<T>() to the element types Tensor actually knows how to
// tag via TensorDataType. Catches e.g. as_span<double>() at compile time
// instead of silently aliasing an kInt64 buffer (sizeof(double) ==
// sizeof(int64_t), so a sizeof()-only check wouldn't have caught it).
template <typename T>
concept TensorElement = std::same_as<T, float> || std::same_as<T, std::int64_t> || std::same_as<T, std::uint8_t>;

template <TensorElement T>
constexpr TensorDataType dtype_for() {
  if constexpr (std::same_as<T, float>) {
    return TensorDataType::kFloat32;
  } else if constexpr (std::same_as<T, std::int64_t>) {
    return TensorDataType::kInt64;
  } else {
    return TensorDataType::kUInt8;
  }
}

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

  // Reinterprets the buffer as a typed span. Throws std::logic_error if T
  // isn't this tensor's actual element type (checked against dtype(), not
  // just sizeof(T) — so this can't alias e.g. an kInt64 buffer as double).
  template <TensorElement T>
  std::span<T> as_span() {
    if (dtype_for<T>() != dtype_) {
      throw std::logic_error("Tensor::as_span<T>: T does not match this tensor's dtype");
    }
    return std::span<T>(reinterpret_cast<T*>(data_.data()), element_count());
  }
  template <TensorElement T>
  std::span<const T> as_span() const {
    if (dtype_for<T>() != dtype_) {
      throw std::logic_error("Tensor::as_span<T>: T does not match this tensor's dtype");
    }
    return std::span<const T>(reinterpret_cast<const T*>(data_.data()), element_count());
  }

  // Multi-dimensional typed view over as_span<T>(), with extents taken from
  // shape(). Throws std::logic_error if Rank doesn't match shape().size()
  // (same spirit as as_span<T>()'s dtype check: a mismatch is a caller bug,
  // not a recoverable condition). Rank is a template parameter (not runtime)
  // because std::mdspan's extents are part of its type.
  template <TensorElement T, std::size_t Rank>
  std::mdspan<T, std::dextents<std::size_t, Rank>> as_mdspan() {
    return std::mdspan<T, std::dextents<std::size_t, Rank>>(as_span<T>().data(), extents_for<Rank>());
  }
  template <TensorElement T, std::size_t Rank>
  std::mdspan<const T, std::dextents<std::size_t, Rank>> as_mdspan() const {
    return std::mdspan<const T, std::dextents<std::size_t, Rank>>(as_span<T>().data(), extents_for<Rank>());
  }

 private:
  template <std::size_t Rank>
  std::array<std::size_t, Rank> extents_for() const {
    if (shape_.size() != Rank) {
      throw std::logic_error("Tensor::as_mdspan<T, Rank>: Rank does not match shape().size()");
    }
    std::array<std::size_t, Rank> extents{};
    // shape_ dims are guaranteed > 0 by the constructor, so this narrowing
    // int64_t -> size_t is always value-preserving, not just bit-preserving.
    std::ranges::transform(shape_, extents.begin(), [](int64_t dim) { return static_cast<std::size_t>(dim); });
    return extents;
  }

  std::string name_;
  std::vector<int64_t> shape_;
  TensorDataType dtype_;
  std::vector<std::byte> data_;
};

}  // namespace observatory::inference
