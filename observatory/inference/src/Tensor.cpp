#include "observatory/inference/Tensor.hpp"

#include <stdexcept>

namespace observatory::inference {

std::size_t element_byte_width(TensorDataType dtype) {
  switch (dtype) {
    case TensorDataType::kFloat32:
      return sizeof(float);
    case TensorDataType::kInt64:
      return sizeof(int64_t);
    case TensorDataType::kUInt8:
      return sizeof(uint8_t);
  }
  throw std::invalid_argument("element_byte_width: unknown TensorDataType");
}

Tensor::Tensor(std::string name, std::vector<int64_t> shape, TensorDataType dtype)
    : name_(std::move(name)), shape_(std::move(shape)), dtype_(dtype) {
  for (const int64_t dim : shape_) {
    if (dim <= 0) {
      throw std::invalid_argument("Tensor: shape dimensions must be positive, got " +
                                   std::to_string(dim));
    }
  }
  data_.resize(element_count() * element_byte_width(dtype_));
}

std::size_t Tensor::element_count() const {
  std::size_t count = 1;
  for (const int64_t dim : shape_) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

}  // namespace observatory::inference
