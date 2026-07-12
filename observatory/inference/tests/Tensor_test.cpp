#include "observatory/inference/Tensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

namespace observatory::inference {
namespace {

TEST(Tensor, ElementCountAndByteSize) {
  const Tensor tensor("input", {1, 3, 2, 2}, TensorDataType::kFloat32);
  EXPECT_EQ(tensor.element_count(), 12U);
  EXPECT_EQ(tensor.byte_size(), 12U * sizeof(float));
  EXPECT_EQ(tensor.name(), "input");
}

TEST(Tensor, ScalarShapeHasOneElement) {
  const Tensor tensor("scalar", {}, TensorDataType::kInt64);
  EXPECT_EQ(tensor.element_count(), 1U);
  EXPECT_EQ(tensor.byte_size(), sizeof(int64_t));
}

TEST(Tensor, RejectsNonPositiveDimensions) {
  EXPECT_THROW(Tensor("bad", {1, 0, 3}, TensorDataType::kUInt8), std::invalid_argument);
  EXPECT_THROW(Tensor("bad", {1, -3}, TensorDataType::kUInt8), std::invalid_argument);
}

TEST(Tensor, AsSpanRoundTripsData) {
  Tensor tensor("blob", {4}, TensorDataType::kFloat32);
  auto span = tensor.as_span<float>();
  ASSERT_EQ(span.size(), 4U);
  span[0] = 1.5F;
  span[3] = 2.5F;

  const Tensor& const_tensor = tensor;
  EXPECT_FLOAT_EQ(const_tensor.as_span<float>()[0], 1.5F);
  EXPECT_FLOAT_EQ(const_tensor.as_span<float>()[3], 2.5F);
}

TEST(Tensor, AsSpanThrowsOnDtypeMismatch) {
  Tensor tensor("blob", {4}, TensorDataType::kFloat32);
  EXPECT_THROW(tensor.as_span<int64_t>(), std::logic_error);
}

}  // namespace
}  // namespace observatory::inference
