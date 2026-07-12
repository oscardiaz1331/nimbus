#include "observatory/inference/OnnxRuntimeBackend.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace observatory::inference {
namespace {

TEST(OnnxRuntimeBackend, ThrowsOnMissingFile) {
  EXPECT_THROW(OnnxRuntimeBackend backend("/nonexistent/path/model.onnx"), std::runtime_error);
}

TEST(OnnxRuntimeBackend, ThrowsOnNonOnnxExtension) {
  EXPECT_THROW(OnnxRuntimeBackend backend(__FILE__), std::runtime_error);
}

}  // namespace
}  // namespace observatory::inference
