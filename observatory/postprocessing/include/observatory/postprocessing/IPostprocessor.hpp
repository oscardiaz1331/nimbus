#pragma once

#include <vector>
#include <expected>

#include "observatory/postprocessing/Detection.hpp"
#include "observatory/inference/Tensor.hpp"

namespace observatory::postprocessing {

// Postprocessing strategy: NMS, mask decoding, polygon extraction,
// cloud/sky % stats, connected components, confidence filtering, temporal
// smoothing.
class IPostprocessor {
 public:
  virtual ~IPostprocessor() = default;
  virtual std::expected<std::vector<std::vector<Detection>>, std::string> process(const std::vector<inference::Tensor>& outputs) = 0;

};

}  // namespace observatory::postprocessing
