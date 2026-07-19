#pragma once

namespace observatory::inference {

// Common return type for IInferenceModel::metadata(). Deliberately minimal:
// a caller that only treats models generically (logging, benchmarking) sees
// this base. Concrete models override metadata() with a covariant return
// type exposing their own derived struct (see YoloModel::YoloModelMetadata)
// - a caller that already knows the concrete model type (e.g. a
// configuration/ factory that just constructed it) gets the richer struct
// directly, no dynamic_cast needed.
struct ModelMetadata {
  virtual ~ModelMetadata() = default;
};

}  // namespace observatory::inference
