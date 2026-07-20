#pragma once

#include <expected>
#include <memory>
#include <string>
#include <unordered_map>

#include "observatory/configuration/Config.hpp"
#include "observatory/inference/IInferenceModel.hpp"
#include "observatory/postprocessing/IPostprocessor.hpp"
#include "observatory/preprocessing/IPreprocessor.hpp"

namespace observatory::configuration
{

    // The strategies making up one config-driven inference pipeline:
    // [preprocessor] -> [model] -> [postprocessor]. camera/ isn't wired in
    // yet - Config has no camera settings so far.
    struct Pipeline
    {
        std::unique_ptr<inference::IInferenceModel> model;
        std::unique_ptr<preprocessing::IPreprocessor> preprocessor;
        std::unique_ptr<postprocessing::IPostprocessor> postprocessor;
    };

    // Which model family `raw_metadata` (a backend's getMetadata() map)
    // describes, read from its "framework" key. The .onnx file is the only
    // source of truth for this - never overridden by Config - so an older
    // export that predates academy embedding it fails clearly here instead
    // of silently being guessed at. Exposed separately from buildPipeline()
    // so this decision is testable against a hand-built map, without a real
    // backend/model file.
    std::expected<std::string, std::string> ResolveFramework(const std::unordered_map<std::string, std::string> &raw_metadata);

    // Builds a Pipeline from `config`: loads the model at config.model_path,
    // asks ResolveFramework() what family it is, and dispatches to that
    // family's concrete model/preprocessor/postprocessor - reading the
    // model's own metadata (input size, class count, whether NMS is baked
    // in) to configure them, combined with config's user-supplied
    // thresholds. Only "yolo" is implemented today (see observatory/CLAUDE.md's
    // Interfaces section) - a second model family adds a branch here, not a
    // parallel factory.
    std::expected<Pipeline, std::string> buildPipeline(const Config &config);

} // namespace observatory::configuration
