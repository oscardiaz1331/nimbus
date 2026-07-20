#include "observatory/configuration/ConfigLoader.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace observatory::configuration {
namespace {

std::filesystem::path WriteTestConfig(const std::string& filename, const std::string& contents) {
  const auto path = std::filesystem::path(::testing::TempDir()) / filename;
  std::ofstream(path) << contents;
  return path;
}

TEST(ConfigLoader, LoadsModelPathFromValidYaml) {
  const auto path = WriteTestConfig("config_loader_valid.yaml",
                                     "%YAML:1.0\n---\nmodel_path: \"/models/yolo.onnx\"\n");

  const auto result = loadConfig(path);
  ASSERT_TRUE(result.has_value()) << result.error();
  EXPECT_EQ(result->model_path, "/models/yolo.onnx");
}

TEST(ConfigLoader, FailsCleanlyOnMissingFile) {
  const auto result = loadConfig("/nonexistent/path/config.yaml");
  EXPECT_FALSE(result.has_value());
}

TEST(ConfigLoader, FailsCleanlyWhenModelPathKeyIsMissing) {
  const auto path = WriteTestConfig("config_loader_missing_key.yaml", "%YAML:1.0\n---\nother_key: 1\n");

  const auto result = loadConfig(path);
  EXPECT_FALSE(result.has_value());
}

// The example config.yaml at the repo root (see its own comments for what
// each field means) should always be a valid, loadable Config - this catches
// it drifting out of sync with Config's schema (a renamed/removed field,
// etc.) rather than only finding out when someone tries to run the app.
TEST(ConfigLoader, ExampleRepoRootConfigIsValid) {
  const auto path = std::filesystem::path(OBSERVATORY_ROOT_DIR) / "config.yaml";
  ASSERT_TRUE(std::filesystem::exists(path)) << path << " not found";

  const auto result = loadConfig(path);
  ASSERT_TRUE(result.has_value()) << result.error();
  EXPECT_FALSE(result->model_path.empty());
  EXPECT_EQ(result->backend, "onnx-cpu");
}

}  // namespace
}  // namespace observatory::configuration
