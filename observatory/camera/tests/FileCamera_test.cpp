#include "observatory/camera/FileCamera.hpp"

#include <gtest/gtest.h>

#include <opencv5/opencv2/imgcodecs.hpp>

namespace observatory::camera {
namespace {

std::filesystem::path WriteTestImage(const std::string& filename, cv::Scalar color) {
  const auto path = std::filesystem::path(::testing::TempDir()) / filename;
  cv::Mat image(4, 4, CV_8UC3, color);
  cv::imwrite(path.string(), image);
  return path;
}

TEST(FileCamera, TriggerReturnsImagesInOrder) {
  const auto red = WriteTestImage("file_camera_red.png", cv::Scalar(0, 0, 255));
  const auto blue = WriteTestImage("file_camera_blue.png", cv::Scalar(255, 0, 0));
  FileCamera camera(FileCameraConfig{.image_paths = {red, blue}, .loop = false});

  auto first = camera.trigger();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 255));

  auto second = camera.trigger();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->at<cv::Vec3b>(0, 0), cv::Vec3b(255, 0, 0));
}

TEST(FileCamera, LoopsBackToStartWhenConfigured) {
  const auto red = WriteTestImage("file_camera_loop_red.png", cv::Scalar(0, 0, 255));
  FileCamera camera(FileCameraConfig{.image_paths = {red}, .loop = true});

  ASSERT_TRUE(camera.trigger().has_value());
  auto second = camera.trigger();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 255));
}

TEST(FileCamera, FailsCleanlyWhenSequenceExhaustedAndNotLooping) {
  const auto red = WriteTestImage("file_camera_noloop_red.png", cv::Scalar(0, 0, 255));
  FileCamera camera(FileCameraConfig{.image_paths = {red}, .loop = false});

  ASSERT_TRUE(camera.trigger().has_value());
  EXPECT_FALSE(camera.trigger().has_value());
}

TEST(FileCamera, FailsCleanlyWhenNoImagePathsConfigured) {
  FileCamera camera(FileCameraConfig{});
  EXPECT_FALSE(camera.trigger().has_value());
}

TEST(FileCamera, FailsCleanlyOnMissingFile) {
  FileCamera camera(FileCameraConfig{.image_paths = {"/nonexistent/path/does_not_exist.png"}});
  EXPECT_FALSE(camera.trigger().has_value());
}

}  // namespace
}  // namespace observatory::camera
