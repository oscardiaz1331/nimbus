#include "observatory/preprocessing/YoloSegPreprocessor.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace observatory::preprocessing {
namespace {

YoloSegPreprocessor MakePreprocessor() {
  return YoloSegPreprocessor(YoloSegPreprocessorConfig{.stride = 32, .target_size = 640});
}

TEST(YoloSegPreprocessor, SquareInputMatchingTargetSizeNeedsNoScalingOrPadding) {
  auto pp = MakePreprocessor();
  cv::Mat img(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto result = pp.process({img});
  ASSERT_TRUE(result.has_value());
  const auto &ctx = result->second.front();
  EXPECT_FLOAT_EQ(ctx.scale, 1.0f);
  EXPECT_FLOAT_EQ(ctx.pad.x, 0.0f);
  EXPECT_FLOAT_EQ(ctx.pad.y, 0.0f);
}

TEST(YoloSegPreprocessor, LandscapeInputPadsTopAndBottomNotSides) {
  auto pp = MakePreprocessor();
  cv::Mat img(360, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto result = pp.process({img});
  ASSERT_TRUE(result.has_value());
  const auto &ctx = result->second.front();
  EXPECT_GT(ctx.pad.y, 0.0f);
  EXPECT_FLOAT_EQ(ctx.pad.x, 0.0f);
}

TEST(YoloSegPreprocessor, EmptyBatchFailsCleanly) {
  auto pp = MakePreprocessor();
  auto result = pp.process({});
  EXPECT_FALSE(result.has_value());
}

}  // namespace
}  // namespace observatory::preprocessing
