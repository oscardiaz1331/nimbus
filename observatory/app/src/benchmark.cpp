// Drives the full camera -> preprocessor -> model -> postprocessor pipeline
// (see observatory/CLAUDE.md) end-to-end through FileCamera, once per
// requested backend, and reports latency/FPS plus a per-backend mIoU
// accuracy check - the "benchmark-driven, not assumed" model/backend choice
// CLAUDE.md's Benchmark module section talks about. Still not the full GPU
// memory/CPU/RAM set that section eventually wants.
//
// The mIoU check is a pipeline-correctness check, not a backend-vs-backend
// or model-vs-model comparison: --dataset points at a YOLO-layout dataset
// root (images/*.{png,jpg,...} + labels/*.txt, same layout as e.g.
// academy/datasets/merged_yolo/train/ - see ParseYoloSegLabels), and each
// backend's predicted mask is scored against that image's label file,
// independently - never against another backend's output. Every --backend
// entry loads the exact same .onnx weights, so if
// preprocessor->model->postprocessor is implemented correctly, every
// backend's mIoU should land near the same (near 1.0 if the labels are a
// clean fit) value; one backend diverging from the rest is what would
// point at a real bug in that path. Class-agnostic (union of every
// detection's mask, and every label polygon) since academy's
// cloud-segmentation models are nc: 1.
//
// model_path/backend/thresholds come from the same config.yaml (via
// ConfigLoader) the rest of the pipeline is meant to be driven by - not a
// separate ad hoc set of flags - so "which backend" here is answered the
// same way it already is everywhere else (see observatory/config.yaml).
// --backend overrides just that one field, to A/B backends against the same
// model/thresholds without editing the file each time. --dataset/--images
// stay CLI flags since Config has no camera/image-source field yet (see
// PipelineFactory.hpp).

#include "observatory/camera/FileCamera.hpp"
#include "observatory/configuration/ConfigLoader.hpp"
#include "observatory/configuration/PipelineFactory.hpp"
#include "observatory/postprocessing/Detection.hpp"
#include "observatory/preprocessing/IPreprocessor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace observatory;

namespace
{

  struct LatencyStats
  {
    double avg_ms = 0, min_ms = 0, max_ms = 0, fps = 0;
  };

  LatencyStats Summarize(const std::vector<double> &samples_ms)
  {
    LatencyStats stats;
    if (samples_ms.empty())
      return stats;
    stats.min_ms = *std::ranges::min_element(samples_ms);
    stats.max_ms = *std::ranges::max_element(samples_ms);
    stats.avg_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / static_cast<double>(samples_ms.size());
    stats.fps = stats.avg_ms > 0 ? 1000.0 / stats.avg_ms : 0.0;
    return stats;
  }

  // Combines every detection's per-instance mask into one binary "is-cloud"
  // canvas, image_size x image_size - the network's own letterboxed
  // coordinate space (see YoloSegPreprocessor::letterbox), not the original
  // frame. Class-agnostic union: which class_id a pixel belongs to doesn't
  // matter for the ground-truth mIoU check below, only whether the pipeline
  // called it foreground at all.
  cv::Mat BuildFrameMask(const std::vector<postprocessing::Detection> &detections, int canvas_size)
  {
    cv::Mat mask = cv::Mat::zeros(canvas_size, canvas_size, CV_8U);
    for (const postprocessing::Detection &detection : detections)
    {
      if (detection.seg_masks.empty())
        continue;
      // detection.box is already clipped to [0, canvas_size) by the
      // postprocessor, and seg_masks.front() is sized to detection.box (see
      // YoloSegPostprocessor::decodeMasks), so this ROI write is always
      // in-bounds and shape-matched.
      mask(detection.box) |= detection.seg_masks.front();
    }
    return mask;
  }

  // Intersection-over-union of two binary (0/255) masks. Both-empty counts
  // as perfect agreement (1.0) rather than an undefined 0/0 - "no cloud"
  // matching "no cloud" is still the pipeline getting it right.
  double ComputeIoU(const cv::Mat &a, const cv::Mat &b)
  {
    const int intersection = cv::countNonZero(a & b);
    const int union_count = cv::countNonZero(a | b);
    return union_count > 0 ? static_cast<double>(intersection) / static_cast<double>(union_count) : 1.0;
  }

  // One instance's outline, normalized [0, 1] against its image's own
  // width/height - straight off a YOLO-seg label line, no pixel space yet.
  using Polygon = std::vector<cv::Point2f>;

  // Parses one YOLO-seg label .txt (see e.g.
  // academy/datasets/merged_yolo/train/labels/*.txt): one instance polygon
  // per line, "class_id x1 y1 x2 y2 ... xn yn", x/y normalized to [0, 1].
  // class_id is read (to advance past it) and discarded - class-agnostic to
  // match BuildFrameMask(), since academy's cloud-segmentation models are
  // nc: 1. An existing-but-empty file is a valid label (an image with no
  // annotated instances at all, e.g. a clear-sky frame) and correctly comes
  // back as an empty vector, not an error.
  std::vector<Polygon> ParseYoloSegLabels(const std::filesystem::path &label_path)
  {
    std::ifstream file(label_path);
    std::vector<Polygon> polygons;
    std::string line;
    while (std::getline(file, line))
    {
      std::istringstream tokens(line);
      int class_id;
      if (!(tokens >> class_id))
        continue;
      Polygon polygon;
      float x, y;
      while (tokens >> x >> y)
        polygon.emplace_back(x, y);
      // A degenerate (<3-point) line isn't a real polygon - skip rather
      // than hand cv::fillPoly something meaningless.
      if (polygon.size() >= 3)
        polygons.push_back(std::move(polygon));
    }
    return polygons;
  }

  // image_path's label file under labels_dir (same stem, ".txt"), if one
  // exists - nullopt (not an empty vector) when there isn't one, so "no
  // annotation for this image" reads as "skip it for mIoU", distinct from
  // ParseYoloSegLabels's empty-file "annotated as having nothing in it".
  std::optional<std::vector<Polygon>> LoadGroundTruthLabel(const std::filesystem::path &labels_dir, const std::filesystem::path &image_path)
  {
    const std::filesystem::path label_path = labels_dir / (image_path.stem().string() + ".txt");
    if (!std::filesystem::exists(label_path))
      return std::nullopt;
    return ParseYoloSegLabels(label_path);
  }

  // Rasterizes a label's polygons straight into the same image_size x
  // image_size canvas BuildFrameMask() builds predictions in, applying the
  // identical original-image -> letterboxed-canvas transform
  // YoloSegPreprocessor::letterbox applied to this same frame (ctx.scale,
  // ctx.pad, read back off `ctx` rather than re-derived). No intermediate
  // full-resolution raster mask needed: normalized polygon coords go
  // straight to canvas pixel coords in one step (original_size undoes the
  // normalization, then scale/pad is the same letterbox every predicted
  // detection's box/mask already went through).
  cv::Mat RasterizeGroundTruth(const std::vector<Polygon> &polygons, const preprocessing::PreprocessContext &ctx, int canvas_size)
  {
    cv::Mat mask = cv::Mat::zeros(canvas_size, canvas_size, CV_8U);
    for (const Polygon &polygon : polygons)
    {
      std::vector<cv::Point> canvas_points;
      canvas_points.reserve(polygon.size());
      for (const cv::Point2f &normalized : polygon)
      {
        const float original_x = normalized.x * static_cast<float>(ctx.original_size.width);
        const float original_y = normalized.y * static_cast<float>(ctx.original_size.height);
        canvas_points.emplace_back(static_cast<int>(std::lround(original_x * ctx.scale + ctx.pad.x)),
                                    static_cast<int>(std::lround(original_y * ctx.scale + ctx.pad.y)));
      }
      cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{canvas_points}, cv::Scalar(255));
    }
    return mask;
  }

  std::vector<std::filesystem::path> ListImages(const std::filesystem::path &dir)
  {
    static constexpr std::array<std::string_view, 4> kImageExtensions{".png", ".jpg", ".jpeg", ".bmp"};
    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
      if (!entry.is_regular_file())
        continue;
      const std::filesystem::path &path = entry.path();
      if (std::ranges::find(kImageExtensions, path.extension().string()) == kImageExtensions.end())
        continue;
      paths.push_back(path);
    }
    std::ranges::sort(paths);
    return paths;
  }

  // Runs `iterations` frames through one backend's full pipeline. Times the
  // model->infer() call separately from the full frame (camera + preprocess
  // + infer + postprocess): infer is the only stage that differs between
  // backends, so it's the fair comparison; end-to-end is reported too, for
  // context on what a caller actually experiences. Timing stops right after
  // postprocess, same as before this function grew a mask-scoring tail -
  // the accuracy check below isn't part of what the latency numbers measure.
  //
  // `ground_truth_polygons` is parallel to the FileCamera's own image_paths
  // (one entry per frame, nullopt if that image has no labels/*.txt file -
  // see LoadGroundTruthLabel); `i % ground_truth_polygons.size()` recovers
  // which original image this iteration is on, matching FileCamera's own
  // looping. Each frame with a label contributes one IoU sample (including
  // a label file with zero polygons - a valid "nothing annotated here");
  // the mean is this backend's mIoU, printed independently of every other
  // backend - this is never a backend-vs-backend comparison, only
  // backend-vs-ground-truth.
  void RunBenchmark(const std::string &backend_name, const configuration::Config &config, camera::FileCamera &camera,
                     int warmup_iterations, int iterations, const std::vector<std::optional<std::vector<Polygon>>> &ground_truth_polygons)
  {
    std::printf("\n=== backend: %s ===\n", backend_name.c_str());

    auto pipeline_result = configuration::buildPipeline(config);
    if (!pipeline_result)
    {
      std::printf("  pipeline build failed: %s\n", pipeline_result.error().c_str());
      return;
    }
    configuration::Pipeline pipeline = std::move(*pipeline_result);

    pipeline.model->warmup(warmup_iterations);

    std::vector<double> infer_ms, total_ms;
    infer_ms.reserve(static_cast<std::size_t>(iterations));
    total_ms.reserve(static_cast<std::size_t>(iterations));
    std::size_t total_detections = 0;
    std::vector<double> iou_samples;

    for (int i = 0; i < iterations; ++i)
    {
      const auto frame_start = std::chrono::steady_clock::now();

      auto frame = camera.trigger();
      if (!frame)
      {
        std::printf("  camera.trigger() failed: %s\n", frame.error().c_str());
        return;
      }

      auto preprocessed = pipeline.preprocessor->process({*frame});
      if (!preprocessed)
      {
        std::printf("  preprocess failed: %s\n", preprocessed.error().c_str());
        return;
      }
      auto &[tensors, contexts] = *preprocessed;

      const auto infer_start = std::chrono::steady_clock::now();
      auto outputs = pipeline.model->infer(tensors);
      const auto infer_end = std::chrono::steady_clock::now();
      if (!outputs)
      {
        std::printf("  infer failed: %s\n", outputs.error().c_str());
        return;
      }

      auto detections = pipeline.postprocessor->process(*outputs);
      if (!detections)
      {
        std::printf("  postprocess failed: %s\n", detections.error().c_str());
        return;
      }
      if (!detections->empty())
        total_detections += (*detections)[0].size();

      const auto frame_end = std::chrono::steady_clock::now();
      infer_ms.push_back(std::chrono::duration<double, std::milli>(infer_end - infer_start).count());
      total_ms.push_back(std::chrono::duration<double, std::milli>(frame_end - frame_start).count());

      const std::size_t frame_index = static_cast<std::size_t>(i) % ground_truth_polygons.size();
      if (const auto &label = ground_truth_polygons[frame_index]; label.has_value())
      {
        const int canvas_size = static_cast<int>(tensors.front().shape().back());
        const cv::Mat predicted_mask = BuildFrameMask(detections->empty() ? std::vector<postprocessing::Detection>{} : (*detections)[0], canvas_size);
        const cv::Mat ground_truth = RasterizeGroundTruth(*label, contexts.front(), canvas_size);
        iou_samples.push_back(ComputeIoU(predicted_mask, ground_truth));
      }
    }

    const LatencyStats infer_stats = Summarize(infer_ms);
    const LatencyStats total_stats = Summarize(total_ms);
    std::printf("  frames: %d, avg detections/frame: %.1f\n", iterations,
                static_cast<double>(total_detections) / static_cast<double>(iterations));
    std::printf("  infer   avg/min/max: %7.2f / %7.2f / %7.2f ms  (%.1f FPS)\n", infer_stats.avg_ms, infer_stats.min_ms,
                infer_stats.max_ms, infer_stats.fps);
    std::printf("  end2end avg/min/max: %7.2f / %7.2f / %7.2f ms  (%.1f FPS)\n", total_stats.avg_ms, total_stats.min_ms,
                total_stats.max_ms, total_stats.fps);
    if (iou_samples.empty())
      std::printf("  mIoU: no ground-truth labels available (pass --dataset <root> with images/ + labels/)\n");
    else
    {
      const double mean_iou = std::accumulate(iou_samples.begin(), iou_samples.end(), 0.0) / static_cast<double>(iou_samples.size());
      std::printf("  mIoU vs ground truth: %.4f over %zu scored frames (of %d total)\n", mean_iou, iou_samples.size(), iterations);
    }
  }

  std::vector<std::string> SplitCommaList(const std::string &csv)
  {
    std::vector<std::string> parts;
    std::string current;
    for (char c : csv)
    {
      if (c == ',')
      {
        if (!current.empty())
          parts.push_back(std::exchange(current, ""));
      }
      else
        current += c;
    }
    if (!current.empty())
      parts.push_back(current);
    return parts;
  }

  void PrintUsage()
  {
    std::fprintf(stderr, "usage: observatory_benchmark --config <config.yaml> (--dataset <root> | --images <dir>) "
                          "[--backend onnx-cpu,opencv] [--iterations N] [--warmup N]\n"
                          "  --config   path to a Config yaml (see observatory/config.yaml) - model_path and\n"
                          "             thresholds always come from here.\n"
                          "  --dataset  a YOLO-layout dataset root: <root>/images/*.{png,jpg,...} plus a\n"
                          "             same-stem <root>/labels/*.txt per image (see\n"
                          "             academy/datasets/merged_yolo/train/ for the layout this expects).\n"
                          "             Enables the mIoU accuracy check; --images alone doesn't.\n"
                          "  --images   a plain image directory, no labels/ - runs the FPS/latency benchmark\n"
                          "             only, mIoU not computed. Mutually exclusive with --dataset.\n"
                          "  --backend  overrides config's \"backend\" field for this run; comma-separate to\n"
                          "             benchmark several backends back to back. Defaults to whatever --config says.\n");
  }

} // namespace

int main(int argc, char **argv)
{
  std::string config_path, images_dir, dataset_path, backend_arg;
  int iterations = 30, warmup = 5;

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (arg == "--config")
      config_path = next();
    else if (arg == "--images")
      images_dir = next();
    else if (arg == "--dataset")
      dataset_path = next();
    else if (arg == "--backend")
      backend_arg = next();
    else if (arg == "--iterations")
      iterations = std::stoi(next());
    else if (arg == "--warmup")
      warmup = std::stoi(next());
    else
    {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      PrintUsage();
      return 1;
    }
  }

  if (config_path.empty() || (images_dir.empty() && dataset_path.empty()))
  {
    PrintUsage();
    return 1;
  }
  if (!images_dir.empty() && !dataset_path.empty())
  {
    std::fprintf(stderr, "--images and --dataset are mutually exclusive\n");
    PrintUsage();
    return 1;
  }

  try
  {
    const auto base_config = configuration::loadConfig(config_path);
    if (!base_config)
    {
      std::fprintf(stderr, "error: %s\n", base_config.error().c_str());
      return 1;
    }

    // --dataset <root> means images live at <root>/images and each one's
    // label at <root>/labels/<same stem>.txt (see LoadGroundTruthLabel);
    // plain --images has no labels dir, so ground truth stays nullopt for
    // every frame and mIoU is simply not computed (see RunBenchmark).
    const bool have_dataset = !dataset_path.empty();
    const std::filesystem::path effective_images_dir = have_dataset ? std::filesystem::path(dataset_path) / "images" : std::filesystem::path(images_dir);
    const std::filesystem::path labels_dir = std::filesystem::path(dataset_path) / "labels";

    const std::vector<std::filesystem::path> image_paths = ListImages(effective_images_dir);
    if (image_paths.empty())
    {
      std::fprintf(stderr, "no images found in \"%s\"\n", effective_images_dir.string().c_str());
      return 1;
    }
    std::printf("loaded %zu images from %s\n", image_paths.size(), effective_images_dir.string().c_str());
    std::printf("model: %s\n", base_config->model_path.c_str());

    // Parallel to image_paths - nullopt where there's no labels/*.txt for
    // that image (or no --dataset at all). Loaded once here (not per
    // backend) since it's the same ground truth regardless of which
    // backend is under test.
    std::vector<std::optional<std::vector<Polygon>>> ground_truth_polygons;
    ground_truth_polygons.reserve(image_paths.size());
    std::size_t labels_found = 0;
    for (const std::filesystem::path &image_path : image_paths)
    {
      auto label = have_dataset ? LoadGroundTruthLabel(labels_dir, image_path) : std::nullopt;
      if (label)
        ++labels_found;
      ground_truth_polygons.push_back(std::move(label));
    }
    if (have_dataset)
      std::printf("ground-truth labels: %zu/%zu images\n", labels_found, image_paths.size());

    // No --backend override: run exactly what config.yaml says, once - the
    // same single-backend behavior the rest of the pipeline has.
    const std::vector<std::string> backend_names = backend_arg.empty() ? std::vector<std::string>{base_config->backend} : SplitCommaList(backend_arg);

    for (const std::string &backend_name : backend_names)
    {
      // A fresh FileCamera per backend, both starting at image_paths[0], so
      // every backend sees the exact same frame sequence - a fair
      // comparison instead of whichever frame the shared camera happened to
      // be on.
      camera::FileCamera camera(camera::FileCameraConfig{.image_paths = image_paths, .loop = true});
      configuration::Config config = *base_config;
      config.backend = backend_name;
      RunBenchmark(backend_name, config, camera, warmup, iterations, ground_truth_polygons);
    }
  }
  catch (const std::exception &ex)
  {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }

  return 0;
}
