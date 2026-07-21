// Drives the full camera -> preprocessor -> model -> postprocessor pipeline
// (see observatory/CLAUDE.md) end-to-end through FileCamera, once per
// requested backend, and reports latency/FPS - the "benchmark-driven, not
// assumed" model/backend choice CLAUDE.md's Benchmark module section talks
// about. This is a first cut: FPS + latency avg/min/max only, not the full
// GPU memory/CPU/RAM/accuracy set that section eventually wants.
//
// model_path/backend/thresholds come from the same config.yaml (via
// ConfigLoader) the rest of the pipeline is meant to be driven by - not a
// separate ad hoc set of flags - so "which backend" here is answered the
// same way it already is everywhere else (see observatory/config.yaml).
// --backend overrides just that one field, to A/B backends against the same
// model/thresholds without editing the file each time. --images stays a CLI
// flag since Config has no camera/image-source field yet (see
// PipelineFactory.hpp).

#include "observatory/camera/FileCamera.hpp"
#include "observatory/configuration/ConfigLoader.hpp"
#include "observatory/configuration/PipelineFactory.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <numeric>
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

  // Skips files ending in "_mask" - web/data/images (the repo's only real
  // sky/cloud sample photos today) pairs each frame_NN.png with a
  // frame_NN_mask.png ground-truth mask; harmless everywhere else since no
  // unrelated dataset happens to use that suffix.
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
      if (path.stem().string().ends_with("_mask"))
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
  // context on what a caller actually experiences.
  void RunBenchmark(const std::string &backend_name, const configuration::Config &config, camera::FileCamera &camera,
                     int warmup_iterations, int iterations)
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
    }

    const LatencyStats infer_stats = Summarize(infer_ms);
    const LatencyStats total_stats = Summarize(total_ms);
    std::printf("  frames: %d, avg detections/frame: %.1f\n", iterations,
                static_cast<double>(total_detections) / static_cast<double>(iterations));
    std::printf("  infer   avg/min/max: %7.2f / %7.2f / %7.2f ms  (%.1f FPS)\n", infer_stats.avg_ms, infer_stats.min_ms,
                infer_stats.max_ms, infer_stats.fps);
    std::printf("  end2end avg/min/max: %7.2f / %7.2f / %7.2f ms  (%.1f FPS)\n", total_stats.avg_ms, total_stats.min_ms,
                total_stats.max_ms, total_stats.fps);
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
    std::fprintf(stderr, "usage: observatory_benchmark --config <config.yaml> --images <dir> "
                          "[--backend onnx-cpu,opencv] [--iterations N] [--warmup N]\n"
                          "  --config   path to a Config yaml (see observatory/config.yaml) - model_path and\n"
                          "             thresholds always come from here.\n"
                          "  --backend  overrides config's \"backend\" field for this run; comma-separate to\n"
                          "             benchmark several backends back to back. Defaults to whatever --config says.\n");
  }

} // namespace

int main(int argc, char **argv)
{
  std::string config_path, images_dir, backend_arg;
  int iterations = 30, warmup = 5;

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (arg == "--config")
      config_path = next();
    else if (arg == "--images")
      images_dir = next();
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

  if (config_path.empty() || images_dir.empty())
  {
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

    const std::vector<std::filesystem::path> image_paths = ListImages(images_dir);
    if (image_paths.empty())
    {
      std::fprintf(stderr, "no images found in \"%s\"\n", images_dir.c_str());
      return 1;
    }
    std::printf("loaded %zu images from %s\n", image_paths.size(), images_dir.c_str());
    std::printf("model: %s\n", base_config->model_path.c_str());

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
      RunBenchmark(backend_name, config, camera, warmup, iterations);
    }
  }
  catch (const std::exception &ex)
  {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }

  return 0;
}
