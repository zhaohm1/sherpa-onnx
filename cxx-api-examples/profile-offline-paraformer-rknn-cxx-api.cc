// Offline Paraformer profiler for RKNN and ONNX Runtime model files.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"
#include "sherpa-onnx/csrc/profiling.h"

namespace {

struct Options {
  std::string model;
  std::string tokens;
  std::string wav;
  std::string output_dir;
  std::string provider = "rknn";
  int32_t num_threads = 1;
  int32_t warmup = 2;
  int32_t repeat = 20;
};

void Usage(const char *program) {
  std::cerr << "Usage: " << program
            << " --model encoder.rknn,predictor.rknn,decoder.rknn"
               "|model.onnx"
               " --tokens tokens.txt --wav input.wav --output-dir DIR"
               " [--provider rknn|cpu] [--num-threads N] [--warmup N]"
               " [--repeat N]\n"
               "Use three comma-separated RKNN files with --provider rknn, or"
               " one ONNX file with --provider cpu.\n";
}

bool ParseInt(const std::string &s, int32_t *value) {
  char *end = nullptr;
  long parsed = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || parsed < 0 || parsed > 1000000) {
    return false;
  }
  *value = static_cast<int32_t>(parsed);
  return true;
}

bool ParseArgs(int argc, char **argv, Options *options) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      return false;
    }
    if (i + 1 >= argc) {
      return false;
    }
    std::string value = argv[++i];
    if (key == "--model") {
      options->model = value;
    } else if (key == "--tokens") {
      options->tokens = value;
    } else if (key == "--wav") {
      options->wav = value;
    } else if (key == "--output-dir") {
      options->output_dir = value;
    } else if (key == "--provider") {
      options->provider = value;
    } else if (key == "--num-threads") {
      if (!ParseInt(value, &options->num_threads)) return false;
    } else if (key == "--warmup") {
      if (!ParseInt(value, &options->warmup)) return false;
    } else if (key == "--repeat") {
      if (!ParseInt(value, &options->repeat)) return false;
    } else {
      std::cerr << "Unknown option: " << key << "\n";
      return false;
    }
  }
  return !options->model.empty() && !options->tokens.empty() &&
         !options->wav.empty() && !options->output_dir.empty() &&
         options->num_threads > 0 && options->repeat > 0;
}

bool FileExists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

std::string Csv(const std::string &value) {
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') escaped += '"';
    escaped += c;
  }
  escaped += '"';
  return escaped;
}

double Percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  size_t rank = static_cast<size_t>(std::ceil(p * values.size()));
  rank = std::max<size_t>(1, rank);
  return values[std::min(rank, values.size()) - 1];
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    Usage(argv[0]);
    return 2;
  }

  if (options.provider != "rknn" && options.provider != "cpu") {
    std::cerr << "--provider must be rknn or cpu\n";
    return 2;
  }
  std::vector<std::string> model_files;
  if (options.provider == "rknn") {
    size_t start = 0;
    while (start <= options.model.size()) {
      size_t end = options.model.find(',', start);
      if (end == std::string::npos) end = options.model.size();
      model_files.push_back(options.model.substr(start, end - start));
      if (end == options.model.size()) break;
      start = end + 1;
    }
    if (model_files.size() != 3 || model_files[0].empty() ||
        model_files[1].empty() || model_files[2].empty()) {
      std::cerr << "With --provider rknn, --model must list exactly three "
                   "comma-separated RKNN files\n";
      return 2;
    }
  } else {
    if (options.model.find(',') != std::string::npos) {
      std::cerr << "With --provider cpu, --model must be one ONNX file\n";
      return 2;
    }
    model_files.push_back(options.model);
  }
  for (const auto &path : model_files) {
    if (!FileExists(path)) {
      std::cerr << "Model file not found: " << path << "\n";
      return 2;
    }
  }
  if (!FileExists(options.tokens) || !FileExists(options.wav)) {
    std::cerr << "Tokens or WAV file not found\n";
    return 2;
  }
  if (mkdir(options.output_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    std::perror("mkdir output directory");
    return 2;
  }

  const std::string module_csv = options.output_dir + "/module_profile.csv";
  setenv("SHERPA_ONNX_PROFILE", "1", 1);
  setenv("SHERPA_ONNX_PROFILE_OUTPUT", module_csv.c_str(), 1);

  using Clock = std::chrono::steady_clock;
  using namespace sherpa_onnx::cxx;  // NOLINT
  OfflineRecognizerConfig config;
  config.model_config.paraformer.model = options.model;
  config.model_config.tokens = options.tokens;
  config.model_config.provider = options.provider;
  config.model_config.model_type = "paraformer";
  config.model_config.num_threads = options.num_threads;
  config.model_config.debug = true;

  auto init_start = Clock::now();
  OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
  double model_init_s = std::chrono::duration<double>(Clock::now() - init_start).count();
  if (!recognizer.Get()) {
    std::cerr << "Failed to initialize the Paraformer recognizer with provider "
              << options.provider << "\n";
    return 1;
  }

  Wave wave = ReadWave(options.wav);
  if (wave.samples.empty() || wave.sample_rate <= 0) {
    std::cerr << "Failed to read a non-empty WAV: " << options.wav << "\n";
    return 1;
  }
  const double audio_s = static_cast<double>(wave.samples.size()) / wave.sample_rate;

  for (int32_t i = 0; i < options.warmup; ++i) {
    OfflineStream stream = recognizer.CreateStream();
    stream.AcceptWaveform(wave.sample_rate, wave.samples.data(), wave.samples.size());
    recognizer.Decode(&stream);
  }
  sherpa_onnx::ResetModuleProfileCounters();

  std::ofstream csv(options.output_dir + "/e2e.csv");
  if (!csv) {
    std::cerr << "Failed to create e2e.csv\n";
    return 1;
  }
  csv << "iteration,audio_s,e2e_s,rtf,text\n" << std::fixed << std::setprecision(6);
  std::vector<double> latencies;
  std::string reference_text;
  bool consistent_text = true;
  for (int32_t i = 1; i <= options.repeat; ++i) {
    const auto begin = Clock::now();
    OfflineStream stream = recognizer.CreateStream();
    stream.AcceptWaveform(wave.sample_rate, wave.samples.data(), wave.samples.size());
    recognizer.Decode(&stream);
    OfflineRecognizerResult result = recognizer.GetResult(&stream);
    const double elapsed_s = std::chrono::duration<double>(Clock::now() - begin).count();
    latencies.push_back(elapsed_s);
    if (i == 1) {
      reference_text = result.text;
    } else if (result.text != reference_text) {
      consistent_text = false;
    }
    csv << i << ',' << audio_s << ',' << elapsed_s << ',' << elapsed_s / audio_s
        << ',' << Csv(result.text) << '\n';
  }
  csv.close();

  struct rusage usage {};
  double peak_rss_mb = -1.0;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    peak_rss_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
  }
  std::ofstream summary(options.output_dir + "/summary.csv");
  summary << "provider,model_init_s,audio_s,warmup,repeat,num_threads,mean_e2e_s,p50_e2e_s,"
             "p95_e2e_s,max_e2e_s,mean_rtf,peak_rss_mb,text_consistent\n"
          << options.provider << ',' << std::fixed << std::setprecision(6)
          << model_init_s << ',' << audio_s
          << ',' << options.warmup << ',' << options.repeat << ','
          << options.num_threads << ','
          << std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                 latencies.size()
          << ',' << Percentile(latencies, 0.50) << ','
          << Percentile(latencies, 0.95) << ','
          << *std::max_element(latencies.begin(), latencies.end()) << ','
          << std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                 latencies.size() / audio_s
          << ',' << peak_rss_mb << ',' << (consistent_text ? "true" : "false")
          << '\n';

  std::cout << "Model init: " << model_init_s << " s\n"
            << "Audio duration: " << audio_s << " s\n"
            << "Runs: " << options.repeat << ", warmup: " << options.warmup
            << "\n"
            << "Mean E2E: "
            << std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                   latencies.size()
            << " s, p95: " << Percentile(latencies, 0.95) << " s\n"
            << "Text consistent: " << (consistent_text ? "yes" : "no") << "\n"
            << "Results: " << options.output_dir << "\n";
  return consistent_text ? 0 : 3;
}
