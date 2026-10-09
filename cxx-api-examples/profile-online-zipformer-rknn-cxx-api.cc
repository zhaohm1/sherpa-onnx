// Online Zipformer profiler for RKNN and ONNX Runtime model files.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
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
  std::string encoder;
  std::string decoder;
  std::string joiner;
  std::string tokens;
  std::string wav;
  std::string output_dir;
  std::string provider = "rknn";
  int32_t num_threads = 1;
  int32_t warmup = 2;
  int32_t repeat = 20;
  double chunk_seconds = 0.2;
  double tail_padding_seconds = 0.3;
};

void Usage(const char *program) {
  std::cerr << "Usage: " << program
            << " --encoder encoder.onnx|encoder.rknn"
               " --decoder decoder.onnx|decoder.rknn"
               " --joiner joiner.onnx|joiner.rknn --tokens tokens.txt"
               " --wav input.wav --output-dir DIR [--provider rknn|cpu]"
               " [--num-threads N] [--warmup N]"
               " [--repeat N] [--chunk-seconds S]\n";
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

bool ParseDouble(const std::string &s, double *value) {
  char *end = nullptr;
  *value = std::strtod(s.c_str(), &end);
  return end != s.c_str() && *end == '\0' && std::isfinite(*value) &&
         *value > 0;
}

bool ParseArgs(int argc, char **argv, Options *options) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help" || key == "-h") return false;
    if (i + 1 >= argc) return false;
    std::string value = argv[++i];
    if (key == "--encoder") options->encoder = value;
    else if (key == "--decoder") options->decoder = value;
    else if (key == "--joiner") options->joiner = value;
    else if (key == "--tokens") options->tokens = value;
    else if (key == "--wav") options->wav = value;
    else if (key == "--output-dir") options->output_dir = value;
    else if (key == "--provider") options->provider = value;
    else if (key == "--num-threads") {
      if (!ParseInt(value, &options->num_threads)) return false;
    } else if (key == "--warmup") {
      if (!ParseInt(value, &options->warmup)) return false;
    } else if (key == "--repeat") {
      if (!ParseInt(value, &options->repeat)) return false;
    } else if (key == "--chunk-seconds") {
      if (!ParseDouble(value, &options->chunk_seconds)) return false;
    } else if (key == "--tail-padding-seconds") {
      if (!ParseDouble(value, &options->tail_padding_seconds)) return false;
    } else {
      std::cerr << "Unknown option: " << key << "\n";
      return false;
    }
  }
  return !options->encoder.empty() && !options->decoder.empty() &&
         !options->joiner.empty() && !options->tokens.empty() &&
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
  size_t rank = std::max<size_t>(1, static_cast<size_t>(std::ceil(p * values.size())));
  return values[std::min(rank, values.size()) - 1];
}

struct RunResult {
  double e2e_s = 0;
  double first_result_s = -1;
  std::string text;
  std::vector<double> decode_s;
  std::vector<int32_t> decode_trigger_chunk;
};

RunResult DecodeOne(sherpa_onnx::cxx::OnlineRecognizer *recognizer,
                    const sherpa_onnx::cxx::Wave &wave, double chunk_seconds,
                    double tail_padding_seconds,
                    std::ofstream *chunk_csv, int32_t iteration) {
  using Clock = std::chrono::steady_clock;
  RunResult run;
  const auto begin = Clock::now();
  auto stream = recognizer->CreateStream();
  const int32_t hop = std::max<int32_t>(1, static_cast<int32_t>(
      std::llround(wave.sample_rate * chunk_seconds)));
  int32_t feed_chunk = 0;

  auto drain = [&](const char *phase) {
    while (recognizer->IsReady(&stream)) {
      const auto decode_begin = Clock::now();
      recognizer->Decode(&stream);
      const double elapsed = std::chrono::duration<double>(Clock::now() - decode_begin).count();
      run.decode_s.push_back(elapsed);
      run.decode_trigger_chunk.push_back(feed_chunk);
      auto partial = recognizer->GetResult(&stream);
      if (run.first_result_s < 0 && !partial.text.empty()) {
        run.first_result_s = std::chrono::duration<double>(Clock::now() - begin).count();
      }
      if (chunk_csv != nullptr) {
        *chunk_csv << iteration << ',' << run.decode_s.size() << ',' << phase
                   << ',' << feed_chunk << ',' << std::fixed << std::setprecision(6)
                   << elapsed << '\n';
      }
    }
  };

  for (size_t offset = 0; offset < wave.samples.size(); offset += hop) {
    ++feed_chunk;
    const size_t count = std::min<size_t>(hop, wave.samples.size() - offset);
    stream.AcceptWaveform(wave.sample_rate, wave.samples.data() + offset,
                          static_cast<int32_t>(count));
    drain("audio");
  }

  const int32_t padding_n = static_cast<int32_t>(
      std::llround(wave.sample_rate * tail_padding_seconds));
  std::vector<float> padding(padding_n, 0.0f);
  stream.AcceptWaveform(wave.sample_rate, padding.data(), padding_n);
  stream.InputFinished();
  drain("tail");
  auto result = recognizer->GetResult(&stream);
  run.text = result.text;
  run.e2e_s = std::chrono::duration<double>(Clock::now() - begin).count();
  return run;
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    Usage(argv[0]);
    return 2;
  }
  for (const auto &path : {options.encoder, options.decoder, options.joiner,
                           options.tokens, options.wav}) {
    if (!FileExists(path)) {
      std::cerr << "Input file not found: " << path << "\n";
      return 2;
    }
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
  OnlineRecognizerConfig config;
  config.model_config.transducer.encoder = options.encoder;
  config.model_config.transducer.decoder = options.decoder;
  config.model_config.transducer.joiner = options.joiner;
  config.model_config.tokens = options.tokens;
  config.model_config.provider = options.provider;
  config.model_config.num_threads = options.num_threads;
  config.model_config.debug = true;
  config.decoding_method = "greedy_search";
  config.enable_endpoint = false;

  const auto init_begin = Clock::now();
  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  const double model_init_s = std::chrono::duration<double>(Clock::now() - init_begin).count();
  if (!recognizer.Get()) {
    std::cerr << "Failed to initialize the Zipformer recognizer with provider "
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
    DecodeOne(&recognizer, wave, options.chunk_seconds,
              options.tail_padding_seconds, nullptr, i + 1);
  }
  sherpa_onnx::ResetModuleProfileCounters();

  std::ofstream e2e(options.output_dir + "/utterances.csv");
  std::ofstream chunks(options.output_dir + "/chunks.csv");
  if (!e2e || !chunks) {
    std::cerr << "Failed to create result CSV files\n";
    return 1;
  }
  e2e << "iteration,audio_s,e2e_s,first_nonempty_result_s,decode_calls,rtf,text\n"
      << std::fixed << std::setprecision(6);
  chunks << "iteration,decode_index,phase,feed_chunk_index,decode_s\n";

  std::vector<double> latencies;
  std::vector<double> chunk_latencies;
  std::string reference_text;
  bool consistent_text = true;
  for (int32_t i = 1; i <= options.repeat; ++i) {
    RunResult run = DecodeOne(&recognizer, wave, options.chunk_seconds,
                              options.tail_padding_seconds, &chunks, i);
    latencies.push_back(run.e2e_s);
    chunk_latencies.insert(chunk_latencies.end(), run.decode_s.begin(),
                           run.decode_s.end());
    if (i == 1) reference_text = run.text;
    else if (run.text != reference_text) consistent_text = false;
    e2e << i << ',' << audio_s << ',' << run.e2e_s << ','
        << (run.first_result_s < 0 ? "" : std::to_string(run.first_result_s))
        << ',' << run.decode_s.size() << ',' << run.e2e_s / audio_s << ','
        << Csv(run.text) << '\n';
  }
  e2e.close();
  chunks.close();

  struct rusage usage {};
  double peak_rss_mb = -1.0;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    peak_rss_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
  }
  std::ofstream summary(options.output_dir + "/summary.csv");
  summary << "provider,model_init_s,audio_s,warmup,repeat,num_threads,chunk_seconds,"
             "mean_e2e_s,p50_e2e_s,p95_e2e_s,max_e2e_s,mean_decode_s,"
             "p50_decode_s,p95_decode_s,peak_rss_mb,text_consistent\n"
          << options.provider << ',' << std::fixed << std::setprecision(6)
          << model_init_s << ',' << audio_s
          << ',' << options.warmup << ',' << options.repeat << ','
          << options.num_threads << ',' << options.chunk_seconds << ','
          << std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                 latencies.size()
          << ',' << Percentile(latencies, 0.50) << ','
          << Percentile(latencies, 0.95) << ','
          << *std::max_element(latencies.begin(), latencies.end()) << ','
          << std::accumulate(chunk_latencies.begin(), chunk_latencies.end(), 0.0) /
                 std::max<size_t>(1, chunk_latencies.size())
          << ',' << (chunk_latencies.empty() ? 0 : Percentile(chunk_latencies, 0.50))
          << ',' << (chunk_latencies.empty() ? 0 : Percentile(chunk_latencies, 0.95))
          << ',' << peak_rss_mb << ',' << (consistent_text ? "true" : "false")
          << '\n';
  std::cout << "Model init: " << model_init_s << " s; audio duration: " << audio_s
            << " s; repeat: " << options.repeat << "; text consistent: "
            << (consistent_text ? "yes" : "no") << "\nResults: "
            << options.output_dir << "\n";
  return consistent_text ? 0 : 3;
}
