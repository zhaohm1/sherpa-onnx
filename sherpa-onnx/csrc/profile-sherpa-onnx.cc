// Fixed-WAV profiling counterpart of sherpa-onnx.

#include <cerrno>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <direct.h>
#endif

#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/profiling.h"
#include "sherpa-onnx/csrc/wave-reader.h"

namespace {
using Clock = std::chrono::steady_clock;
struct Input { std::string path; int32_t rate; std::vector<float> samples; };

std::string Csv(const std::string &value) {
  std::string out = "\"";
  for (char c : value) { if (c == '"') out += '"'; out += c; }
  return out + '"';
}

bool CreateOutputDir(const std::string &path) {
#if defined(_WIN32)
  if (_mkdir(path.c_str()) == 0) return true;
#else
  if (mkdir(path.c_str(), 0755) == 0) return true;
#endif
  if (errno == EEXIST) std::fprintf(stderr, "Output directory exists: %s\n", path.c_str());
  else std::perror("mkdir profile output directory");
  return false;
}
void SetEnv(const char *name, const std::string &value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}
}

int main(int argc, char *argv[]) {
  const char *usage = "Profile one WAV with the same model options as sherpa-onnx.\n"
      "Usage: profile-sherpa-onnx [recognizer options] [--profile-output-dir DIR] "
      "[--profile-warmup N] [--profile-repeat N] [--profile-streams N] wav\n"
      "The input WAV is copied internally to N concurrent streams.\n";
  sherpa_onnx::ParseOptions po(usage);
  sherpa_onnx::OnlineRecognizerConfig config;
  config.Register(&po);
  std::string output_dir = "profile-online";
  int32_t warmup = 1, repeat = 10, profile_streams = 1;
  po.Register("profile-output-dir", &output_dir, "New output directory for profile CSV files");
  po.Register("profile-warmup", &warmup, "Number of unmeasured batch warm-up runs");
  po.Register("profile-repeat", &repeat, "Number of measured batch runs");
  po.Register("profile-streams", &profile_streams,
              "Number of concurrent streams fed from the same WAV");
  po.Read(argc, argv);
  if (po.NumArgs() != 1 || warmup < 0 || repeat < 1 || profile_streams < 1 ||
      !config.Validate()) {
    po.PrintUsage(); return 2;
  }
  if (!CreateOutputDir(output_dir)) return 2;
  Input input; input.path = po.GetArg(1); bool ok = false;
  input.samples = sherpa_onnx::ReadWave(input.path, &input.rate, &ok);
  if (!ok) { std::fprintf(stderr, "Failed to read '%s'\n", input.path.c_str()); return 2; }
  SetEnv("SHERPA_ONNX_PROFILE", "1");
  const std::string module_path = output_dir + "/module_profile.csv";
  SetEnv("SHERPA_ONNX_PROFILE_OUTPUT", module_path);
  const auto init_begin = Clock::now();
  sherpa_onnx::OnlineRecognizer recognizer(config);
  const double model_init_s = std::chrono::duration<double>(Clock::now() - init_begin).count();
  if (warmup > 0) {
    for (int32_t i = 0; i < warmup; ++i) {
      std::vector<std::unique_ptr<sherpa_onnx::OnlineStream>> streams;
      std::vector<sherpa_onnx::OnlineStream *> ready;
      for (int32_t j = 0; j < profile_streams; ++j) {
        auto s = recognizer.CreateStream();
        s->AcceptWaveform(input.rate, input.samples.data(), input.samples.size());
        std::vector<float> padding(static_cast<size_t>(0.8 * input.rate), 0.0f);
        s->AcceptWaveform(input.rate, padding.data(), padding.size()); s->InputFinished();
        streams.push_back(std::move(s));
      }
      for (;;) {
        ready.clear();
        for (auto &s : streams) if (recognizer.IsReady(s.get())) ready.push_back(s.get());
        if (ready.empty()) break;
        recognizer.DecodeStreams(ready.data(), ready.size());
      }
    }
  }
  sherpa_onnx::ResetModuleProfileCounters();
  std::ofstream rows(output_dir + "/runs.csv");
  std::ofstream utterances(output_dir + "/utterances.csv");
  std::ofstream summary(output_dir + "/summary.csv");
  if (!rows || !utterances || !summary) { std::fprintf(stderr, "Failed to create output CSV files\n"); return 2; }
  rows << "run,streams,audio_s,wall_s,rtf\n";
  utterances << "run,stream,wav,text\n";
  summary << "provider,num_threads,decoding_method,streams,warmup,repeat,model_init_s,total_audio_s,mean_batch_wall_s,p50_batch_wall_s,p95_batch_wall_s,max_batch_wall_s,mean_rtf\n";
  std::vector<double> elapsed;
  const double audio_per_stream = input.samples.size() / static_cast<double>(input.rate);
  const double total_audio = audio_per_stream * profile_streams;
  for (int32_t run = 1; run <= repeat; ++run) {
    const auto begin = Clock::now();
    std::vector<std::unique_ptr<sherpa_onnx::OnlineStream>> streams;
    std::vector<sherpa_onnx::OnlineStream *> ready;
    for (int32_t j = 0; j < profile_streams; ++j) {
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(input.rate, input.samples.data(), input.samples.size());
      std::vector<float> padding(static_cast<size_t>(0.8 * input.rate), 0.0f);
      s->AcceptWaveform(input.rate, padding.data(), padding.size()); s->InputFinished();
      streams.push_back(std::move(s));
    }
    for (;;) {
      ready.clear();
      for (auto &s : streams) if (recognizer.IsReady(s.get())) ready.push_back(s.get());
      if (ready.empty()) break;
      recognizer.DecodeStreams(ready.data(), ready.size());
    }
    const double wall = std::chrono::duration<double>(Clock::now() - begin).count();
    elapsed.push_back(wall); rows << run << ',' << streams.size() << ',' << std::fixed << std::setprecision(6)
        << total_audio << ',' << wall << ',' << wall / total_audio << '\n';
    for (size_t i = 0; i < streams.size(); ++i)
      utterances << run << ',' << (i + 1) << ',' << Csv(input.path) << ','
                 << Csv(recognizer.GetResult(streams[i].get()).text) << '\n';
  }
  auto sorted = elapsed; std::sort(sorted.begin(), sorted.end());
  const double mean = total_audio ? std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size() : 0;
  auto percentile = [&](double p) { return sorted[static_cast<size_t>(std::ceil(p * sorted.size()) - 1)]; };
  summary << config.model_config.provider_config.provider << ',' << config.model_config.num_threads << ',' << config.decoding_method << ','
      << profile_streams << ',' << warmup << ',' << repeat << ',' << std::fixed << std::setprecision(6)
      << model_init_s << ',' << total_audio << ',' << mean << ',' << percentile(.50) << ',' << percentile(.95) << ','
      << sorted.back() << ',' << mean / total_audio << '\n';
  return 0;
}
