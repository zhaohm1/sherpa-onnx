// Fixed-WAV profiling counterpart of sherpa-onnx-offline.

#include <algorithm>
#include <cerrno>
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

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"
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
  const char *usage = "Profile one WAV with the same model options as sherpa-onnx-offline.\n"
      "Usage: profile-sherpa-onnx-offline [recognizer options] [--profile-output-dir DIR] "
      "[--profile-warmup N] [--profile-repeat N] [--profile-streams N] wav\n"
      "The input WAV is copied internally to N concurrent streams.\n";
  sherpa_onnx::ParseOptions po(usage);
  sherpa_onnx::OfflineRecognizerConfig config; config.Register(&po);
  std::string output_dir = "profile-offline";
  int32_t warmup = 1, repeat = 10, profile_streams = 1;
  po.Register("profile-output-dir", &output_dir, "New output directory for profile CSV files");
  po.Register("profile-warmup", &warmup, "Number of unmeasured batch warm-up runs");
  po.Register("profile-repeat", &repeat, "Number of measured batch runs");
  po.Register("profile-streams", &profile_streams,
              "Number of concurrent streams fed from the same WAV");
  po.Read(argc, argv);
  if (po.NumArgs() != 1 || warmup < 0 || repeat < 1 || profile_streams < 1 ||
      !config.Validate()) { po.PrintUsage(); return 2; }
  if (!CreateOutputDir(output_dir)) return 2;
  Input input; input.path = po.GetArg(1); bool ok = false;
  input.samples = sherpa_onnx::ReadWave(input.path, &input.rate, &ok);
  if (!ok) { std::fprintf(stderr, "Failed to read '%s'\n", input.path.c_str()); return 2; }
  SetEnv("SHERPA_ONNX_PROFILE", "1");
  const std::string module_path = output_dir + "/module_profile.csv";
  SetEnv("SHERPA_ONNX_PROFILE_OUTPUT", module_path);
  const auto init_begin = Clock::now();
  sherpa_onnx::OfflineRecognizer recognizer(config);
  const double model_init_s = std::chrono::duration<double>(Clock::now() - init_begin).count();
  auto decode_batch = [&]() {
    const auto begin = Clock::now();
    std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> streams;
    std::vector<sherpa_onnx::OfflineStream *> pointers;
    for (int32_t i = 0; i < profile_streams; ++i) {
      auto stream = recognizer.CreateStream();
      stream->AcceptWaveform(input.rate, input.samples.data(), input.samples.size());
      pointers.push_back(stream.get()); streams.push_back(std::move(stream));
    }
    recognizer.DecodeStreams(pointers.data(), pointers.size());
    return std::pair<double, std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>>>(
        std::chrono::duration<double>(Clock::now() - begin).count(), std::move(streams));
  };
  for (int32_t i = 0; i < warmup; ++i) decode_batch();
  sherpa_onnx::ResetModuleProfileCounters();
  std::ofstream rows(output_dir + "/runs.csv"), utterances(output_dir + "/utterances.csv"),
      summary(output_dir + "/summary.csv");
  if (!rows || !utterances || !summary) { std::fprintf(stderr, "Failed to create output CSV files\n"); return 2; }
  rows << "run,streams,audio_s,wall_s,rtf\n"; utterances << "run,stream,wav,text\n";
  summary << "provider,num_threads,decoding_method,streams,warmup,repeat,model_init_s,total_audio_s,mean_batch_wall_s,p50_batch_wall_s,p95_batch_wall_s,max_batch_wall_s,mean_rtf\n";
  const double total_audio = input.samples.size() / static_cast<double>(input.rate) * profile_streams;
  std::vector<double> elapsed;
  for (int32_t run = 1; run <= repeat; ++run) {
    auto result = decode_batch(); elapsed.push_back(result.first);
    rows << run << ',' << profile_streams << ',' << std::fixed << std::setprecision(6)
         << total_audio << ',' << result.first << ',' << result.first / total_audio << '\n';
    for (size_t i = 0; i < result.second.size(); ++i)
      utterances << run << ',' << (i + 1) << ',' << Csv(input.path) << ','
                 << Csv(result.second[i]->GetResult().text) << '\n';
  }
  auto sorted = elapsed; std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
  auto percentile = [&](double p) { return sorted[static_cast<size_t>(std::ceil(p * sorted.size()) - 1)]; };
  summary << config.model_config.provider << ',' << config.model_config.num_threads << ',' << config.decoding_method << ','
      << profile_streams << ',' << warmup << ',' << repeat << ',' << std::fixed << std::setprecision(6)
      << model_init_s << ',' << total_audio << ',' << mean << ',' << percentile(.50) << ',' << percentile(.95) << ','
      << sorted.back() << ',' << mean / total_audio << '\n';
  return 0;
}
