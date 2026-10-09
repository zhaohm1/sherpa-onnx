// Live ALSA profiling runner using the same model options as sherpa-onnx-alsa.

#include <signal.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/profiling.h"

namespace {

volatile sig_atomic_t g_stop = 0;

void HandleSignal(int /*sig*/) { g_stop = 1; }

std::string Csv(const std::string &value) {
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') escaped += '"';
    escaped += c;
  }
  escaped += '"';
  return escaped;
}

bool MakeDirectory(const std::string &path) {
  if (mkdir(path.c_str(), 0755) == 0) return true;
  if (errno == EEXIST) {
    std::cerr << "Output directory already exists: " << path << '\n';
    return false;
  }
  std::perror("mkdir profile output directory");
  return false;
}

}  // namespace

int main(int argc, char *argv[]) {
  using Clock = std::chrono::steady_clock;

  const char *usage_message = R"usage(
Usage:
  ./bin/profile-online-alsa \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.rknn \
    --decoder=/path/to/decoder.rknn \
    --joiner=/path/to/joiner.rknn \
    --provider=rknn \
    --num-threads=1 \
    --decoding-method=greedy_search \
    [--profile-output-dir=DIR] \
    [--profile-chunk-seconds=0.1] \
    [--profile-duration-seconds=0] \
    device_name

All OnlineRecognizer options and the positional ALSA device argument are the
same as sherpa-onnx-alsa. The runner captures until Ctrl+C by default. A
positive --profile-duration-seconds stops capture automatically.
Use arecord -l to list devices (for example: plughw:3,0).
)usage";

  sherpa_onnx::ParseOptions po(usage_message);
  sherpa_onnx::OnlineRecognizerConfig config;
  config.Register(&po);

  std::string output_dir = "profile-alsa";
  double chunk_seconds = 0.1;
  double duration_seconds = 0;
  po.Register("profile-output-dir", &output_dir,
              "Directory for ALSA profiling CSV files");
  po.Register("profile-chunk-seconds", &chunk_seconds,
              "Audio duration read per ALSA call; default is 0.1 seconds");
  po.Register("profile-duration-seconds", &duration_seconds,
              "Stop after this many seconds; 0 means until Ctrl+C");
  po.Read(argc, argv);

  if (po.NumArgs() != 1 || chunk_seconds <= 0 || duration_seconds < 0 ||
      !std::isfinite(chunk_seconds) || !std::isfinite(duration_seconds)) {
    po.PrintUsage();
    return 2;
  }
  if (!config.Validate()) {
    std::cerr << "Invalid OnlineRecognizer configuration\n";
    return 2;
  }
  if (!MakeDirectory(output_dir)) return 2;

  const std::string module_csv = output_dir + "/module_profile.csv";
  setenv("SHERPA_ONNX_PROFILE", "1", 1);
  setenv("SHERPA_ONNX_PROFILE_OUTPUT", module_csv.c_str(), 1);

  const auto init_begin = Clock::now();
  sherpa_onnx::OnlineRecognizer recognizer(config);
  const double model_init_s =
      std::chrono::duration<double>(Clock::now() - init_begin).count();

  const std::string device_name = po.GetArg(1);
  sherpa_onnx::Alsa alsa(device_name.c_str());
  const int32_t expected_sample_rate = config.feat_config.sampling_rate;
  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    std::cerr << "ALSA sample rate mismatch: expected " << expected_sample_rate
              << ", got " << alsa.GetExpectedSampleRate() << '\n';
    return 2;
  }

  std::ofstream chunks(output_dir + "/chunks.csv");
  std::ofstream utterances(output_dir + "/utterances.csv");
  if (!chunks || !utterances) {
    std::cerr << "Failed to create profiling CSV files in " << output_dir
              << '\n';
    return 2;
  }
  chunks << "chunk,read_s,audio_s,accept_s,decode_calls,decode_s,"
            "first_nonempty_result_s,endpoint,text\n";
  utterances << "segment,completion,audio_s,wall_s,first_nonempty_result_s,"
                "text\n";
  chunks << std::fixed << std::setprecision(6);
  utterances << std::fixed << std::setprecision(6);

  auto stream = recognizer.CreateStream();
  if (!stream) {
    std::cerr << "Failed to create online stream\n";
    return 1;
  }

  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);
  sherpa_onnx::ResetModuleProfileCounters();

  const int32_t read_frames = std::max<int32_t>(
      1, static_cast<int32_t>(std::llround(chunk_seconds *
                                           alsa.GetActualSampleRate())));
  auto run_begin = Clock::now();
  auto segment_begin = run_begin;
  double captured_audio_s = 0;
  double segment_audio_s = 0;
  double first_result_s = -1;
  uint64_t chunk_index = 0;
  uint64_t segment_index = 0;
  std::string last_text;

  std::cerr << "Recording from " << device_name << " at "
            << alsa.GetActualSampleRate() << " Hz; model init " << std::fixed
            << std::setprecision(3) << model_init_s << " s. Press Ctrl+C to stop.\n";

  while (!g_stop) {
    if (duration_seconds > 0 &&
        std::chrono::duration<double>(Clock::now() - run_begin).count() >=
            duration_seconds) {
      break;
    }

    const auto read_begin = Clock::now();
    const std::vector<float> &samples = alsa.Read(read_frames);
    const double read_s =
        std::chrono::duration<double>(Clock::now() - read_begin).count();
    if (samples.empty()) continue;  // ALSA recovery after an XRUN.

    const double audio_s =
        static_cast<double>(samples.size()) / expected_sample_rate;
    const auto accept_begin = Clock::now();
    stream->AcceptWaveform(expected_sample_rate, samples.data(), samples.size());
    const double accept_s =
        std::chrono::duration<double>(Clock::now() - accept_begin).count();

    int32_t decode_calls = 0;
    double decode_s = 0;
    while (recognizer.IsReady(stream.get())) {
      const auto decode_begin = Clock::now();
      recognizer.DecodeStream(stream.get());
      decode_s += std::chrono::duration<double>(Clock::now() - decode_begin).count();
      ++decode_calls;
    }

    captured_audio_s += audio_s;
    segment_audio_s += audio_s;
    std::string text = recognizer.GetResult(stream.get()).text;
    const auto now = Clock::now();
    if (first_result_s < 0 && !text.empty()) {
      first_result_s = std::chrono::duration<double>(now - segment_begin).count();
    }

    bool endpoint = recognizer.IsEndpoint(stream.get());
    if (endpoint && !config.model_config.paraformer.encoder.empty()) {
      std::vector<float> padding(expected_sample_rate, 0.0f);
      stream->AcceptWaveform(expected_sample_rate, padding.data(), padding.size());
      while (recognizer.IsReady(stream.get())) {
        const auto decode_begin = Clock::now();
        recognizer.DecodeStream(stream.get());
        decode_s += std::chrono::duration<double>(Clock::now() - decode_begin).count();
        ++decode_calls;
      }
      text = recognizer.GetResult(stream.get()).text;
    }

    ++chunk_index;
    chunks << chunk_index << ',' << read_s << ',' << audio_s << ',' << accept_s
           << ',' << decode_calls << ',' << decode_s << ',';
    if (first_result_s >= 0) chunks << first_result_s;
    chunks << ',' << (endpoint ? 1 : 0) << ',' << Csv(text) << '\n';

    if (!text.empty() && text != last_text) {
      std::cerr << "[partial] " << text << '\n';
      last_text = text;
    }

    if (endpoint) {
      const double wall_s =
          std::chrono::duration<double>(Clock::now() - segment_begin).count();
      utterances << ++segment_index << ",endpoint," << segment_audio_s << ','
                 << wall_s << ',';
      if (first_result_s >= 0) utterances << first_result_s;
      utterances << ',' << Csv(text) << '\n';
      recognizer.Reset(stream.get());
      segment_begin = Clock::now();
      segment_audio_s = 0;
      first_result_s = -1;
      last_text.clear();
    }
  }

  // Keep any unfinished segment as partial evidence; live capture has no EOF
  // signal, so it is deliberately not treated as a completed utterance.
  const std::string partial = recognizer.GetResult(stream.get()).text;
  if (segment_audio_s > 0 || !partial.empty()) {
    utterances << ++segment_index << ",stopped_partial," << segment_audio_s << ','
               << std::chrono::duration<double>(Clock::now() - segment_begin).count()
               << ',';
    if (first_result_s >= 0) utterances << first_result_s;
    utterances << ',' << Csv(partial) << '\n';
  }

  const double capture_wall_s =
      std::chrono::duration<double>(Clock::now() - run_begin).count();
  struct rusage resource_usage {};
  getrusage(RUSAGE_SELF, &resource_usage);
  std::ofstream summary(output_dir + "/summary.csv");
  summary << "model_init_s,capture_wall_s,captured_audio_s,chunk_count,"
             "segment_count,peak_rss_mb\n"
          << std::fixed << std::setprecision(6) << model_init_s << ','
          << capture_wall_s << ',' << captured_audio_s << ',' << chunk_index
          << ',' << segment_index << ',' << resource_usage.ru_maxrss / 1024.0
          << '\n';
  return 0;
}
