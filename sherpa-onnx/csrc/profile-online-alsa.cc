// ALSA profiling runner; one capture device feeds multiple recognizer streams.

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
#include <memory>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/profiling.h"

namespace {

volatile sig_atomic_t g_stop = 0;
using Clock = std::chrono::steady_clock;

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
  const char *usage_message = R"usage(
Usage:
  ./bin/profile-sherpa-onnx-alsa \
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
    [--profile-streams=1] \
    device_name

All OnlineRecognizer options and the final positional ALSA device argument
are the same as sherpa-onnx-alsa. The single device's audio is fed to N
concurrent streams. Capture continues until Ctrl+C by default. A positive
--profile-duration-seconds stops capture automatically.
Use arecord -l to list devices (for example: plughw:3,0).
)usage";

  sherpa_onnx::ParseOptions po(usage_message);
  sherpa_onnx::OnlineRecognizerConfig config;
  config.Register(&po);

  std::string output_dir = "profile-alsa";
  double chunk_seconds = 0.1;
  double duration_seconds = 0;
  int32_t profile_streams = 1;
  po.Register("profile-output-dir", &output_dir,
              "Directory for ALSA profiling CSV files");
  po.Register("profile-chunk-seconds", &chunk_seconds,
              "Audio duration read per ALSA call; default is 0.1 seconds");
  po.Register("profile-duration-seconds", &duration_seconds,
              "Stop after this many seconds; 0 means until Ctrl+C");
  po.Register("profile-streams", &profile_streams,
              "Number of concurrent streams fed from the same ALSA device");
  po.Read(argc, argv);

  if (po.NumArgs() != 1 || chunk_seconds <= 0 || duration_seconds < 0 ||
      profile_streams < 1 || !std::isfinite(chunk_seconds) ||
      !std::isfinite(duration_seconds)) {
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
  chunks << "chunk,stream,device,read_s,audio_s,accept_s,decode_calls,"
            "decode_batch_s,first_nonempty_result_s,endpoint,text\n";
  utterances << "stream,device,segment,completion,audio_s,wall_s,"
                "first_nonempty_result_s,text\n";
  chunks << std::fixed << std::setprecision(6);
  utterances << std::fixed << std::setprecision(6);

  std::vector<std::unique_ptr<sherpa_onnx::OnlineStream>> streams;
  for (int32_t i = 0; i < profile_streams; ++i) {
    auto stream = recognizer.CreateStream();
    if (!stream) {
      std::cerr << "Failed to create online stream " << (i + 1) << '\n';
      return 1;
    }
    streams.push_back(std::move(stream));
  }
  std::vector<double> segment_audio_s(profile_streams, 0);
  std::vector<double> first_result_s(profile_streams, -1);
  std::vector<uint64_t> segment_count(profile_streams, 0);
  std::vector<uint64_t> chunk_count(profile_streams, 0);
  std::vector<std::string> last_text(profile_streams);

  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);
  sherpa_onnx::ResetModuleProfileCounters();

  const int32_t read_frames = std::max<int32_t>(
      1, static_cast<int32_t>(std::llround(chunk_seconds *
                                           alsa.GetActualSampleRate())));
  const auto run_begin = Clock::now();
  std::vector<Clock::time_point> segment_begin(profile_streams, run_begin);
  double captured_audio_s = 0;
  uint64_t chunk_index = 0;

  std::cerr << "Recording from " << device_name << " at "
            << alsa.GetActualSampleRate() << " Hz; feeding " << profile_streams
            << " streams; model init " << std::fixed << std::setprecision(3)
            << model_init_s << " s. Press Ctrl+C to stop.\n";

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
    if (samples.empty()) continue;

    const double audio_s =
        static_cast<double>(samples.size()) / expected_sample_rate;
    captured_audio_s += audio_s;
    ++chunk_index;
    std::vector<double> accept_s(profile_streams, 0);
    for (int32_t i = 0; i < profile_streams; ++i) {
      const auto accept_begin = Clock::now();
      streams[i]->AcceptWaveform(expected_sample_rate, samples.data(),
                                 samples.size());
      accept_s[i] = std::chrono::duration<double>(Clock::now() - accept_begin)
                        .count();
      segment_audio_s[i] += audio_s;
    }

    std::vector<int32_t> decode_calls(profile_streams, 0);
    std::vector<double> decode_batch_s(profile_streams, 0);
    for (;;) {
      std::vector<sherpa_onnx::OnlineStream *> ready;
      std::vector<int32_t> ready_indices;
      for (int32_t i = 0; i < profile_streams; ++i) {
        if (recognizer.IsReady(streams[i].get())) {
          ready.push_back(streams[i].get());
          ready_indices.push_back(i);
        }
      }
      if (ready.empty()) break;
      const auto decode_begin = Clock::now();
      recognizer.DecodeStreams(ready.data(), ready.size());
      const double elapsed =
          std::chrono::duration<double>(Clock::now() - decode_begin).count();
      for (int32_t i : ready_indices) {
        ++decode_calls[i];
        decode_batch_s[i] += elapsed;
      }
    }

    std::vector<bool> endpoint(profile_streams, false);
    std::vector<sherpa_onnx::OnlineStream *> tail_streams;
    std::vector<int32_t> tail_indices;
    if (!config.model_config.paraformer.encoder.empty()) {
      for (int32_t i = 0; i < profile_streams; ++i) {
        if (recognizer.IsEndpoint(streams[i].get())) {
          endpoint[i] = true;
          tail_streams.push_back(streams[i].get());
          tail_indices.push_back(i);
        }
      }
    }
    if (!tail_streams.empty()) {
      std::vector<float> padding(expected_sample_rate, 0.0f);
      for (auto *stream : tail_streams) {
        stream->AcceptWaveform(expected_sample_rate, padding.data(),
                               padding.size());
      }
      for (;;) {
        std::vector<sherpa_onnx::OnlineStream *> ready;
        std::vector<int32_t> ready_indices;
        for (size_t j = 0; j < tail_streams.size(); ++j) {
          if (recognizer.IsReady(tail_streams[j])) {
            ready.push_back(tail_streams[j]);
            ready_indices.push_back(tail_indices[j]);
          }
        }
        if (ready.empty()) break;
        const auto decode_begin = Clock::now();
        recognizer.DecodeStreams(ready.data(), ready.size());
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - decode_begin).count();
        for (int32_t i : ready_indices) {
          ++decode_calls[i];
          decode_batch_s[i] += elapsed;
        }
      }
    }

    const auto now = Clock::now();
    for (int32_t i = 0; i < profile_streams; ++i) {
      std::string text = recognizer.GetResult(streams[i].get()).text;
      if (first_result_s[i] < 0 && !text.empty()) {
        first_result_s[i] =
            std::chrono::duration<double>(now - segment_begin[i]).count();
      }
      if (!endpoint[i]) endpoint[i] = recognizer.IsEndpoint(streams[i].get());

      chunks << chunk_index << ',' << (i + 1) << ',' << Csv(device_name) << ','
             << read_s << ',' << audio_s << ',' << accept_s[i] << ','
             << decode_calls[i] << ',' << decode_batch_s[i] << ',';
      if (first_result_s[i] >= 0) chunks << first_result_s[i];
      chunks << ',' << (endpoint[i] ? 1 : 0) << ',' << Csv(text) << '\n';

      if (!text.empty() && text != last_text[i]) {
        std::cerr << "[stream " << (i + 1) << " partial] " << text << '\n';
        last_text[i] = text;
      }
      if (endpoint[i]) {
        const double wall_s = std::chrono::duration<double>(
                                  now - segment_begin[i])
                                  .count();
        utterances << (i + 1) << ',' << Csv(device_name) << ','
                   << ++segment_count[i] << ",endpoint," << segment_audio_s[i]
                   << ',' << wall_s << ',';
        if (first_result_s[i] >= 0) utterances << first_result_s[i];
        utterances << ',' << Csv(text) << '\n';
        recognizer.Reset(streams[i].get());
        segment_begin[i] = Clock::now();
        segment_audio_s[i] = 0;
        first_result_s[i] = -1;
        last_text[i].clear();
      }
      ++chunk_count[i];
    }
  }

  g_stop = 1;
  for (int32_t i = 0; i < profile_streams; ++i) {
    const std::string partial = recognizer.GetResult(streams[i].get()).text;
    if (segment_audio_s[i] > 0 || !partial.empty()) {
      utterances << (i + 1) << ',' << Csv(device_name) << ','
                 << ++segment_count[i] << ",stopped_partial," << segment_audio_s[i]
                 << ',' << std::chrono::duration<double>(Clock::now() -
                                                          segment_begin[i])
                                .count()
                 << ',';
      if (first_result_s[i] >= 0) utterances << first_result_s[i];
      utterances << ',' << Csv(partial) << '\n';
    }
  }

  const double capture_wall_s =
      std::chrono::duration<double>(Clock::now() - run_begin).count();
  struct rusage resource_usage {};
  getrusage(RUSAGE_SELF, &resource_usage);
  std::ofstream summary(output_dir + "/summary.csv");
  summary << "device,stream,model_init_s,capture_wall_s,captured_audio_s,"
             "chunk_count,segment_count,peak_rss_mb\n"
          << std::fixed << std::setprecision(6);
  for (int32_t i = 0; i < profile_streams; ++i) {
    summary << Csv(device_name) << ',' << (i + 1) << ',' << model_init_s << ','
            << capture_wall_s << ',' << captured_audio_s << ',' << chunk_count[i]
            << ',' << segment_count[i] << ','
            << resource_usage.ru_maxrss / 1024.0 << '\n';
  }
  return 0;
}
