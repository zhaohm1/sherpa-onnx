// Copyright (c) 2026 Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_PROFILING_H_
#define SHERPA_ONNX_CSRC_PROFILING_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <vector>

namespace sherpa_onnx {

inline bool IsModuleProfilingEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SHERPA_ONNX_PROFILE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

class ProfileCounter;

class ProfileRegistry {
 public:
  static ProfileRegistry &Get() {
    static ProfileRegistry registry;
    return registry;
  }

  void Register(ProfileCounter *counter) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_.push_back(counter);
  }

  void Reset();

  ~ProfileRegistry();

 private:
  std::mutex mutex_;
  std::vector<ProfileCounter *> counters_;
};

class ProfileCounter {
 public:
  explicit ProfileCounter(const char *name) : name_(name) {
    for (auto &bucket : histogram_) {
      bucket.store(0, std::memory_order_relaxed);
    }
    ProfileRegistry::Get().Register(this);
  }

  void Add(uint64_t duration_ns) {
    count_.fetch_add(1, std::memory_order_relaxed);
    total_ns_.fetch_add(duration_ns, std::memory_order_relaxed);

    uint64_t old_max = max_ns_.load(std::memory_order_relaxed);
    while (old_max < duration_ns &&
           !max_ns_.compare_exchange_weak(old_max, duration_ns,
                                          std::memory_order_relaxed)) {
    }

    size_t bucket = 0;
    for (uint64_t value = duration_ns; value > 1 && bucket + 1 < kNumBuckets;
         value >>= 1) {
      ++bucket;
    }
    histogram_[bucket].fetch_add(1, std::memory_order_relaxed);
  }

  const char *Name() const { return name_; }
  uint64_t Count() const { return count_.load(std::memory_order_relaxed); }
  uint64_t TotalNs() const { return total_ns_.load(std::memory_order_relaxed); }
  uint64_t MaxNs() const { return max_ns_.load(std::memory_order_relaxed); }

  void Reset() {
    count_.store(0, std::memory_order_relaxed);
    total_ns_.store(0, std::memory_order_relaxed);
    max_ns_.store(0, std::memory_order_relaxed);
    for (auto &bucket : histogram_) {
      bucket.store(0, std::memory_order_relaxed);
    }
  }

  uint64_t QuantileUpperNs(uint32_t percentile) const {
    const uint64_t count = Count();
    if (count == 0) {
      return 0;
    }
    const uint64_t target = (count * percentile + 99) / 100;
    uint64_t cumulative = 0;
    for (size_t i = 0; i != kNumBuckets; ++i) {
      cumulative += histogram_[i].load(std::memory_order_relaxed);
      if (cumulative >= target) {
        if (i + 1 == kNumBuckets) {
          return std::numeric_limits<uint64_t>::max();
        }
        return (uint64_t{1} << (i + 1)) - 1;
      }
    }
    return MaxNs();
  }

 private:
  static constexpr size_t kNumBuckets = 64;

  const char *name_;
  std::atomic<uint64_t> count_{0};
  std::atomic<uint64_t> total_ns_{0};
  std::atomic<uint64_t> max_ns_{0};
  std::array<std::atomic<uint64_t>, kNumBuckets> histogram_{};
};

inline void ProfileRegistry::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (ProfileCounter *counter : counters_) {
    counter->Reset();
  }
}

inline ProfileRegistry::~ProfileRegistry() {
  if (!IsModuleProfilingEnabled()) {
    return;
  }

  const char *path = std::getenv("SHERPA_ONNX_PROFILE_OUTPUT");
  std::ofstream file(path == nullptr || path[0] == '\0'
                         ? "sherpa-onnx-profile.csv"
                         : path);
  if (!file) {
    std::cerr << "Failed to write sherpa-onnx profiling output\n";
    return;
  }

  file << "stage,count,total_ms,mean_us,max_us,p50_upper_us,p95_upper_us\n";
  file << std::fixed << std::setprecision(3);
  for (const ProfileCounter *counter : counters_) {
    const uint64_t count = counter->Count();
    if (count == 0) {
      continue;
    }
    const uint64_t total_ns = counter->TotalNs();
    file << counter->Name() << ',' << count << ','
         << static_cast<double>(total_ns) / 1.0e6 << ','
         << static_cast<double>(total_ns) / count / 1.0e3 << ','
         << static_cast<double>(counter->MaxNs()) / 1.0e3 << ','
         << static_cast<double>(counter->QuantileUpperNs(50)) / 1.0e3 << ','
         << static_cast<double>(counter->QuantileUpperNs(95)) / 1.0e3 << '\n';
  }
}

class ScopedProfileTimer {
 public:
  explicit ScopedProfileTimer(ProfileCounter *counter)
      : counter_(counter), enabled_(IsModuleProfilingEnabled()) {
    if (enabled_) {
      start_ = Clock::now();
    }
  }

  ~ScopedProfileTimer() { Stop(); }

  ScopedProfileTimer(const ScopedProfileTimer &) = delete;
  ScopedProfileTimer &operator=(const ScopedProfileTimer &) = delete;

  void Stop() {
    if (enabled_) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - start_);
      counter_->Add(static_cast<uint64_t>(elapsed.count()));
      enabled_ = false;
    }
  }

 private:
  using Clock = std::chrono::steady_clock;
  ProfileCounter *counter_;
  bool enabled_;
  Clock::time_point start_{};
};

// Call after warm-up and before the measured window to exclude warm-up calls
// from the process-wide aggregate. Call only when inference threads are idle.
inline void ResetModuleProfileCounters() { ProfileRegistry::Get().Reset(); }

#define SHERPA_ONNX_PROFILE_CONCAT_IMPL(a, b) a##b
#define SHERPA_ONNX_PROFILE_CONCAT(a, b) SHERPA_ONNX_PROFILE_CONCAT_IMPL(a, b)
#define SHERPA_ONNX_PROFILE_TIMER_IMPL(var, name, id)                     \
  static ::sherpa_onnx::ProfileCounter *                                  \
      SHERPA_ONNX_PROFILE_CONCAT(_sherpa_profile_counter_, id) =          \
          new ::sherpa_onnx::ProfileCounter(name);                        \
  ::sherpa_onnx::ScopedProfileTimer var(                                  \
      SHERPA_ONNX_PROFILE_CONCAT(_sherpa_profile_counter_, id))
#define SHERPA_ONNX_PROFILE_TIMER(var, name) \
  SHERPA_ONNX_PROFILE_TIMER_IMPL(var, name, __COUNTER__)
#define SHERPA_ONNX_PROFILE_SCOPE_IMPL(name, id)                           \
  SHERPA_ONNX_PROFILE_TIMER_IMPL(                                         \
      SHERPA_ONNX_PROFILE_CONCAT(_sherpa_profile_timer_, id), name, id)
#define SHERPA_ONNX_PROFILE_SCOPE(name) \
  SHERPA_ONNX_PROFILE_SCOPE_IMPL(name, __COUNTER__)

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PROFILING_H_
