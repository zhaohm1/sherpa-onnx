#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_offline_paraformer_rknn.sh ENCODER.rknn,PREDICTOR.rknn,DECODER.rknn TOKENS WAV OUTPUT_DIR [RUNNER]

Run the locally built offline Paraformer RKNN profiler on an air-gapped Linux board.
The output directory must not already exist.
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage
  exit 2
fi

model_set=$1
tokens=$2
wav=$3
output_dir=$4
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner=${5:-"${repo_root}/build-linux-aarch64/bin/profile-offline-paraformer-rknn-cxx-api"}

if [[ ! -x "${runner}" ]]; then
  echo "Profiler executable not found or not executable: ${runner}" >&2
  exit 2
fi
if [[ ! -f "${tokens}" || ! -f "${wav}" ]]; then
  echo "Tokens or WAV file not found" >&2
  exit 2
fi
IFS=',' read -r -a model_files <<< "${model_set}"
if [[ ${#model_files[@]} -ne 3 ]]; then
  echo "Expected exactly three comma-separated RKNN model files" >&2
  exit 2
fi
for file in "${model_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "RKNN model file not found: ${file}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing output directory: ${output_dir}" >&2
  exit 2
fi
mkdir -p "${output_dir}"

{
  echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "runner=${runner}"
  echo "runner_sha256=$(sha256sum "${runner}" | awk '{print $1}')"
  echo "sherpa_onnx_git_commit=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -r /proc/device-tree/model ]]; then
    printf 'device_model='
    tr -d '\0' < /proc/device-tree/model
    echo
  fi
  echo "cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
  echo "memory_info:"
  grep -E '^(MemTotal|MemAvailable):' /proc/meminfo 2>/dev/null || true
  echo "cpu_frequency_governors_and_current_khz:"
  for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -r "${p}" ]] && echo "${p}=$(cat "${p}")"
  done
  for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    [[ -r "${p}" ]] && echo "${p}=$(cat "${p}")"
  done
  echo "devfreq_devices:"
  for d in /sys/class/devfreq/*; do
    [[ -d "${d}" ]] || continue
    for attr in name governor cur_freq; do
      [[ -r "${d}/${attr}" ]] && echo "${d}/${attr}=$(cat "${d}/${attr}")"
    done
  done
  echo "thermal_zones_millicelsius:"
  for p in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "${p}" ]] && echo "${p}=$(cat "${p}")"
  done
  echo "rknn_runtime_libraries:"
  ldconfig -p 2>/dev/null | grep -E 'librknn(rt|_api)' || true
  echo "rknpu_driver_version_files:"
  for p in /sys/kernel/debug/rknpu/version /proc/rknpu/version \
           /sys/devices/platform/*rknpu*/driver/module/version; do
    [[ -r "${p}" ]] && echo "${p}=$(cat "${p}")"
  done
  echo "power=not measured"
} > "${output_dir}/environment.txt"

{
  sha256sum "${model_files[@]}" "${tokens}" "${wav}"
} > "${output_dir}/input_sha256.txt"

set +e
"${runner}" --model "${model_set}" --tokens "${tokens}" --wav "${wav}" \
  --output-dir "${output_dir}" --warmup 2 --repeat 20 --num-threads 1 \
  2>&1 | tee "${output_dir}/run.log"
status=${PIPESTATUS[0]}
set -e
echo "runner_exit_code=${status}" >> "${output_dir}/environment.txt"
{
  echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "cpu_frequencies_khz:"
  for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    [[ -r "${p}" ]] && echo "${p}=$(cat "${p}")"
  done
  echo "devfreq:"
  for d in /sys/class/devfreq/*; do
    [[ -d "${d}" ]] || continue
    for attr in name cur_freq; do
      [[ -r "${d}/${attr}" ]] && echo "${d}/${attr}=$(cat "${d}/${attr}")"
    done
  done
  echo "thermal_zones_millicelsius:"
  for p in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "${p}" ]] && echo "${p}=$(cat "${p}")"
  done
} > "${output_dir}/device_state_after.txt"
exit "${status}"
