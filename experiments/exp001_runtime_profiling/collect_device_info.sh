#!/usr/bin/env bash

# Collect Linux/SoC, CPU, ALSA, and RKNN/NPU details for profiling records.
set -uo pipefail

section() {
  printf '\n=== %s ===\n' "$1"
}

section 'OS / kernel / architecture'
uname -a
if [[ -r /etc/os-release ]]; then
  cat /etc/os-release
fi

section 'CPU / SoC / device tree'
if [[ -r /proc/cpuinfo ]]; then
  grep -E '^(Model|Hardware|Revision)' /proc/cpuinfo | sort -u || true
fi
for path in /proc/device-tree/model /proc/device-tree/compatible; do
  if [[ -r "$path" ]]; then
    printf '%s\n' "--- $path"
    tr '\0' '\n' < "$path"
  fi
done

section 'ALSA capture devices'
if command -v arecord >/dev/null 2>&1; then
  arecord -l 2>&1 || true
else
  echo 'arecord is not installed'
fi

section 'RKNN devices / kernel driver'
ls -l /dev/rknpu* /dev/dri/* 2>/dev/null || true
if command -v lsmod >/dev/null 2>&1; then
  lsmod 2>/dev/null | grep -i rknpu || true
elif [[ -r /proc/modules ]]; then
  grep -i rknpu /proc/modules || true
fi
if dmesg >/dev/null 2>&1; then
  dmesg 2>/dev/null | grep -iE 'rknpu|rknn|npu' | tail -80 || true
else
  echo 'dmesg is not readable; rerun with sudo dmesg for kernel driver logs'
fi

section 'RKNN runtime libraries'
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -i rknn || true
fi
find /usr /lib /opt -type f \( -name 'librknnrt.so*' -o -name 'librknn_api.so*' \) \
  -print 2>/dev/null || true

section 'CPU and memory'
if command -v lscpu >/dev/null 2>&1; then
  lscpu 2>&1 || true
fi
if command -v free >/dev/null 2>&1; then
  free -h 2>&1 || true
fi

section 'NPU utilization interfaces'
find /sys \( -iname '*npu*' -o -iname '*rknpu*' \) -print 2>/dev/null \
  | head -100 || true
