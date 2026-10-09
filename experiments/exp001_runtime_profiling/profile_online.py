#!/usr/bin/env python3
"""E2E and public-API module timing for sherpa-onnx online ASR models.

Decode time is an aggregate: sherpa-onnx's public Python API does not expose
separate Fbank, encoder, decoder, joiner, state-update, or search timers.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx


def percentile(values, q):
    if q == 95 and len(values) < 20:
        return None
    return float(np.percentile(values, q)) if values else None


def read_wave(path):
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM WAV")
        rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0, rate


def make_recognizer(args):
    common = dict(tokens=args.tokens, num_threads=args.num_threads,
                  provider=args.provider, sample_rate=args.sample_rate,
                  feature_dim=80, decoding_method=args.decoding_method)
    if args.model_kind == "paraformer":
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=args.encoder, decoder=args.decoder, **common)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=args.encoder, decoder=args.decoder, joiner=args.joiner, **common)


def process(recognizer, audio, rate, args, utterance, iteration):
    timings = {k: [] for k in ("accept_waveform", "readiness_check", "decode_streams", "get_result")}
    chunk_rows = []
    t0 = time.perf_counter()
    stream = recognizer.create_stream()
    step = max(1, int(rate * args.chunk_seconds))
    first_result_latency = None
    for offset in range(0, len(audio), step):
        t = time.perf_counter()
        stream.accept_waveform(rate, audio[offset:offset + step])
        timings["accept_waveform"].append(time.perf_counter() - t)
        while True:
            t = time.perf_counter()
            ready = recognizer.is_ready(stream)
            timings["readiness_check"].append(time.perf_counter() - t)
            if not ready:
                break
            t = time.perf_counter()
            recognizer.decode_streams([stream])
            elapsed = time.perf_counter() - t
            timings["decode_streams"].append(elapsed)
            chunk_rows.append({"utterance": str(utterance), "iteration": iteration,
                               "decode_index": len(chunk_rows) + 1,
                               "input_chunk_index": offset // step + 1,
                               "decode_s": elapsed,
                               "input_chunk_audio_s": min(step, len(audio) - offset) / rate})
            t = time.perf_counter()
            result = recognizer.get_result(stream)
            timings["get_result"].append(time.perf_counter() - t)
            if first_result_latency is None and result.text:
                first_result_latency = time.perf_counter() - t0

    padding = np.zeros(int(rate * args.tail_padding_seconds), dtype=np.float32)
    t = time.perf_counter()
    stream.accept_waveform(rate, padding)
    stream.input_finished()
    timings["accept_waveform"].append(time.perf_counter() - t)
    while True:
        t = time.perf_counter()
        ready = recognizer.is_ready(stream)
        timings["readiness_check"].append(time.perf_counter() - t)
        if not ready:
            break
        t = time.perf_counter()
        recognizer.decode_streams([stream])
        elapsed = time.perf_counter() - t
        timings["decode_streams"].append(elapsed)
        chunk_rows.append({"utterance": str(utterance), "iteration": iteration,
                           "decode_index": len(chunk_rows) + 1,
                           "input_chunk_index": "tail_padding",
                           "decode_s": elapsed, "input_chunk_audio_s": args.tail_padding_seconds})
        t = time.perf_counter()
        result = recognizer.get_result(stream)
        timings["get_result"].append(time.perf_counter() - t)
        if first_result_latency is None and result.text:
            first_result_latency = time.perf_counter() - t0
    t = time.perf_counter()
    result = recognizer.get_result(stream)
    timings["get_result"].append(time.perf_counter() - t)
    total = time.perf_counter() - t0
    duration = len(audio) / rate
    row = {
        "utterance": str(utterance), "iteration": iteration,
        "audio_duration_s": duration, "sample_rate": rate,
        "text": result.text, "e2e_s": total,
        "first_nonempty_result_s": first_result_latency,
        "decode_calls": len(timings["decode_streams"]),
        "rtf": total / duration if duration else None,
    }
    for stage, values in timings.items():
        row[f"{stage}_total_s"] = sum(values)
        row[f"{stage}_calls"] = len(values)
        row[f"{stage}_mean_s"] = statistics.mean(values) if values else None
        row[f"{stage}_p50_s"] = percentile(values, 50)
        row[f"{stage}_p95_s"] = percentile(values, 95)
        row[f"{stage}_max_s"] = max(values) if values else None
    return row, chunk_rows


def system_info(args):
    info = {"python": sys.version, "platform": platform.platform(),
            "cpu_count_logical": os.cpu_count(), "num_threads_requested": args.num_threads,
            "provider_requested": args.provider, "npu": "not measured / unavailable in this host session",
            "power": "not measured", "temperature_c": None,
            "cpu_governors": [], "cpu_frequencies_khz": []}
    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"):
        try: info["cpu_governors"].append(p.read_text().strip())
        except OSError: pass
    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try: info["cpu_frequencies_khz"].append(int(p.read_text().strip()))
        except (OSError, ValueError): pass
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-kind", choices=("paraformer", "transducer"), required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--joiner", help="Required for transducer")
    p.add_argument("--provider", default="cpu")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--decoding-method", default="greedy_search")
    p.add_argument("--chunk-seconds", type=float, default=0.2)
    p.add_argument("--tail-padding-seconds", type=float, default=0.66)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("wav", nargs="+")
    args = p.parse_args()
    if args.model_kind == "transducer" and not args.joiner:
        p.error("--joiner is required for transducer")
    if args.chunk_seconds <= 0 or args.repeat < 1 or args.warmup < 0 or args.num_threads < 1:
        p.error("chunk-seconds and num-threads must be positive; repeat >= 1; warmup >= 0")
    for filename in [args.tokens, args.encoder, args.decoder, *([args.joiner] if args.joiner else []), *args.wav]:
        if not Path(filename).is_file():
            p.error(f"file does not exist: {filename}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    recognizer = make_recognizer(args)
    init_s = time.perf_counter() - t
    loaded_audio = []
    read_times = {}
    for path in args.wav:
        read_start = time.perf_counter()
        samples, rate = read_wave(path)
        read_times[str(path)] = time.perf_counter() - read_start
        loaded_audio.append((Path(path), samples, rate))
    for _, audio, rate in loaded_audio:
        for i in range(args.warmup):
            process(recognizer, audio, rate, args, "warmup", i + 1)
    rows = []
    chunk_rows = []
    for path, audio, rate in loaded_audio:
        for iteration in range(1, args.repeat + 1):
            utterance_row, utterance_chunks = process(recognizer, audio, rate, args, path, iteration)
            rows.append(utterance_row)
            chunk_rows.extend(utterance_chunks)

    with (out / "utterances.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (out / "chunks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(chunk_rows[0]) if chunk_rows else
                                ["utterance", "iteration", "decode_index", "input_chunk_index", "decode_s", "input_chunk_audio_s"])
        writer.writeheader(); writer.writerows(chunk_rows)
    e2e = [r["e2e_s"] for r in rows]
    chunks = [r["decode_streams_mean_s"] for r in rows if r["decode_calls"]]
    chunk_decode_s = [r["decode_s"] for r in chunk_rows]
    model_paths = {k: getattr(args, k) for k in ("tokens", "encoder", "decoder", "joiner") if getattr(args, k)}
    model_hashes = {}
    for key, filename in model_paths.items():
        digest = hashlib.sha256()
        with open(filename, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        model_hashes[key] = digest.hexdigest()
    metadata = {
        "experiment": "Exp001_runtime_profiling", "model_kind": args.model_kind,
        "model_files": model_paths, "model_sha256": model_hashes,
        "model_size_bytes": sum(Path(v).stat().st_size for v in {args.encoder, args.decoder, args.joiner} if v),
        "input_wav": args.wav, "provider_requested": args.provider,
        "wav_read_s": read_times,
        "decoding_method": args.decoding_method, "num_threads": args.num_threads,
        "chunk_seconds": args.chunk_seconds, "tail_padding_seconds": args.tail_padding_seconds,
        "warmup_per_utterance": args.warmup, "repeat_per_utterance": args.repeat,
        "recognizer_initialization_s": init_s, "sample_count": len(rows),
        "audio_duration_s_total_per_iteration": sum(len(a) / sr for _, a, sr in loaded_audio),
        "e2e_s": {"mean": statistics.mean(e2e), "p50": percentile(e2e, 50),
                  "p95": percentile(e2e, 95), "max": max(e2e)},
        "decode_streams_mean_per_call_s": {"mean": statistics.mean(chunks) if chunks else None,
            "p50": percentile(chunks, 50), "p95": percentile(chunks, 95), "max": max(chunks) if chunks else None},
        "per_decode_call_s": {"count": len(chunk_decode_s),
            "mean": statistics.mean(chunk_decode_s) if chunk_decode_s else None,
            "p50": percentile(chunk_decode_s, 50), "p95": percentile(chunk_decode_s, 95),
            "max": max(chunk_decode_s) if chunk_decode_s else None},
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1),
        "sherpa_onnx_version": getattr(sherpa_onnx, "__version__", "not exposed by installed package"),
        "system": system_info(args),
        "timing_scope": "E2E includes stream creation, chunk acceptance, decode, result retrieval and tail padding; WAV reading and recognizer initialization reported separately.",
        "module_boundary": "Public API timers only. decode_streams aggregates feature extraction, model inference (encoder/decoder/joiner as applicable), state/cache update and search. Separate internal module timing requires instrumentation in sherpa-onnx C++/RKNN runtime.",
    }
    (out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "initialization_s": init_s,
                      "e2e_s": metadata["e2e_s"], "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
