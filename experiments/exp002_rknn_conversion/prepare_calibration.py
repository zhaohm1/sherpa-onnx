#!/usr/bin/env python3
"""Create RKNN calibration NPYs by running real WAVs through float ONNX graphs.

Currently supported adapters are the fixed-shape Paraformer triplet and the
chunk=32 Zipformer Transducer triplet used by Exp002.
"""

import argparse
import hashlib
import json
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly


SAMPLE_RATE = 16000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError("empty WAV")
    samples = samples.mean(axis=1)
    if source_rate != SAMPLE_RATE:
        divisor = int(np.gcd(source_rate, SAMPLE_RATE))
        samples = resample_poly(
            samples, SAMPLE_RATE // divisor, source_rate // divisor
        ).astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("WAV contains NaN or Inf")
    return samples, int(source_rate)


def compute_fbank(samples: np.ndarray, *, paraformer: bool) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.mel_opts.num_bins = 80
    if paraformer:
        # Match sherpa-onnx OfflineParaformerRecognizer::InitFeatConfig.
        opts.frame_opts.snip_edges = True
        opts.frame_opts.window_type = "hamming"
        opts.mel_opts.high_freq = 0
        samples = samples * 32768.0
    else:
        # Match sherpa-onnx's default online Zipformer feature configuration.
        opts.frame_opts.snip_edges = False
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, samples.tolist())
    fbank.input_finished()
    if fbank.num_frames_ready == 0:
        raise ValueError("Fbank produced no frames")
    return np.stack(
        [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
    ).astype(np.float32, copy=False)


def lfr(features: np.ndarray) -> np.ndarray:
    width, shift = 7, 6
    # sherpa-onnx ApplyLfr: centered windows, repeating edge frames; the
    # number of outputs is 1 + floor((num_input_frames - 1) / shift).
    count = 1 + (len(features) - 1) // shift
    centers = np.arange(count)[:, None] * shift
    idx = np.clip(centers + np.arange(width)[None, :] - (width - 1) // 2,
                  0, len(features) - 1)
    return np.ascontiguousarray(features[idx].reshape(count, -1), dtype=np.float32)


def make_session(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def fixed_shape(meta) -> tuple[int, ...]:
    shape = []
    for dim in meta.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        elif dim in ("N", "batch", "batch_size"):
            shape.append(1)
        else:
            raise ValueError(f"Unsupported dynamic input {meta.name}: {meta.shape}")
    return tuple(shape)


def fit_time(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return np.ascontiguousarray(values[:length])
    if len(values) == 0:
        raise ValueError("cannot pad an empty tensor")
    return np.ascontiguousarray(
        np.concatenate([values, np.repeat(values[-1:], length - len(values), axis=0)])
    )


def cif_embeddings(alphas: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    accumulated = 0.0
    current = np.zeros((hidden.shape[-1],), dtype=np.float32)
    embeddings = []
    for alpha, frame in zip(alphas.reshape(-1), hidden):
        accumulated += float(alpha)
        if accumulated >= 1.0:
            overflow = accumulated - 1.0
            current += (float(alpha) - overflow) * frame
            embeddings.append(current)
            current = overflow * frame
            accumulated = overflow
        else:
            current += float(alpha) * frame
    if not embeddings:
        return np.zeros((0, hidden.shape[-1]), dtype=np.float32)
    return np.stack(embeddings).astype(np.float32, copy=False)


class TensorWriter:
    """Write one RKNN dataset line per real graph invocation."""

    def __init__(self, output_dir: Path, components: dict[str, int]):
        self.output_dir = output_dir
        self.lists_dir = output_dir / "lists"
        self.lists_dir.mkdir(parents=True, exist_ok=True)
        self.lines = {name: [] for name in components}
        self.component_inputs = components
        self.tensor_records = []

    def add(self, sample_id: str, component: str, inputs: list[np.ndarray]):
        expected = self.component_inputs[component]
        if len(inputs) != expected:
            raise ValueError(
                f"{component} expects {expected} inputs, got {len(inputs)}"
            )
        paths = []
        for index, value in enumerate(inputs):
            value = np.ascontiguousarray(value)
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{component} input {index} contains NaN or Inf")
            path = self.output_dir / f"{sample_id}_{component}_input{index}.npy"
            np.save(path, value)
            paths.append(path.resolve())
            self.tensor_records.append(
                {
                    "sample_id": sample_id,
                    "component": component,
                    "input_index": index,
                    "path": str(path.resolve()),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": sha256(path),
                }
            )
        self.lines[component].append(" ".join(map(str, paths)))

    def finish(self) -> dict[str, str]:
        result = {}
        for component, lines in self.lines.items():
            path = self.lists_dir / f"{component}.txt"
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            result[component] = str(path.resolve())
        return result


class ParaformerAdapter:
    def __init__(self, args):
        self.sessions = {
            "encoder": make_session(args.encoder),
            "predictor": make_session(args.predictor),
            "decoder": make_session(args.decoder),
        }
        self.inputs = {name: session.get_inputs() for name, session in self.sessions.items()}
        expected = {"encoder": 1, "predictor": 1, "decoder": 3}
        if {k: len(v) for k, v in self.inputs.items()} != expected:
            raise ValueError("Paraformer must have encoder=1, predictor=1, decoder=3 inputs")
        self.shapes = {k: [fixed_shape(x) for x in v] for k, v in self.inputs.items()}
        if self.shapes["encoder"][0][0] != 1 or self.shapes["encoder"][0][-1] != 560:
            raise ValueError(f"Expected encoder input [1,T,560], got {self.shapes['encoder'][0]}")
        if self.shapes["predictor"][0][0] != 1 or self.shapes["predictor"][0][-1] != 512:
            raise ValueError(f"Expected predictor input [1,T,512], got {self.shapes['predictor'][0]}")
        self.writer = None
        self.manifest_models = {
            name: {"path": str(Path(getattr(args, name)).resolve()), "sha256": sha256(Path(getattr(args, name)))}
            for name in self.sessions
        }

    def prepare(self, wav: Path, output_dir: Path, index: int):
        samples, rate = read_audio(wav)
        features = lfr(compute_fbank(samples, paraformer=True))
        enc_shape = self.shapes["encoder"][0]
        x = fit_time(features, enc_shape[1])[None, ...].astype(np.float32)
        enc = self.sessions["encoder"]
        hidden = enc.run(None, {self.inputs["encoder"][0].name: x})[0]

        pred_shape = self.shapes["predictor"][0]
        if tuple(hidden.shape) != pred_shape:
            raise ValueError(f"encoder output {hidden.shape} != predictor input {pred_shape}")
        pred = self.sessions["predictor"]
        alphas = pred.run(None, {self.inputs["predictor"][0].name: hidden})[0]
        embeddings = cif_embeddings(alphas, hidden[0])

        dec_shapes = self.shapes["decoder"]
        dec_t = dec_shapes[0][1]
        if dec_shapes[1][1] != dec_t or dec_shapes[2] != (dec_t,):
            raise ValueError(f"decoder input dimensions disagree: {dec_shapes}")
        enc_dec = fit_time(hidden[0], dec_t)[None, ...].astype(np.float32)
        emb = np.zeros((dec_t, dec_shapes[1][-1]), dtype=np.float32)
        n_valid = min(len(embeddings), dec_t)
        if embeddings.shape[-1] != emb.shape[-1]:
            raise ValueError("decoder acoustic embedding width mismatch")
        emb[:n_valid] = embeddings[:n_valid]
        mask_dtype = np.int32 if "int32" in self.inputs["decoder"][2].type else np.float32
        mask = np.zeros((dec_t,), dtype=mask_dtype)
        mask[:n_valid] = 1
        dec = self.sessions["decoder"]
        dec.run(
            None,
            {
                meta.name: value
                for meta, value in zip(self.inputs["decoder"], [enc_dec, emb[None, ...], mask])
            },
        )
        sample_id = f"{index:05d}_{wav.stem}"
        self.writer.add(sample_id, "encoder", [x])
        self.writer.add(sample_id, "predictor", [hidden.astype(np.float32)])
        self.writer.add(sample_id, "decoder", [enc_dec, emb[None, ...], mask])
        return {
            "id": sample_id,
            "wav": str(wav.resolve()),
            "wav_sha256": sha256(wav),
            "duration_seconds": len(samples) / SAMPLE_RATE,
            "source_sample_rate": rate,
            "lfr_frames_before_fixed_shape": len(features),
            "decoder_valid_embedding_count": n_valid,
            "graph_invocations": {"encoder": 1, "predictor": 1, "decoder": 1},
        }


class ZipformerAdapter:
    """Replay sherpa-onnx's chunk=32 Zipformer greedy ONNX data path."""

    def __init__(self, args):
        self.paths = {k: Path(getattr(args, k)) for k in ("encoder", "decoder", "joiner")}
        self.sessions = {k: make_session(v) for k, v in self.paths.items()}
        self.ins = {k: s.get_inputs() for k, s in self.sessions.items()}
        self.outs = {k: s.get_outputs() for k, s in self.sessions.items()}
        self.encoder_meta = self.sessions["encoder"].get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(self.encoder_meta["decode_chunk_len"])
        self.encoder_chunk = fixed_shape(self.ins["encoder"][0])
        if self.encoder_chunk[0] != 1 or self.encoder_chunk[-1] != 80:
            raise ValueError(f"Expected Zipformer encoder x=[1,T,80], got {self.encoder_chunk}")
        self.chunk_size = self.encoder_chunk[1]
        if self.chunk_size <= self.decode_chunk_len:
            raise ValueError("encoder ONNX window must exceed decode_chunk_len")
        self.state_input_names = [x.name for x in self.ins["encoder"][1:]]
        if len(self.state_input_names) != len(self.outs["encoder"]) - 1:
            raise ValueError("encoder cache input/output counts do not match")
        if self.ins["decoder"][0].type != "tensor(int64)":
            raise ValueError("Zipformer decoder input is expected to be int64 token context")
        self.writer = None
        self.chunk_count = 0
        self.models = {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in self.paths.items()
        }
        self.tokens_path = Path(args.tokens)
        self.unk_id = self._get_unk_id(self.tokens_path)

    @staticmethod
    def _get_unk_id(path: Path) -> int:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip().split()
                if len(fields) >= 2 and fields[0] == "<unk>":
                    return int(fields[1])
        return -1

    def _initial_cache(self):
        cache = {}
        for meta in self.ins["encoder"][1:]:
            shape = fixed_shape(meta)
            dtype = np.int64 if meta.type == "tensor(int64)" else np.float32
            # sherpa-onnx initializes all Zipformer2 encoder states to zero.
            cache[meta.name] = np.zeros(shape, dtype=dtype)
        return cache

    def _run_decoder(self, context: np.ndarray, sample_id: str):
        meta = self.ins["decoder"][0]
        output = self.sessions["decoder"].run(None, {meta.name: context})[0]
        self.writer.add(sample_id, "decoder", [context])
        return output

    def prepare(self, wav: Path, output_dir: Path, index: int):
        samples, rate = read_audio(wav)
        frames = compute_fbank(samples, paraformer=False)
        x_length = self.chunk_size
        n_frames = len(frames)
        cache = self._initial_cache()
        context_size = int(self.sessions["decoder"].get_modelmeta().custom_metadata_map["context_size"])
        if context_size != 2:
            raise ValueError(f"This greedy adapter expects context_size=2, got {context_size}")
        context = np.full((1, context_size), -1, dtype=np.int64)
        context[0, -1] = 0
        sample_prefix = f"{index:05d}_{wav.stem}"
        decoder_out = None
        invocation_counts = {"encoder": 0, "decoder": 0, "joiner": 0}
        chunk_idx = 0
        offset = 0

        # sherpa-onnx IsReady: processed_frames + T < NumFramesReady;
        # GetFrames then consumes T frames and advances by decode_chunk_len.
        while offset + x_length < n_frames:
            chunk_id = f"{sample_prefix}_chunk{chunk_idx:05d}"
            x = np.ascontiguousarray(frames[offset : offset + x_length][None, ...], dtype=np.float32)
            encoder_values = [x] + [cache[name] for name in self.state_input_names]
            values = self.sessions["encoder"].run(None, {
                meta.name: value for meta, value in zip(self.ins["encoder"], encoder_values)
            })
            self.writer.add(chunk_id, "encoder", encoder_values)
            invocation_counts["encoder"] += 1
            encoder_out = values[0]
            for in_name, out_meta, value in zip(self.state_input_names, self.outs["encoder"][1:], values[1:]):
                expected_out_name = f"new_{in_name}"
                if out_meta.name != expected_out_name:
                    raise ValueError(f"Unexpected encoder cache output order/name: {out_meta.name}, expected {expected_out_name}")
                cache[in_name] = value

            if decoder_out is None:
                decoder_out = self._run_decoder(context, f"{chunk_id}_decoder_init")
                invocation_counts["decoder"] += 1
            for t in range(encoder_out.shape[1]):
                enc_frame = np.ascontiguousarray(encoder_out[:, t, :], dtype=np.float32)
                join_ins = self.ins["joiner"]
                logits = self.sessions["joiner"].run(None, {
                    join_ins[0].name: enc_frame,
                    join_ins[1].name: decoder_out,
                })[0]
                self.writer.add(f"{chunk_id}_frame{t:03d}", "joiner", [enc_frame, decoder_out])
                invocation_counts["joiner"] += 1
                token = int(np.asarray(logits).reshape(-1).argmax())
                if token not in (0, self.unk_id):
                    context = np.concatenate([context[:, 1:], np.array([[token]], dtype=np.int64)], axis=1)
                    decoder_out = self._run_decoder(context, f"{chunk_id}_frame{t:03d}_decoder")
                    invocation_counts["decoder"] += 1
            offset += self.decode_chunk_len
            chunk_idx += 1
            self.chunk_count += 1

        if chunk_idx == 0:
            raise ValueError(
                f"audio produced {n_frames} Fbank frames; at least {x_length + 1} "
                "are required by the sherpa-onnx ready condition"
            )

        return {
            "id": sample_prefix,
            "wav": str(wav.resolve()),
            "wav_sha256": sha256(wav),
            "duration_seconds": len(samples) / SAMPLE_RATE,
            "source_sample_rate": rate,
            "fbank_frames": n_frames,
            "encoder_chunks": chunk_idx,
            "unprocessed_tail_frames": n_frames - offset,
            "graph_invocations": invocation_counts,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("paraformer", "zipformer"), required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True, help="Float encoder ONNX")
    parser.add_argument("--predictor", type=Path, help="Paraformer predictor ONNX")
    parser.add_argument("--decoder", type=Path, required=True, help="Float decoder ONNX")
    parser.add_argument("--joiner", type=Path, help="Zipformer joiner ONNX")
    parser.add_argument("--tokens", type=Path, help="Zipformer tokens.txt")
    args = parser.parse_args()

    if args.model == "paraformer" and not args.predictor:
        parser.error("--predictor is required for --model paraformer")
    if args.model == "zipformer" and (not args.joiner or not args.tokens):
        parser.error("--joiner and --tokens are required for --model zipformer")

    wavs = sorted(p for p in args.wav_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not wavs:
        parser.error(f"No WAV files found under {args.wav_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "paraformer":
        adapter = ParaformerAdapter(args)
        counts = {"encoder": 1, "predictor": 1, "decoder": 3}
        models = adapter.manifest_models
        preprocessing = "mono mean; polyphase resample 16 kHz; PCM-scale samples by 32768; KaldiNativeFbank 80 bins, dither=0, snip_edges=True, hamming window, high_freq=0; centered 7-frame/6-shift LFR with edge replication; fixed ONNX length by crop/repeat-last-frame padding; float ONNX encoder (CMVN inside graph)->predictor->CIF->decoder"
    else:
        adapter = ZipformerAdapter(args)
        counts = {
            "encoder": len(adapter.ins["encoder"]),
            "decoder": 1,
            "joiner": 2,
        }
        models = {**adapter.models, "tokens": {"path": str(args.tokens.resolve()), "sha256": sha256(args.tokens)}}
        preprocessing = f"mono mean; polyphase resample 16 kHz; normalized waveform samples; sherpa-onnx default KaldiNativeFbank 80 bins, dither=0, snip_edges=False; Zipformer2 chunk windows T={adapter.chunk_size}, shift={adapter.decode_chunk_len}; zero-initialized cache; float ONNX encoder->greedy decoder/joiner"
    writer = TensorWriter(args.output_dir, counts)
    adapter.writer = writer

    prepared, failures = [], []
    for index, wav in enumerate(wavs):
        try:
            prepared.append(adapter.prepare(wav, args.output_dir, index))
        except Exception as exc:
            failures.append({"wav": str(wav.resolve()), "error": f"{type(exc).__name__}: {exc}"})
    lists = writer.finish()
    manifest = {
        "framework": "Exp002 float-ONNX real-input calibration capture",
        "model_type": args.model,
        "sample_rate_hz": SAMPLE_RATE,
        "preprocessing_and_chain": preprocessing,
        "models": models,
        "counts": {"wav_files": len(wavs), "prepared_wavs": len(prepared), "failed_wavs": len(failures)},
        "calibration_lists": lists,
        "samples": prepared,
        "tensor_files": writer.tensor_records,
        "failures": failures,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    for component, path in lists.items():
        print(f"{component}: {path}")
    print(f"manifest: {manifest_path}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
