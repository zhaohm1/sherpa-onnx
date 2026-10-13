# Zipformer RKNN artifact

- Checkpoint: https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t
- HF revision: `e2382758de9a0219b4efe682b95af30b399db3b8`
- Sherpa-ONNX model unit: char+BPE; `tokens.txt` is from the same revision.
- Export source: ONNX triplet already published in the checkpoint repository, `exp/32/` (decode chunk length 32).
- Conversion target: RK3568, RKNN-Toolkit2 2.4.2a15, batch fixed to 1, `optimization_level=0`, `do_quantization=False` (float model; toolkit emits float16 weights/activations where supported).
- Encoder fixed input: `x=[1,39,80]`; symbolic `N` dimensions of the cache inputs are fixed to 1. ONNX metadata is copied into RKNN custom metadata.
- This is an offline host conversion artifact. It has not been run on a RK3568 NPU.

Conversion command from the repository root:

```bash
.venv/bin/python experiments/exp002_rknn_conversion/export_zipformer_rknn.py \
  --input-dir experiments/exp002_rknn_conversion/inputs/zipformer/32 \
  --output-dir experiments/exp002_rknn_conversion/results/rk3568/zipformer-32 \
  --target-platform rk3568
```

The RKNN Toolkit2 log contains repeated `Unkown op target: 0` diagnostics even though all three exports returned success. Toolkit simulator outputs were compared against ONNX Runtime using deterministic synthetic inputs: all output shapes matched; max absolute errors were 0.01165 (encoder), 0.0000482 (decoder), and 0.00422 (joiner); cosine similarities were at least 0.999997. These checks do not replace runtime validation on the RK3568 board.
