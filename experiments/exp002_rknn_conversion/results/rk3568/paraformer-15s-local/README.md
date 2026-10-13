# Paraformer-large-Chuan RK3568 15 秒 RKNN 模型

来源 checkpoint：`ASLP-lab/WSChuan-ASR/Paraformer-large-Chuan`，revision `aaae1ad1b28a0c390d1c6f4912bb75601f53dcc4`。

模型由 sherpa-onnx Paraformer RKNN 导出脚本生成固定 15 秒 encoder/predictor/decoder ONNX，再用 RKNN-Toolkit2 2.4.2a15、目标 `rk3568`、关闭量化转换。文件及 tokens 的 SHA-256 见 `SHA256SUMS`。详细日志、来源校验和复现步骤见 sherpa-onnx `experiments/exp002_rknn_conversion/README.md`。

本包尚未在 RK3568 目标板验证加载、NPU 执行、转写正确性或性能。此固定输入模型面向最长 15 秒音频。
