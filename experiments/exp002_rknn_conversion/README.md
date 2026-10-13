# RK3568 模型转 RKNN

状态：**转换产物已准备；目标板验证待完成**。本实验为 Exp001 profiling runner 准备同一目标芯片的 Paraformer 和 Zipformer 模型。模型文件、ONNX 文件、WAV、压缩包和日志均不纳入 Git；本 README、转换脚本和 SHA256 清单用于追溯。

## 目标和边界

- 芯片目标：RK3568。
- Paraformer：离线 `Paraformer-large-Chuan`，生成/复用 15 秒固定输入形状的 encoder/predictor/decoder RKNN 三件套。
- Zipformer：在线 Transducer，使用 checkpoint 仓库提供的 chunk=32 ONNX encoder/decoder/joiner，转换成 RKNN。
- 本机没有 RK3568 NPU。完成主机侧格式转换和 Toolkit 模拟器检查，不等于板端加载、NPU 执行、识别质量或性能验收。

## Paraformer-large-Chuan

- Checkpoint：[ASLP-lab/WSChuan-ASR/Paraformer-large-Chuan](https://huggingface.co/ASLP-lab/WSChuan-ASR/tree/main/Paraformer-large-Chuan)，HF revision `aaae1ad1b28a0c390d1c6f4912bb75601f53dcc4`。本机下载的 `model.pt` 为 2,642,208,221 bytes，SHA-256 `3cb8f6c514ada6029c3504f6629a4756a583fe801e56a383a095636dc4c3ee77`，与 HF LFS 元数据一致。
- 转换链：该 checkpoint 由仓库的 Paraformer RKNN 导出脚本导出为 15 秒定长 encoder/predictor/decoder ONNX，再由 RKNN-Toolkit2 2.4.2a15 转为 `rk3568`，未量化。上游导出脚本来自当前仓库 `scripts/paraformer/rknn/`；本地导出副本只对 encoder 的 `torch.load` 增加 `weights_only=True`，变更记录在 [`patches/paraformer-safe-torch-load.patch`](patches/paraformer-safe-torch-load.patch)。ONNX shape/log 与导出日志见 `logs/paraformer-export-*-15s.log`，RKNN 构建日志见 `logs/paraformer-rknn-*-15s.log`。三项均显示 `rknn building done.`，无 traceback；encoder/predictor 日志各有 1/2 条 `Unkown op target: 0` 诊断，decoder 无此诊断，需板端实际加载验证。
- 固定图输出：encoder 输入为 15 秒声学特征（250 个 LFR 帧），输出 `[1,250,512]`；predictor/decoder 使用对应固定形状，decoder 输出 `[1,250,8404]`。
- 产物包：[本机从指定 checkpoint 转换的 RK3568 Paraformer-large-Chuan 15 秒模型包](results/rk3568/sherpa-onnx-rk3568-paraformer-large-chuan-15s-aaae1ad1.tar.bz2)，SHA-256 `8123377b3d698440df6d5926e0eba31db9a0a5ab81a2b0e59e23a1945ee50e54`。内含 `encoder.rknn`、`predictor.rknn`、`decoder.rknn`、与 checkpoint token 顺序一致的 `tokens.txt` 及 `SHA256SUMS`。逐文件校验和在 [`paraformer-15s-local/SHA256SUMS`](results/rk3568/paraformer-15s-local/SHA256SUMS)。
- 另保留 sherpa-onnx 官方 release 的 RK3568 15 秒模型包作为参考对照：`results/rk3568/sherpa-onnx-rk3568-15-seconds-paraformer-zh-2025-10-07.tar.bz2`，SHA-256 `f468657d20dbaf894d0fe8d984756dcafc0e5de8313c20471bfe623dccdd9be9`。它的内部 README 指向 ASLP-lab 模型家族，但没有锁定 HF revision；正式复现本次指定 checkpoint 请用上面的本机转换产物。

运行本 Paraformer 模型时使用离线 runner：

```bash
P=experiments/exp002_rknn_conversion/results/rk3568/paraformer-15s-local
experiments/exp001_runtime_profiling/run_offline_paraformer_rknn.sh \
  "$P/encoder.rknn,$P/predictor.rknn,$P/decoder.rknn" "$P/tokens.txt" \
  /path/to/fixed.wav results/paraformer-rknn \
  "$PWD/build-rknn/bin/profile-offline-paraformer-rknn-cxx-api"
```

该模型为 15 秒固定输入图。目标板运行尚未执行；本机转换成功只证明 RKNN 文件已生成，不代表板端加载、NPU 执行、识别正确率或性能通过。

## Zipformer bilingual zh-en-t

- Checkpoint：[csukuangfj/k2fsa-zipformer-bilingual-zh-en-t](https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t)，revision `e2382758de9a0219b4efe682b95af30b399db3b8`，Apache-2.0。
- 官方 checkpoint 中已有 ONNX 导出；复用 `exp/32/` 的 float ONNX，不重新导出 PyTorch checkpoint。该目录 `export-for-onnx.sh`/metadata 对应 decode chunk length 32；模型以 32 为 decode chunk，encoder graph input `x=[N,39,80]`。symbolic batch `N` 固定为 1。tokens 与模型同 revision。
- ONNX SHA-256：
  - `encoder.onnx`: `21b820759ba8a2792838748cdc5a14690fe2010bcb6b370f62e71c87ffeefd84`
  - `decoder.onnx`: `89be509a83175261695bdef5fd1c7b9ab1129a663d1284e7ba9f8507b21e0906`
  - `joiner.onnx`: `bcbcf0cc1b449bc32b7777084ab5a65d1199b822e2aae149d3a43adc9f4b5775`
  - `tokens.txt`: `a8e0e4ec53810e433789b54a5c0134a7eaa2ffca595a6334d54c00da858841d3`
- 转换工具：RKNN-Toolkit2 2.4.2a15，target `rk3568`，`optimization_level=0`、`do_quantization=False`；固定所有符号输入维度的 `N=1`，并将 ONNX metadata 写入 RKNN custom string。使用脚本 [`export_zipformer_rknn.py`](export_zipformer_rknn.py)。
- 产物：[RK3568 Zipformer chunk=32 模型包](results/rk3568/sherpa-onnx-rk3568-zipformer-bilingual-zh-en-t-chunk32.tar.bz2)，SHA-256：`d39b88c77318f75b5cbac3c7073590f748e0d12e098c112ed1028c576fed695e`。解包目录为 `results/rk3568/zipformer-32/`，包含三件套、tokens 和内层 `SHA256SUMS`。
- RKNN 输出 SHA-256：
  - `encoder.rknn`: `d2979bbd8fa6ee418cf5973ae52e52958744e58d8dde63daba2552142ce76e93`
  - `decoder.rknn`: `514f86463eda497eef2b510f88c8131edcae5ef3daba15d2f42c425fb920ee12`
  - `joiner.rknn`: `99ed51880e3b9ceb376aca4b24e0c011aa57238beab6d71467af68f13e2c1c62`
- Toolkit 构建日志：`logs/zipformer-rk3568-chunk32.log`。三模型导出均返回成功，但 log 出现重复的 `Unkown op target: 0` 诊断；因此保留该日志并继续要求板端加载核验。
- 主机 Toolkit simulator 检查：执行 [`validate_zipformer_rknn_sim.py`](validate_zipformer_rknn_sim.py)，日志见 [`logs/zipformer-rk3568-chunk32-simulator.log`](logs/zipformer-rk3568-chunk32-simulator.log)。确定性合成输入下，ONNX Runtime 与 Toolkit simulator 三模型输出形状一致；最大绝对误差 encoder `0.01165`、decoder `0.0000482`、joiner `0.00422`，余弦相似度均大于 `0.999997`。这是模块输出数值检查，不是音频识别测试或真实 NPU 测试。

转换命令：

```bash
.venv/bin/python experiments/exp002_rknn_conversion/export_zipformer_rknn.py \
  --input-dir experiments/exp002_rknn_conversion/inputs/zipformer/32 \
  --output-dir experiments/exp002_rknn_conversion/results/rk3568/zipformer-32 \
  --target-platform rk3568
```

## 可选量化导出（主机端已验证；目标板待验证）

Exp002 的两条 RKNN 转换入口现支持可选的 Toolkit 校准量化，同时保留默认非量化路径。量化必须提供**与对应 ONNX 图输入一致的 RKNN calibration list**；不能把 WAV 文件清单直接当作校准集。下方通用脚本从 WAV 生成图输入张量列表，输出写入独立目录，不覆盖当前 float 产物。

### Zipformer 三图量化

每个图需要自己的校准列表，放在目录中并命名为 `encoder.txt`、`decoder.txt`、`joiner.txt`。列表内容按 RKNN-Toolkit2 `build(dataset=...)` 格式指向 `.npy` 输入张量；多输入图同一行按图输入顺序列出多个 NPY 路径。该列表格式与 NPY 多输入用法可参照 [Rockchip 官方 multi_input 示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/functions/multi_input/test.py)。

```bash
python experiments/exp002_rknn_conversion/export_zipformer_rknn.py \
  --input-dir experiments/exp002_rknn_conversion/inputs/zipformer/32 \
  --output-dir experiments/exp002_rknn_conversion/results/rk3568/zipformer-32-int8 \
  --target-platform rk3568 \
  --quantization-dataset-dir /path/to/zipformer-calibration-lists
```

### WAV 目录通用校准数据生成框架

新增统一入口 [`prepare_calibration.py`](prepare_calibration.py)，按 `--model` 选择 Paraformer 或 Zipformer，递归读取 WAV 后调用对应浮点 ONNX 链，捕获**每次图调用的实际输入张量**，保存 NPY 和 RKNN Toolkit2 calibration list。manifest 记录模型/WAV/张量哈希、输入 shape/dtype、每条音频的图调用次数和失败原因。依赖 `onnxruntime`、`kaldi_native_fbank`、`soundfile`、`scipy`、`numpy`。

Paraformer 固定图示例：

```bash
python experiments/exp002_rknn_conversion/prepare_calibration.py \
  --model paraformer \
  --wav-dir /path/to/representative-wavs \
  --encoder /path/to/encoder-15-seconds.onnx \
  --predictor /path/to/predictor-15-seconds.onnx \
  --decoder /path/to/decoder-15-seconds.onnx \
  --output-dir /path/to/calibration/paraformer
```

Zipformer chunk=32 示例：

```bash
python experiments/exp002_rknn_conversion/prepare_calibration.py \
  --model zipformer \
  --wav-dir /path/to/representative-wavs \
  --encoder experiments/exp002_rknn_conversion/inputs/zipformer/32/encoder.onnx \
  --decoder experiments/exp002_rknn_conversion/inputs/zipformer/32/decoder.onnx \
  --joiner experiments/exp002_rknn_conversion/inputs/zipformer/32/joiner.onnx \
  --tokens experiments/exp002_rknn_conversion/inputs/zipformer/tokens.txt \
  --output-dir /path/to/calibration/zipformer
```

两种模式都输出 `lists/<graph>.txt` 和按实际图输入顺序排列的 NPY 文件。Zipformer adapter 按 sherpa-onnx chunk=32 的窗口/步长、零初始 cache 及 greedy decoder/joiner 调用推进；短于一个可解码窗口的音频不会贡献 encoder 样本。Paraformer adapter 按 sherpa-onnx 离线前端设置提取 Fbank、居中 LFR（边缘帧复制），再按 encoder→predictor→CIF→decoder 顺序运行；encoder ONNX 内含 CMVN。遇到坏音频或模型接口不匹配会记入 manifest，最终以退出码 2 结束。校准音频仍需由使用者挑选并审核覆盖范围。

对 Paraformer encoder/predictor/decoder 分别运行转换脚本，将各自 `lists/*.txt` 传给 `--quantization-dataset`；对 Zipformer 则将生成的 `lists/` 目录传给 Zipformer 转换器的 `--quantization-dataset-dir`。两者都应输出至独立的 int8 目录。

### 量化验证方案与完成门槛

1. 固定一份代表性校准子集和独立评测集；先保存 float ONNX/RKNN 的输出与现有模拟器对照，量化产物写入独立 `*-int8/` 目录。
2. 分别校验三图能构建、导出、加载，输入输出名称/形状未意外改变；用相同输入对 float 与量化图比较逐图误差（最大/平均绝对误差、余弦相似度）并检查 NaN/Inf。
3. 再用固定音频集比较完整识别 CER/WER、转写差异、端到端延迟、模型大小和峰值内存；目标板验证需单独确认 RKNN Runtime/驱动版本、NPU 实际执行和连续运行状态。不得以模拟器或图级误差替代板端及识别质量结论。
4. 每次记录 Toolkit 版本、量化参数/数据集哈希、模型哈希、构建日志、评测输入清单及结果。精度不达预设门槛时保留失败产物和结果，不覆盖当前非量化基线。

现有转换命令不带校准参数时仍生成非量化模型；校准模式可通过下面的实音频验证记录查看当前主机端证据。所有量化文件均为临时主机产物，目标板验证和量化精度/性能评估仍待完成。

### 2026-10-13 校准框架实音频验证

使用 `/mnt/e/datasets/AEC-Challenge/baseline/icassp2021/clean/` 下两条原始 16 kHz PCM16 WAV：`0YP1nl3pDk6paW0ZYAPdpg_farend_singletalk.wav`（SHA-256 `9b10bb150d7451a12a531af3a4f0ab0d32a3163e4128a7208490cbd68a507fcc`）和 `1EbxxdNtTU6sfStbSi-xbA_doubletalk.wav`（SHA-256 `8b421076d57a80e1b799cd0c7e0d9e2232e0640211115fcfec160503225273a4`）。两条 WAV 通过临时目录 symlink 提供给统一脚本；输出位于 `/tmp/exp002-calibration-validation/`，避免将生成的 NPY/RKNN 校准数据放进基线目录。模型输入/WAV/张量 manifest 与 Toolkit 构建日志归档在 [`logs/calibration-framework-validation-2026-10-13/`](logs/calibration-framework-validation-2026-10-13/)。

复现数据目录和 Zipformer 生成/量化命令：

```bash
mkdir -p /tmp/exp002-calibration-validation/wavs
ln -sf /mnt/e/datasets/AEC-Challenge/baseline/icassp2021/clean/0YP1nl3pDk6paW0ZYAPdpg_farend_singletalk.wav /tmp/exp002-calibration-validation/wavs/sample_a.wav
ln -sf /mnt/e/datasets/AEC-Challenge/baseline/icassp2021/clean/1EbxxdNtTU6sfStbSi-xbA_doubletalk.wav /tmp/exp002-calibration-validation/wavs/sample_b.wav

.venv/bin/python experiments/exp002_rknn_conversion/prepare_calibration.py \
  --model zipformer --wav-dir /tmp/exp002-calibration-validation/wavs \
  --encoder experiments/exp002_rknn_conversion/inputs/zipformer/32/encoder.onnx \
  --decoder experiments/exp002_rknn_conversion/inputs/zipformer/32/decoder.onnx \
  --joiner experiments/exp002_rknn_conversion/inputs/zipformer/32/joiner.onnx \
  --tokens experiments/exp002_rknn_conversion/inputs/zipformer/tokens.txt \
  --output-dir /tmp/exp002-calibration-validation/zipformer

.venv/bin/python experiments/exp002_rknn_conversion/export_zipformer_rknn.py \
  --input-dir experiments/exp002_rknn_conversion/inputs/zipformer/32 \
  --output-dir /tmp/exp002-calibration-validation/zipformer-rknn-int8 \
  --target-platform rk3568 \
  --quantization-dataset-dir /tmp/exp002-calibration-validation/zipformer/lists
```

Paraformer 使用同一个 `--wav-dir`，模型参数改为：

```bash
--model paraformer \
--encoder experiments/exp002_rknn_conversion/work/paraformer/encoder-15-seconds.onnx \
--predictor experiments/exp002_rknn_conversion/work/paraformer/predictor-15-seconds.onnx \
--decoder experiments/exp002_rknn_conversion/work/paraformer/decoder-15-seconds.onnx \
--output-dir /tmp/exp002-calibration-validation/paraformer
```

然后对 encoder/predictor/decoder 各调用一次 `work/paraformer/export_rknn.py`，分别把 `--quantization-dataset` 指向 `paraformer/lists/<图名>.txt`，输出到 `paraformer-rknn-int8/<图名>.rknn`。

| 模型/图 | 校准列表行数 | 每行图输入数 | 输入形状摘要 | RKNN-Toolkit2 2.4.2a15 量化构建 |
|---|---:|---:|---|---|
| Paraformer encoder | 2 | 1 | `[1,250,560]` float32 | 成功 |
| Paraformer predictor | 2 | 1 | `[1,250,512]` float32 | 成功 |
| Paraformer decoder | 2 | 3 | 两个 `[1,250,512]`、一个 `[250]` float32 | 成功 |
| Zipformer encoder | 67 | 36 | `x=[1,39,80]` 加 35 个 cache 输入 | 成功 |
| Zipformer decoder | 33 | 1 | `[1,2]` int64 token context | 成功 |
| Zipformer joiner | 536 | 2 | 两个 `[1,512]` float32 | 成功 |

两模型均为 `prepared_wavs=2, failed_wavs=0`；所有 NPY 输入有限，列表行的输入数与 ONNX 图接口相符。六张 int8 RKNN 的 Toolkit 构建均出现 `rknn building done.`，构建日志仍包含 `Unkown op target: 0` 诊断，需保留并在板端进一步核验。量化产物 SHA-256：Zipformer encoder `84080084f1a69e3e04a68639b170671a021fd101177f5374a1da908665b48a9d`、decoder `d5211d05a559556fde76d4991f6129b3a4a8b7bb551f3f66380b01d8b6699b8b`、joiner `477b266a83f934ca27b6a4465a061fbfebf583a36ae40bb45783b2edc711cae4`；Paraformer encoder `617eeef43a2eb86a95a74a5ac2f02c964a9f4136ea9ddf50a0d60490ac4cf72f`、predictor `5c5ba44816384ad82f7f205005799eaa0747d7affb83b9e0d282d6f5a02ab622`、decoder `a5becbbf993db1842b65626f11281b772784ebc45e013bb9c381bbb1cec509ae`。模型体积和 hashes 对应的文件位于临时目录：`/tmp/exp002-calibration-validation/{zipformer,paraformer}-rknn-int8/`。

首次 Paraformer 生成尝试因脚本 NodeArg 输入元信息索引 bug 失败，失败 manifest 已归档；修正后上述两条 WAV 均成功。当前验证证明了 WAV→float ONNX 实际输入采集、NPY/list 格式及主机 Toolkit 量化构建链可用；没有运行量化 RKNN 的板端加载/推理，也没有评价 CER/WER 或量化精度收益。

目标板 profiling 命令见 [Exp001 README](../exp001_runtime_profiling/README.md)。Zipformer 使用 `run_online_zipformer_rknn.sh`；模型包目录内含 `encoder.rknn`、`decoder.rknn`、`joiner.rknn`、`tokens.txt`。Paraformer 使用 `run_offline_paraformer_rknn.sh`。

## 尚未完成的验证

- 未在 RK3568 板端执行 `sherpa-onnx-offline` / C++ runner，未确认 Runtime/driver 兼容、NPU 实际调用、转写结果、CER/WER、温度、频率或性能。
- Zipformer 没有真实音频/状态跨 chunk RKNN 执行验证；`chunk=32` 来源于导出 metadata，不单独证明运行时流式功能。
- Paraformer 已从指定 HF revision 完成本机 checkpoint→ONNX→RKNN 转换；仍需在板端核对模型加载、识别质量与 NPU 执行。官方 release 包仅作未锁定 revision 的参考。
- 当前 host Toolkit2 build 产生的 `Unkown op target: 0` 诊断需在目标板通过模型加载和实际转写确认是否只是非致命日志。
