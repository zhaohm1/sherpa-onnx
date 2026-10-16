# Exp001：端到端与模块级 Profiling

状态：**进行中**。Profiling 插桩、三个通用 profile runner、两种 RK3568 RKNN 模型包和面向无网络设备的 C++ 测试程序已准备；固定测试音频与 RKNN 产物尚未在目标板上运行，因此没有板端识别或性能结果。

## 目的与假设

建立 Paraformer 与 Zipformer-RNNT 在线推理的可重复入口，分开记录模型初始化、音频读取、流创建、波形输入、ready 检查、decode 调用、结果读取和端到端耗时。假设公开 API 只能把模型计算作为 `decode_streams` 整体测量；需以运行版本/API 核验，不把聚合时间拆写为独立 encoder/decoder/joiner 数值。

## 基线、模型与数据

- 目标基线：当前 sherpa-onnx 线上模型、CPU provider、相同测试 WAV、greedy search、1 线程；硬件可用后另采 RKNN 路径，不能与本机结果混为一组。
- Paraformer：当前选择为离线 `Paraformer-large-Chuan`，来源为 Hugging Face `ASLP-lab/WSChuan-ASR/Paraformer-large-Chuan`；已从指定 Hugging Face revision 完成本机 RK3568、15 秒 encoder/predictor/decoder 转换；板端尚未运行。早先提到的 ModelScope online Paraformer 不属于当前这套离线 RKNN 测试模型。
- Zipformer-RNNT：Hugging Face `csukuangfj/k2fsa-zipformer-bilingual-zh-en-t`，revision `e2382758de9a0219b4efe682b95af30b399db3b8`；复用同 revision 的 chunk=32 ONNX encoder/decoder/joiner 并转成 RK3568 RKNN。导出 metadata 与实际状态/cache运行路径仍待板端核实；不把“存在 cache”当作流式证明。
- 数据：命令行传入单声道 16-bit PCM WAV；尚无固定音频清单、转写标签或正确率证据。本实验不产生 CER/WER。
- 脚本验证适用范围：只面向 `OnlineRecognizer` 的 Paraformer 与 transducer 两类模型；离线 Paraformer、RKNN Runtime 专属 profiling 不在此入口内。

### RKNN 离线板端测试入口（本次补充）

Python 在线入口不能代表 RKNN C++ 路径。源码路径选择表明在线 RKNN 目前适配 Zipformer Transducer；在线 Paraformer 的 RKNN 请求会回退 CPU。离线 RKNN Paraformer 则由独立实现支持。因此提供两个 C++ API runner，直接连入本地构建的 sherpa-onnx 库；运行时不需要网络或 Python 包：

**模型对应关系：** 当前 Paraformer checkpoint 是 sherpa-onnx 离线 Paraformer RKNN 转换流程支持的 `Paraformer-large-Chuan`，运行时使用 `sherpa-onnx-offline` 对应的 `OfflineRecognizer` 路径。Zipformer 仍使用 online Transducer 路径。

- `cxx-api-examples/profile-online-zipformer-rknn-cxx-api.cc`：在线 Zipformer Transducer，支持 `--provider rknn`（默认）和 `--provider cpu`。按固定音频分块输入，记录每次 `Decode` 的调用耗时和触发它的输入块编号，同时记录 E2E、首个非空结果和最终文本。
- `sherpa-onnx/csrc/profile-online-alsa.cc`：ALSA 实时采集 profiling runner。模型参数复用 `sherpa-onnx-alsa` 的 `OnlineRecognizerConfig` 参数注册与解析；只接收一个末尾设备参数，通过 `--profile-streams` 将同一采集音频送入多个 stream；另有 `--profile-output-dir`、`--profile-chunk-seconds` 和 `--profile-duration-seconds`。
- `cxx-api-examples/profile-offline-paraformer-rknn-cxx-api.cc`：离线 Paraformer，支持 `--provider rknn`（默认，`--model` 接收按 encoder、predictor、decoder 顺序逗号分隔的三个 `.rknn` 文件）和 `--provider cpu`（`--model` 接收一个 Paraformer `.onnx` 文件）。
- 两者默认各执行 2 次 warm-up、20 次测量、单线程；warm-up 后调用 `ResetModuleProfileCounters()`。重复测量文本不一致时 runner 返回退出码 3，并保留已经写出的结果。
- runner 启动时启用 `SHERPA_ONNX_PROFILE`；进程退出时写 `module_profile.csv`。E2E CSV/summary 与模块 CSV 是不同口径。模块计时含应用侧 RKNN API 耗时，不是 NPU 硬件算子细节；计时行存在嵌套，不可直接相加。
- 配套板端脚本 `run_online_zipformer_rknn.sh`、`run_offline_paraformer_rknn.sh` 会拒绝覆盖既有输出目录，并保存运行前的 `environment.txt`、运行后的 `device_state_after.txt`、输入 SHA-256、`run.log`、E2E 结果和 `module_profile.csv`。两者均只用设备上已有的 runner、模型和 WAV，不访问网络。

目标板构建时确保启用 RKNN 和 C++ API 示例，并使用该板对应的 RKNN headers/librknnrt：

```bash
cmake -S . -B build-rknn \
  -DSHERPA_ONNX_ENABLE_RKNN=ON \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=ON \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rknn --target \
  profile-online-zipformer-rknn-cxx-api \
  profile-offline-paraformer-rknn-cxx-api --parallel
```

板端运行示例：

```bash
experiments/exp001_runtime_profiling/run_online_zipformer_rknn.sh \
  /models/encoder.rknn /models/decoder.rknn /models/joiner.rknn \
  /models/tokens.txt /audio/fixed.wav results/zipformer-rknn \
  "$PWD/build-rknn/bin/profile-online-zipformer-rknn-cxx-api"

experiments/exp001_runtime_profiling/run_offline_paraformer_rknn.sh \
  /models/encoder.rknn,/models/predictor.rknn,/models/decoder.rknn \
  /models/tokens.txt /audio/fixed.wav results/paraformer-rknn \
  "$PWD/build-rknn/bin/profile-offline-paraformer-rknn-cxx-api"
```

Zipformer 的 `feed_chunk_index` 仅标识触发该轮 `is_ready` drain 的输入块，不承诺模型 cache 内帧与该单个输入块一一对应。第一轮只使用单一 WAV，输出重复文本一致性而非 CER/WER；正式比较仍需固定带标签测试集。

### ONNX Runtime profiling

两个固定 WAV runner 也可用于 CPU ONNX 基线。`module_profile.csv` 记录源码已有插桩的模块调用；Zipformer 可观察 encoder/decoder/joiner session、batch preparation 和 greedy search，离线 Paraformer 可观察整体 model forward 与后处理等阶段。该计时是模块/API 调用耗时，不是 ONNX 图中逐算子耗时。

### 通用三类 profile runner

新增 `profile-sherpa-onnx`、`profile-sherpa-onnx-offline` 和 `profile-sherpa-onnx-alsa`，分别对应 `sherpa-onnx`、`sherpa-onnx-offline` 和 `sherpa-onnx-alsa`。每个 runner 都用原类的 `OnlineRecognizerConfig` 或 `OfflineRecognizerConfig` 注册模型参数，因而保留对应主程序支持的模型、provider、线程、解码及特征参数；profile 自身的参数独立增加。

固定 WAV runner 命令格式如下，原程序支持的识别参数可以直接沿用：

```bash
build/bin/profile-sherpa-onnx \
  --tokens=tokens.txt --encoder=encoder.onnx --decoder=decoder.onnx \
  --joiner=joiner.onnx --provider=cpu --num-threads=1 \
  --profile-output-dir=results/online --profile-warmup=2 --profile-repeat=20 \
  --profile-streams=8 a.wav

build/bin/profile-sherpa-onnx-offline \
  --tokens=tokens.txt --paraformer=paraformer.onnx --provider=cpu \
  --num-threads=1 --profile-output-dir=results/offline \
  --profile-warmup=2 --profile-repeat=20 --profile-streams=8 a.wav
```

RKNN 使用相同的 runner 和识别参数接口，只需把 provider 与模型文件改为 RKNN 配置：

```bash
# 在线 Zipformer Transducer：三个 RKNN 图
build-rknn/bin/profile-sherpa-onnx \
  --tokens=tokens.txt --encoder=encoder.rknn --decoder=decoder.rknn \
  --joiner=joiner.rknn --provider=rknn --num-threads=1 \
  --profile-output-dir=results/online-rknn --profile-warmup=2 --profile-repeat=20 \
  --profile-streams=8 a.wav

# 离线 Paraformer：encoder,predictor,decoder 顺序的 RKNN 三件套
build-rknn/bin/profile-sherpa-onnx-offline \
  --tokens=tokens.txt \
  --paraformer=encoder.rknn,predictor.rknn,decoder.rknn \
  --provider=rknn --num-threads=1 \
  --profile-output-dir=results/offline-rknn --profile-warmup=2 --profile-repeat=20 \
  --profile-streams=8 a.wav
```

同样可为 `profile-sherpa-onnx-alsa` 传入 ONNX 模型并设置 `--provider=cpu`，或传入 RKNN 模型并设置 `--provider=rknn`。RKNN 构建需启用 `-DSHERPA_ONNX_ENABLE_RKNN=ON` 并链接目标平台的 Runtime；通用接口不会让当前 runtime 尚不支持的模型获得 RKNN 支持。在线 RKNN 当前覆盖 Zipformer Transducer 与 Zipformer2 CTC，离线 RKNN 覆盖 Paraformer 与 SenseVoice。模型扩展名/provider 不匹配时由 recognizer config 校验拒绝。

固定 WAV runner 只接收一个 WAV；runner 在内部复制该波形，为 `--profile-streams` 指定数量的 stream 输入同一音频，并以 batch decode 测量并发负载。`runs.csv` 的 `audio_s` 是单路时长乘以路数，`utterances.csv` 按 stream 保存输出，`summary.csv` 汇总 provider、线程、解码配置、路数及重复测量分布，`module_profile.csv` 保存源码插桩。默认一路；profile 输出目录必须不存在。

ALSA runner 同时以 `profile-sherpa-onnx-alsa` 名称安装，保留已有 `profile-online-alsa` 兼容名称。模型参数沿用 `sherpa-onnx-alsa`；末尾只传一个设备名，通过 `--profile-streams` 指定内部并发路数：

```bash
build-rknn/bin/profile-sherpa-onnx-alsa \
  --tokens=tokens.txt --encoder=encoder.rknn --decoder=decoder.rknn \
  --joiner=joiner.rknn --provider=rknn --num-threads=1 \
  --profile-output-dir=results/alsa-multi --profile-chunk-seconds=0.1 \
  --profile-duration-seconds=60 --profile-streams=8 plughw:3,0
```

同一设备采集到的每个块会复制后送入所有 stream，再通过 `DecodeStreams` 批量解码。`chunks.csv` 和 `utterances.csv` 按 stream 编号区分结果，`summary.csv` 每路一行。`decode_batch_s` 是批模型调用墙钟耗时，会对参与该批的各 stream 重复记录，不能跨行求和；`read_s` 是单设备采集等待。实时 ALSA 与 WAV 快速回放口径不同，不能直接比较 RTF。

### 板端多路并发测试与指标判读

在目标板先记录模型哈希、板卡/系统/Runtime 版本、provider、线程数、解码参数、CPU governor/频率、温度及可获得的 CPU/NPU 负载和功耗。每次只改变 `--profile-streams`，按 1、2、4、8……逐级增加；每个路数使用独立输出目录，固定模型、输入、线程数和运行时条件。可先运行 `collect_device_info.sh` 保存静态环境信息；温度、频率、利用率和功耗需另用板端 sysfs 或厂商工具记录。

固定 WAV 容量测试示例：

```bash
for n in 1 2 4 8; do
  build-rknn/bin/profile-sherpa-onnx \
    --tokens=/models/tokens.txt --encoder=/models/encoder.rknn \
    --decoder=/models/decoder.rknn --joiner=/models/joiner.rknn \
    --provider=rknn --num-threads=1 --profile-warmup=2 \
    --profile-repeat=50 --profile-streams="$n" \
    --profile-output-dir="results/zipformer-rknn-streams-$n" \
    /audio/fixed.wav
done
```

`summary.csv` 中 `mean/p50/p95/max_batch_wall_s` 是每次批量解码墙钟时间；`total_audio_s` 已乘以路数，因此 `mean_rtf = mean_batch_wall_s / total_audio_s`，吞吐约为 `1 / mean_rtf` 倍实时。关注路数增加时总吞吐是否增长，以及单路等效 RTF（批次 RTF × 路数）是否恶化。`model_init_s` 单独看加载成本，不混入稳态耗时。50 次测量可用于初步看 p95；容量决策建议增加重复次数并报告有效样本数。WAV 是快速回放容量测试，不代表麦克风实时延迟。若检查文本正确率，需用固定标注集计算每路 CER/WER；仅比较各路文本是否一致不能代替准确率。

ALSA 实时测试示例（单设备输入，内部复制到多路）：

```bash
for n in 1 2 4 8; do
  build-rknn/bin/profile-sherpa-onnx-alsa \
    --tokens=/models/tokens.txt --encoder=/models/encoder.rknn \
    --decoder=/models/decoder.rknn --joiner=/models/joiner.rknn \
    --provider=rknn --num-threads=1 --profile-chunk-seconds=0.1 \
    --profile-duration-seconds=300 --profile-streams="$n" \
    --profile-output-dir="results/alsa-rknn-streams-$n" \
    plughw:3,0
done
```

ALSA 的 `read_s` 包含阻塞等音频时间，不是推理耗时。对每个 chunk，`accept_s` 按 stream 汇总，再加一次（不要重复相加）`decode_batch_s`；将此计算时间分布与单个 `audio_s`（例如 100 ms）比较，p95/max 应留有实时余量。同步检查是否出现 ALSA XRUN/overrun、片段是否完整结束、`first_nonempty_result_s` 和 `utterances.csv` 中的 `wall_s/audio_s`。进程峰值 RSS、CPU/NPU 利用率与频率、温度也要一并记录。模块计时可能有父子嵌套，不应把嵌套行简单求和；RKNN API 调用耗时也不等于硬件算子耗时。

这两个 runner 将同一 WAV 或同一 ALSA 采集块复制到所有 stream，适合测批量并发承载能力；它不覆盖不同用户各自输入时的准确率、输入抖动和端到端隔离情况。目标板实测应单独保存原始 CSV、运行日志和环境记录，不能用主机编译或模拟结果替代。

使用启用了 C API、binary 和 C++ API examples 的 CPU 构建：

```bash
cmake -S . -B build-cpu \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=ON \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target \
  profile-online-zipformer-rknn-cxx-api \
  profile-offline-paraformer-rknn-cxx-api --parallel
```

```bash
build-cpu/bin/profile-online-zipformer-rknn-cxx-api \
  --encoder /models/encoder.onnx \
  --decoder /models/decoder.onnx \
  --joiner /models/joiner.onnx \
  --tokens /models/tokens.txt --wav /audio/fixed.wav \
  --provider cpu --num-threads 1 --warmup 2 --repeat 20 \
  --output-dir results/zipformer-onnx-cpu

build-cpu/bin/profile-offline-paraformer-rknn-cxx-api \
  --model /models/paraformer.onnx \
  --tokens /models/tokens.txt --wav /audio/fixed.wav \
  --provider cpu --num-threads 1 --warmup 2 --repeat 20 \
  --output-dir results/paraformer-onnx-cpu
```

原有 RKNN 调用无需增加 `--provider`，仍以 `rknn` 为默认值；Zipformer 使用 encoder/decoder/joiner 三个模型文件，Paraformer 使用三个逗号分隔的 RKNN 文件。summary 中新增 `provider` 字段以区分结果。

### ALSA 实时采集 runner

构建需要 ALSA 开发头文件和启用 binary；其余模型选项与 `sherpa-onnx-alsa` 相同，支持 CPU/RKNN provider 和对应在线模型参数。末尾只传一个设备名；`--profile-streams` 控制 runner 内部由同一设备创建的并发 stream 数。默认持续采集直到 Ctrl+C，也可用 profiling 时长自动停止。ALSA runner 不做文件回放式 warm-up；模型初始化独立计时，模块计数在采集开始前清零。

```bash
cmake --build build-rknn --target profile-online-alsa --parallel

build-rknn/bin/profile-sherpa-onnx-alsa \
  --tokens=/models/tokens.txt \
  --encoder=/models/encoder.rknn \
  --decoder=/models/decoder.rknn \
  --joiner=/models/joiner.rknn \
  --provider=rknn \
  --num-threads=1 \
  --decoding-method=greedy_search \
  --profile-output-dir=results/alsa-zipformer \
  --profile-chunk-seconds=0.1 \
  --profile-duration-seconds=60 \
  --profile-streams=8 \
  plughw:3,0
```

输出目录必须不存在。`chunks.csv` 按 stream 记录采集等待、输入时长、waveform 提交、批量解码耗时、首个非空结果延迟及文本；`utterances.csv` 按 stream 记录 endpoint 片段，停止时未完成的片段标记为 `stopped_partial`；`summary.csv` 每个 stream 一行，包含初始化/采集时间、音频时长、块数、片段数和进程峰值 RSS；`module_profile.csv` 汇总所有 stream 的 C++ 模块插桩。`read_s` 包含 ALSA 等待实时音频的时间；相同 chunk 的 `decode_batch_s` 会在各 stream 行重复记录，不要按路数相加。当前实现尚未完成编译和目标板实测。

采集板卡、内核、CPU、ALSA 设备和 RKNN/NPU 信息：

```bash
experiments/exp001_runtime_profiling/collect_device_info.sh \
  > results/device_info.txt 2>&1
```

若 `dmesg` 无读取权限，脚本会提示单独用 `sudo dmesg` 获取驱动日志；输出中不采集 CPU 序列号。

## 环境与版本

- sherpa-onnx 仓库 commit：`9db1af1871ed78cee9357c23898e5a7d8702d21a`。
- 仓库自带 API 示例为 `python-api-examples/online-decode-files.py`；本实现复用其 `OnlineRecognizer.from_paraformer`、`from_transducer` 及 `create_stream` / `accept_waveform` / `is_ready` / `decode_streams` / `get_result` 公共调用路径，不修改上游 runtime。
- 当前主机：WSL2 Linux x86_64；CPython 3.10.20 和 uv 0.11.15 可用。NPU 板端、驱动、RKNN Runtime、CPU governor/频率、温度、电源和功耗均未测。
- Python 依赖：NumPy、与当前模型版本兼容的 `sherpa-onnx` Python package；请在隔离环境安装，不向系统 Python 添加依赖。

### RKNN 转换环境（本次补充）

- 项目根目录使用 `.python-version` 固定 Python 3.10 系列，`pyproject.toml` 与 `uv.lock` 描述独立的转换环境；`.venv/` 不纳入 Git。
- 当前 WSL2 主机为 Linux x86_64，已检测到 CPython 3.10.20 与 uv 0.11.15。RKNN-Toolkit2 归档位于 `/mnt/c/Users/ZhaoHM/Downloads/2.4.2a15.tgz`，SHA-256：`41dbf92a1e2db74c962a753b4d45293d2ceec687c409a62a5211079e4a6856ae`。包内包含 CPython 3.10 / x86_64 wheel `rknn_toolkit2-2.4.2a15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`；wheel 版本为 `2.4.2a15`。
- 运行时依赖约束来自该 wheel 元数据：NumPy `<=1.26.4`、protobuf `<=4.25.4`、ONNX `>=1.16.1`、ONNX Runtime `>=1.17.0`、Torch `>=1.10.1,<=2.4.0`，另含 SciPy、OpenCV、psutil、ruamel.yaml、fast-histogram 和 tqdm。项目锁定 NumPy 1.26.4、ONNX 1.16.2、ONNX Runtime 1.18.1 和 protobuf 4.25.4。Toolkit2 wheel 的 metadata 将 Torch 列为依赖，但当前主环境先按 ONNX 转换路径省略 Torch；`checkpoint-export` extra 提供兼容的 CPU-only Torch 2.4.0+cpu（命令：`uv sync --extra checkpoint-export --python 3.10`）。若 Toolkit2 导入或目标导出脚本需要 Torch，应启用该 extra。
- 首次部署：`uv sync --python 3.10`，再运行 `scripts/install_rknn_toolkit2.sh`（可用首个参数指定归档路径）。脚本只从归档解出 CPython 3.10 x86_64 wheel，安装到仓库 `.venv`，不复制 Toolkit2 二进制到 Git。2026-10-10 在 PyPI 默认源下载大 wheel 时长时间停滞，改用清华 PyPI 镜像完成基础依赖同步；随后执行 `uv sync --locked --extra checkpoint-export --python 3.10` 安装 CPU Torch 2.4.0。`uv sync` 会清理由脚本手动安装的 Toolkit2 wheel，因此每次同步后需重新运行 `scripts/install_rknn_toolkit2.sh`。
- 环境核验通过：Python 3.10.20、NumPy 1.26.4、ONNX 1.16.2、ONNX Runtime 1.18.1、OpenCV 4.11.0.86、protobuf 4.25.4、SciPy 1.15.3、Torch 2.4.0+cpu、RKNN-Toolkit2 2.4.2a15；`from rknn.api import RKNN` 导入成功。Toolkit2 归档 SHA-256 为 `41dbf92a1e2db74c962a753b4d45293d2ceec687c409a62a5211079e4a6856ae`。这验证主机 Python API 导入，不代表已完成模型转换或板端部署。
- 该归档附带的 Linux aarch64 `librknnrt.so` 标称 `2.4.2a2`，`rknn_server` 标称 `2.4.0`。这些只是归档内组件版本；没有目标板访问权限，未核验 RK3568 实际安装版本、驱动兼容性或 NPU 调用情况。不要把它记作板端已部署或验证。

### 模型来源与转换状态

| 模型 | 来源 / revision | 当前状态 | 详细证据 |
|---|---|---|---|
| Zipformer bilingual zh-en-t | [Hugging Face](https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t)，`e2382758de9a0219b4efe682b95af30b399db3b8` | chunk=32 ONNX 已转为 RK3568 RKNN；Toolkit simulator 做了合成张量对照 | `experiments/exp002_rknn_conversion/README.md` |
| Paraformer-large-Chuan | [Hugging Face](https://huggingface.co/ASLP-lab/WSChuan-ASR/tree/main/Paraformer-large-Chuan)，`aaae1ad1b28a0c390d1c6f4912bb75601f53dcc4` | 15 秒 encoder/predictor/decoder ONNX 与 RKNN 已本机生成；HF checkpoint SHA 已核对 | `experiments/exp002_rknn_conversion/README.md` |

用户先前提到的 ModelScope online Paraformer 属于不同模型；当前离线 RKNN profiling 使用 Paraformer-large-Chuan。模型转换完成不表示板端运行或 NPU 执行已经验证。

## 实现与计时口径

入口：[profile_online.py](profile_online.py)。输出 `summary.json`、逐 utterance 的 `utterances.csv` 和逐次 decode 调用的 `chunks.csv`。

- 初始化与 WAV 读取单独记录；E2E 从创建 stream 开始，覆盖分块输入、流结束填充、解码和最终结果读取。
- decode chunk 的实际输入时长和 decode 调用耗时逐条留存。summary 提供平均、p50、p95、最大值、调用数、RTF、进程峰值 RSS、模型文件大小及 SHA-256。E2E p95 样本数少于 20 时输出 `null`，避免在小样本上报告不稳定的 p95。
- `accept_waveform` 是应用输入 API 耗时；它不等于独立 Fbank 耗时。`decode_streams` 在当前公共 Python API 中包含 Fbank/特征队列消费、模型执行、cache/state 更新和搜索，不能拆成内部模块数字。
- 线程数/provider 是配置值，metadata 记录的是请求配置，并不证明底层每个算子实际落在对应设备。
- 当前未测内存映射峰值（只记录进程 peak RSS）、功耗或板端频率/温度；这些字段必须在目标板另行采集。

## 复现命令

在安装有匹配版 `sherpa_onnx` 与 NumPy 的隔离 Python 环境中，从仓库根目录执行。路径应替换为同一套部署 checkpoint 的实际产物：

```bash
python experiments/exp001_runtime_profiling/profile_online.py \
  --model-kind paraformer \
  --tokens /path/to/paraformer/tokens.txt \
  --encoder /path/to/paraformer/encoder.onnx \
  --decoder /path/to/paraformer/decoder.onnx \
  --provider cpu --num-threads 1 --chunk-seconds 0.2 \
  --warmup 1 --repeat 20 \
  --output-dir experiments/exp001_runtime_profiling/results/paraformer-cpu \
  /path/to/fixed/test.wav

python experiments/exp001_runtime_profiling/profile_online.py \
  --model-kind transducer \
  --tokens /path/to/zipformer/tokens.txt \
  --encoder /path/to/zipformer/encoder.onnx \
  --decoder /path/to/zipformer/decoder.onnx \
  --joiner /path/to/zipformer/joiner.onnx \
  --provider cpu --num-threads 1 --chunk-seconds 0.2 \
  --warmup 1 --repeat 20 \
  --output-dir experiments/exp001_runtime_profiling/results/zipformer-cpu \
  /path/to/fixed/test.wav
```

运行时将拒绝缺失模型或音频文件。请分别固定模型文件、WAV 清单、线程、provider、解码参数、chunk、warm-up 和重复次数；改变其中一个主要变量时写入独立输出目录。

## 检查项与结果

| 检查项 | 结果 |
|---|---|
| Python 3.10 / uv 锁文件生成 | 通过：`uv lock --python 3.10`，锁定 26 个解析后包 |
| 隔离依赖同步 | 通过：Python 3.10.20；基础依赖与 `checkpoint-export` extra 均由锁文件同步 |
| Toolkit2 wheel 安装与导入 | 通过：2.4.2a15 安装，`import rknn` 与 `from rknn.api import RKNN` 均成功 |
| Python 源码语法编译 | 通过：`py_compile` 写入 `/tmp/exp001_profile_online.pyc` |
| 固定 WAV 转写/重复结果一致性 | 未执行：固定 WAV 和目标模型尚未放入实验输入 |
| Paraformer CPU E2E/模块聚合计时 | 未执行：模型产物与 sherpa-onnx Python runtime 尚未准备 |
| Zipformer-RNNT CPU E2E/模块聚合计时 | 未执行：checkpoint/export 映射与模型产物待核实 |
| C++ RKNN runner 源码语法检查 | 通过：两个 runner 通过 `c++ -std=c++17 -I. -fsyntax-only`；两个 shell 脚本通过 `bash -n` |
| 通用 profile runner 源码语法检查 | 未完成：当前工作机未找到 ONNX Runtime C++ 开发头文件 `onnxruntime_cxx_api.h`，C++ 编译停在该缺失依赖；待有匹配依赖的 sherpa-onnx 构建环境验证 |
| ALSA 多 stream profile runner 编译 | 未完成：当前工作机缺少 ALSA 开发头文件 `alsa/asoundlib.h`；未执行设备侧多路运行验证 |
| RKNN-enabled CMake 构建 | 未执行：需在有 RKNN headers/runtime 的目标构建环境验证 |
| RK3568/NPU E2E 与实际 NPU 调用确认 | 未执行：本轮无法连接目标 NPU |
| CER/WER | 未执行：无固定标签清单 |

## 结论与限制

### C++ 模块插桩（本次补充）

- 新增 `sherpa-onnx/csrc/profiling.h`：默认关闭；设置 `SHERPA_ONNX_PROFILE=1` 后以 `std::chrono::steady_clock` 记录调用数、总时长、均值、最大值及 log2 纳秒桶估计的 p50/p95 上界。进程退出时写 CSV；输出路径由 `SHERPA_ONNX_PROFILE_OUTPUT` 指定，未指定则写 `sherpa-onnx-profile.csv`。高频路径用原子累加，不在每次调用时写日志。
- 覆盖模块：stream waveform/Fbank 输入；在线 Paraformer chunk preprocessing、encoder session、CIF/cache、decoder session；在线 Zipformer batch preparation、encoder/decoder/joiner session 与 greedy/modified-beam search；在线 RKNN Zipformer encoder/decoder/joiner 的 context wait、`rknn_inputs_set`、`rknn_run`、`rknn_outputs_get`；离线 Paraformer RKNN 的 encoder/predictor/acoustic embedding/decoder 与对应 RKNN API 调用。
- 本地运行示例：`SHERPA_ONNX_PROFILE=1 SHERPA_ONNX_PROFILE_OUTPUT=results/profile.csv ./your_cpp_recognizer <固定参数>`。若应用有 warm-up，在调用 warm-up 后、开始测量前 include `sherpa-onnx/csrc/profiling.h` 并调用 `sherpa_onnx::ResetModuleProfileCounters()`；需确保推理线程空闲。统计覆盖进程中后续的所有线程和 stream。
- 面向断网 NPU 设备的 runner、构建命令和结果文件说明见上文“RKNN 离线板端测试入口”。本轮仅完成源码语法检查，未用 RKNN 配置实际构建或运行；板端结果待采集。
- CSV 阶段相互嵌套：例如 `rknn.zipformer.encoder_total` 包含 context wait 和各 RKNN API 分项，不能把这些行相加。`rknn_run` 与 `rknn_outputs_get` 的时间是应用调用耗时；只有在板端且 Runtime 支持时再另采 `RKNN_QUERY_PERF_RUN/DETAIL`。近似分位数是 log2 桶上界，不等同精确样本分位数。
- 断网板端 runner 已完成语法检查，但尚未在 RKNN-enabled 配置下构建或运行；当前仍没有 CSV 实测数据。Python 转换环境与 Toolkit2 导入已经核验。修改只覆盖指定 Paraformer/Zipformer 路径，其他 recognizer/backend 未插桩。

目前完成的是 Python 和 C++ profiling 入口/插桩及断网板端测试程序，不是 Exp001 性能验收。高层 API 计时仍提供应用阶段和 decode 总时间；C++ 计时为识别内部模块热点提供额外边界，但尚无实际板端转写或性能结果。

## 证据与关联提交

- 源码：[profile_online.py](profile_online.py)
- 结构化运行产物：尚未生成；模型/音频到位后写入本目录的 `results/`。
- 当前仓库提交：`9db1af1871ed78cee9357c23898e5a7d8702d21a`；本次工作尚未提交。
