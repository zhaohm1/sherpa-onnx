# Exp001：端到端与模块级 Profiling

状态：**进行中**。Profiling 入口和两种 RKNN 测试程序已准备；已收到目标设备运行的 Paraformer 与 Zipformer profiling 汇总（各 2 次 warm-up、20 次测量），原始报告已归档。设备系统信息为 Linaro ALIP / Linux 5.10.209 / aarch64 GNU/Linux；具体 SoC、RKNN Runtime/驱动版本及 NPU 硬件执行证据仍待补齐，识别准确率也尚未评估。

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
- `sherpa-onnx/csrc/profile-online-alsa.cc`：ALSA 实时采集 profiling runner。模型参数复用 `sherpa-onnx-alsa` 的 `OnlineRecognizerConfig` 参数注册与解析，ALSA 设备仍作为最后一个位置参数；增加 `--profile-output-dir`、`--profile-chunk-seconds` 和 `--profile-duration-seconds`。
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

构建需要 ALSA 开发头文件和启用 binary；其余模型选项与 `sherpa-onnx-alsa` 相同，支持 CPU/RKNN provider 和对应在线模型参数。默认持续采集直到 Ctrl+C，也可用 profiling 时长自动停止。ALSA runner 不做文件回放式 warm-up；模型初始化独立计时，模块计数在采集开始前清零。

```bash
cmake --build build-rknn --target profile-online-alsa --parallel

build-rknn/bin/profile-online-alsa \
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
  plughw:3,0
```

输出目录必须不存在。`chunks.csv` 记录每次采集阻塞时间、输入音频时长、waveform 提交时间、解码调用次数/耗时、当前 segment 首个非空结果延迟及部分文本；`utterances.csv` 记录 endpoint 完成的 segment，停止时未完成的 segment 标记为 `stopped_partial`；`summary.csv` 记录初始化/采集时间、累计音频时长、块数、segment 数和峰值 RSS；`module_profile.csv` 为 C++ 模块插桩结果。`read_s` 包含 ALSA 等待实时音频的时间，不能当作推理耗时；实时负载应结合每块 `decode_s` 与 `audio_s` 判断。此实时麦克风结果不能与 WAV 快速回放的 E2E/RTF 直接混作同一测量口径。当前实现尚未完成编译和目标板实测。

采集板卡、内核、CPU、ALSA 设备和 RKNN/NPU 信息：

```bash
experiments/exp001_runtime_profiling/collect_device_info.sh \
  > results/device_info.txt 2>&1
```

若 `dmesg` 无读取权限，脚本会提示单独用 `sudo dmesg` 获取驱动日志；输出中不采集 CPU 序列号。

## 环境与版本

- sherpa-onnx 仓库 commit：`9db1af1871ed78cee9357c23898e5a7d8702d21a`。
- 仓库自带 API 示例为 `python-api-examples/online-decode-files.py`；本实现复用其 `OnlineRecognizer.from_paraformer`、`from_transducer` 及 `create_stream` / `accept_waveform` / `is_ready` / `decode_streams` / `get_result` 公共调用路径，不修改上游 runtime。
- 当前主机：WSL2 Linux x86_64；CPython 3.10.20 和 uv 0.11.15 可用。用户提供的目标设备信息：Linaro ALIP、Linux 5.10.209、aarch64 GNU/Linux。具体 SoC/板卡、RKNN 驱动与 Runtime 版本、CPU governor/频率、温度、电源和功耗尚未记录。
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

## 目标设备运行结果（2026-10-13）

原始汇总文件：[Profil.md](results/board-run-2026-10-13/Profil.md)，SHA-256：`b9502850710903f280e602c9470bc0485fd818c8937e42b2a1692e1ec5feff8b`。该文件由用户提供，包含两种模型各 20 次测量、模块计时表和逐次转写结果。运行参数记录为 2 次 warm-up、20 次 repeat、1 线程；音频时长 10.76456 秒。

| 模型 | 初始化 | 平均 E2E | p50 E2E | p95 E2E | 最大 E2E | 平均 RTF | 峰值 RSS | 重复文本 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Offline Paraformer | 3.000842 s | 2.956278 s | 2.952510 s | 2.994025 s | 2.997028 s | 0.274631 | 916.6914 MB | 一致 |
| Online Zipformer | 0.664195 s | 2.879058 s | 2.877474 s | 3.010760 s | 3.020688 s | 约 0.267 | 163.2891 MB | 一致 |

- 两者处理这段 10.76 秒音频均快于实时播放速度。Zipformer 平均 E2E 比 Paraformer 短约 77 ms（约 2.7%）；这只是当前不同模型/调用路径的单音频观察，不能作为模型或后端优劣的通用结论。
- Paraformer 模块均值中，RKNN encoder 总耗时约 1.912 s，decoder 约 0.960 s；主要热点在 encoder。Zipformer 每次 decode chunk 平均约 82.3 ms，其中 encoder 总耗时约 67.1 ms；这些是应用及 RKNN API 调用层计时，不是硬件算子时延。
- Profiling 的分位数列是 log2 纳秒直方图桶上界，精度较粗；端到端 p50/p95 应以逐次测量汇总表为准。父子模块计时存在嵌套，不得直接求和。
- 报告的 `text_consistent` 只表示同一模型重复运行的输出一致。转写文本在原始 Markdown 中呈现为编码乱码；逆向解码能恢复可读中文，但没有参考文本，不能据此判断 CER/WER 或识别准确率。
- 板端系统信息目前只有 `Linaro ALIP / Linux 5.10.209 / aarch64 GNU/Linux`。原始报告未包含 SoC/板卡型号、RKNN Runtime/驱动版本、设备温度/频率或 NPU 负载；`rknn.*` 模块名称表明走了 RKNN 调用路径，但不足以独立证明 NPU 硬件执行。具体设备与 runtime 信息见 `collect_device_info.sh` 的采集项。
- Zipformer 的首个非空结果延迟是在 runner 快速分块喂入完整 WAV 时测得，输入未按实时播放节奏等待，不能视为实时麦克风的首字延迟。

此结果可作为设备侧初步性能记录；补齐硬件/runtime 身份、实际 NPU 执行证据、标签集和 CER/WER 后，再形成正式性能与正确性基线。

## 检查项与结果

| 检查项 | 结果 |
|---|---|
| Python 3.10 / uv 锁文件生成 | 通过：`uv lock --python 3.10`，锁定 26 个解析后包 |
| 隔离依赖同步 | 通过：Python 3.10.20；基础依赖与 `checkpoint-export` extra 均由锁文件同步 |
| Toolkit2 wheel 安装与导入 | 通过：2.4.2a15 安装，`import rknn` 与 `from rknn.api import RKNN` 均成功 |
| Python 源码语法编译 | 通过：`py_compile` 写入 `/tmp/exp001_profile_online.pyc` |
| 设备侧固定 WAV 重复运行 | 已报告：Paraformer 与 Zipformer 各 20 次；同模型重复文本一致，原始汇总见 `results/board-run-2026-10-13/Profil.md` |
| Paraformer CPU E2E/模块聚合计时 | 未执行：模型产物与 sherpa-onnx Python runtime 尚未准备 |
| Zipformer-RNNT CPU E2E/模块聚合计时 | 未执行：checkpoint/export 映射与模型产物待核实 |
| C++ RKNN runner 源码语法检查 | 通过：两个 runner 通过 `c++ -std=c++17 -I. -fsyntax-only`；两个 shell 脚本通过 `bash -n` |
| RKNN-enabled CMake 构建 | 未执行：需在有 RKNN headers/runtime 的目标构建环境验证 |
| 目标设备 E2E/module profiling | 已收到报告：各 20 次测量；系统为 Linaro ALIP / Linux 5.10.209 / aarch64；SoC、RKNN Runtime/驱动版本及实际 NPU 执行待核验 |
| CER/WER | 未执行：无固定标签清单 |

## 结论与限制

### C++ 模块插桩（本次补充）

- 新增 `sherpa-onnx/csrc/profiling.h`：默认关闭；设置 `SHERPA_ONNX_PROFILE=1` 后以 `std::chrono::steady_clock` 记录调用数、总时长、均值、最大值及 log2 纳秒桶估计的 p50/p95 上界。进程退出时写 CSV；输出路径由 `SHERPA_ONNX_PROFILE_OUTPUT` 指定，未指定则写 `sherpa-onnx-profile.csv`。高频路径用原子累加，不在每次调用时写日志。
- 覆盖模块：stream waveform/Fbank 输入；在线 Paraformer chunk preprocessing、encoder session、CIF/cache、decoder session；在线 Zipformer batch preparation、encoder/decoder/joiner session 与 greedy/modified-beam search；在线 RKNN Zipformer encoder/decoder/joiner 的 context wait、`rknn_inputs_set`、`rknn_run`、`rknn_outputs_get`；离线 Paraformer RKNN 的 encoder/predictor/acoustic embedding/decoder 与对应 RKNN API 调用。
- 本地运行示例：`SHERPA_ONNX_PROFILE=1 SHERPA_ONNX_PROFILE_OUTPUT=results/profile.csv ./your_cpp_recognizer <固定参数>`。若应用有 warm-up，在调用 warm-up 后、开始测量前 include `sherpa-onnx/csrc/profiling.h` 并调用 `sherpa_onnx::ResetModuleProfileCounters()`；需确保推理线程空闲。统计覆盖进程中后续的所有线程和 stream。
- 面向断网 NPU 设备的 runner、构建命令和结果文件说明见上文“RKNN 离线板端测试入口”。现已收到两种模型的设备侧 E2E/module profiling 汇总，原始报告已归档；runner 在目标板的构建过程、具体 SoC/runtime 版本和 NPU 硬件执行证据尚未由本仓库的运行日志独立核验。
- CSV 阶段相互嵌套：例如 `rknn.zipformer.encoder_total` 包含 context wait 和各 RKNN API 分项，不能把这些行相加。`rknn_run` 与 `rknn_outputs_get` 的时间是应用调用耗时；只有在板端且 Runtime 支持时再另采 `RKNN_QUERY_PERF_RUN/DETAIL`。近似分位数是 log2 桶上界，不等同精确样本分位数。
- 断网板端 runner 已完成语法检查；现已收到 Paraformer 与 Zipformer 的设备侧 profiling 汇总，但本次报告不含原始机器 CSV、构建日志和 Runtime/驱动身份，仍需补齐这些证据。Python 转换环境与 Toolkit2 导入已经核验。修改只覆盖指定 Paraformer/Zipformer 路径，其他 recognizer/backend 未插桩。

目前已有设备侧单 WAV 重复运行和模块 profiling 汇总，可作为初步性能记录，但 Exp001 尚未完成正确性/性能验收：缺少带标签测试集、CER/WER、完整设备/RKNN Runtime/驱动信息和独立 NPU 执行证据。

## 证据与关联提交

- 源码：[profile_online.py](profile_online.py)
- 设备侧汇总报告：`results/board-run-2026-10-13/Profil.md`（原始用户提供文件）；结构化逐次 E2E 与模块 CSV 尚待从设备输出目录补入。
- 当前仓库提交：`9db1af1871ed78cee9357c23898e5a7d8702d21a`；本次工作尚未提交。
