# handsoff.md — sherpa-onnx 部署项目交接

> 本文件只保留当前状态、跨实验结论和下一步工作；长期规则见 `AGENTS.md`，实验细节见对应 `experiments/*/README.md`。
> **维护约束**：后续会话仅更新已有章节内容，未经用户同意不调整此文件的章节框架或顺序。

## 1. 项目目标与范围

- **目标**：在端侧稳定运行 sherpa-onnx 语音模型，在识别正确性可接受的前提下，以可重复的 Profiling 数据指导性能优化和 CPU/NPU 异构部署决策。
- **当前平台**：RK3568，Debian，sherpa-onnx，RKNN NPU。
- **当前模型**：Paraformer-large-Chuan（离线 Paraformer，来源：`ASLP-lab/WSChuan-ASR`）；流式 Zipformer-RNNT（checkpoint：`csukuangfj/k2fsa-zipformer-bilingual-zh-en-t`）。
- **本阶段优先级**：先建立基准、厘清 runtime 组成与耗时，再讨论量化、算子及混合后端优化。

## 2. 当前进度

| 项目 | 状态 | 已知事实 / 需核实内容 |
|---|---|---|
| Paraformer RKNN 转换 | 已完成 | 已从用户指定 HF revision 本机导出 15 秒 ONNX 并转换为 RK3568 RKNN 三件套；板端加载和 NPU 执行待验证 |
| Zipformer-RNNT RKNN 转换 | 已完成（主机转换） | 使用 checkpoint 仓库 chunk=32 ONNX 三件套转换到 RK3568 RKNN；工具模拟器与 ONNX 输出形状及数值接近；板端加载和持续流式运行待验证 |
| 模型/环境清单 | 进行中 | Python 3.10.20、uv 与 RKNN-Toolkit2 2.4.2a15；两模型 checkpoint/revision、转换产物哈希已记录；Paraformer 权重 SHA 与 HF LFS 元数据一致；板端 Runtime/驱动待实测 |
| 正确性基线 | 未开始 | 待采集固定测试集上的 CER/WER 及 ONNX/RKNN 差异 |
| E2E 与模块级 Profiling | 进行中 | Python API 计时入口、C++ 模块插桩、支持 RKNN/ONNX 的固定 WAV runner 和复用 `sherpa-onnx-alsa` 参数的实时 ALSA runner 已实现；新增路径尚未完成构建和板端实测；见 `experiments/exp001_runtime_profiling/README.md` |
| NPU 算子级 Profiling | 未开始 | 先确认 RK3568 与 RKNN 工具链支持的采集路径 |
| CPU/NPU 对比 | 未开始 | 需统一输入、chunk、解码配置及线程等条件 |
| 混合后端验证 | 未开始 | 必须在瓶颈和调用/同步开销测清后再决定 |

> 上述“已完成（初步）”表示用户已报告功能性部署，**不代表本仓库内已有可追溯日志和性能验收结果**。首次检查仓库时应以实际证据纠正状态。

## 3. 关键结论与待验证假设

**目前已知**

- 已有模型部署路径，可以优先复用官方导出和推理实现，而非从零开始适配。
- Zipformer 模型包含 cache；但“缓存存在”“导出图分块可用”“sherpa-onnx 实际按块推理并输出”属于三个不同验证事项。

**尚未验证**

- Paraformer-large-Chuan 与 Zipformer-RNNT 在目标板上的识别正确性、实时性及长时间运行稳定性。
- Encoder、decoder、joiner、Fbank、搜索解码和 NPU 数据交互各自的主要耗时。
- CPU-only、当前 RKNN 路径及候选混合后端之间的量化差别。

**跨实验结论摘要**

- Exp002 已用 `/mnt/e/datasets/AEC-Challenge` 两条语音验证 Paraformer 与 chunk=32 Zipformer 的真实浮点 ONNX 输入采集；6 张量化 RKNN 均通过 Toolkit2 2.4.2a15 主机构建。转换日志仍有 `Unkown op target: 0` 诊断，板端加载、识别质量与性能待验证；详见 [Exp002 校准验证记录](experiments/exp002_rknn_conversion/README.md#2026-10-13-校准框架实音频验证)。

## 4. 下一步任务

### P0 — Exp001：端到端与模块级 Profiling

- CPU 公共 API 路径使用 `experiments/exp001_runtime_profiling/profile_online.py`；RKNN 断网板端用 `run_online_zipformer_rknn.sh` 测 online Zipformer，用 `run_offline_paraformer_rknn.sh` 测用户指定的离线 Paraformer-large-Chuan。RKNN 包和配套命令见 `experiments/exp002_rknn_conversion/README.md`。固定 WAV、模型、tokens 和输出目录。
- 固定 WAV 的板端 C++ runner 记录初始化、E2E/逐次 decode、重复转写一致性、模块 CSV、输入哈希和可读设备信息，并支持 RKNN 或 CPU ONNX provider；实时 ALSA runner 复用 `sherpa-onnx-alsa` 的模型参数与位置设备参数，并记录采集块、首个非空结果、endpoint 及模块 CSV。设备信息采集脚本为 `experiments/exp001_runtime_profiling/collect_device_info.sh`。计时范围与输出文件说明见实验 README。ONNX 新路径尚未完成构建和运行验证。
- 测量逐 chunk 的平均/p50/p95、RTF、峰值内存，并记录设备频率、温度、线程数及 warm-up 设置。
- **完成条件**：两种模型均有可重复的运行命令、结构化结果、模块耗时表和端到端时延；结果存入 `experiments/exp001_runtime_profiling/README.md`。

### P1 — Exp003：正确性及 CPU/NPU 对比

- 建立固定音频清单与文本规范；确认 Zipformer 是否真正流式。
- 对照 ONNX Runtime CPU 与现有 RKNN 路径，记录 CER/WER、RTF、p95 延迟和模块热点。
- **完成条件**：两条可比较基线及结论，详见未来的 `experiments/exp003_cpu_npu_baseline/README.md`。

### P2 — Exp004：NPU 深度定位与混合执行试验

- 根据前两项的热点选择候选优化，仅在具备工具支持时执行 RKNN 算子级 profiling。
- 对比选定的 CPU/NPU 模块分配策略，并显式计入同步与数据传输开销。
- **完成条件**：有对照实验支持“采用/不采用混合执行”的结论，详见未来的 `experiments/exp004_heterogeneous/README.md`。

## 5. 风险、阻塞与待确认事项

- 模型和工具链实际版本、sherpa-onnx 分支/commit、`librknnrt` 与 NPU 驱动兼容情况未记录。
- 流式 Zipformer 的 chunk/cache/状态接口及运行路径尚需核验。
- 当前运行路径的 CPU/NPU 模块归属未验证；不得先假定全部 encoder/decoder/joiner 在 NPU 上。
- 尚无统一测试集及正式 Profiling 结果；优化收益和瓶颈位置均待测。

## 6. 最近一次交接

- **日期**：2026-10-13
- **已知进展**：原有 float Paraformer/Zipformer RKNN 主机转换证据保留。Exp002 新增通用 WAV→float ONNX 图输入采集，使用两条 AEC-Challenge WAV 得到两模型校准列表，六张 int8 RKNN 均在 Toolkit2 2.4.2a15 主机成功构建。日志仍见 `Unkown op target: 0`；没有板端加载、转写精度或性能结果。
- **本次文档变更**：补充校准生成框架说明、依赖锁定、真实音频输入采集与六图主机量化构建证据；具体列表行数、模型哈希及失败尝试见 Exp002 README 与 `logs/calibration-framework-validation-2026-10-13/`。
- **下一次工作入口**：量化测试 NPY/RKNN 文件目前在 `/tmp/exp002-calibration-validation/`，必要时按 Exp002 README 命令重建；将所需 RKNN 模型和 profiling runner 复制到 RK3568 板端，验证加载、NPU 实际执行、转写/CER/WER、运行稳定性及设备状态，再回填证据。
- **关联实验**：`experiments/exp001_runtime_profiling/README.md`（profiling 入口已实现、待板端采集）；`experiments/exp002_rknn_conversion/README.md`（float 与 int8 主机转换、校准数据采集证据）。
