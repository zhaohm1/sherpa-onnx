# G2PW 中文 Piper TTS 模型支持 — v1.12.20 Backport

## 背景

sherpa-onnx v1.12.20 之后的版本（v1.13.0+）新增了对 **Piper 中文 TTS 模型** 的支持，这类模型使用 G2PW 拼音音素化而非 espeak-ng。

需要在 v1.12.20 上运行 `vits-piper-zh_CN-chaowen-medium-int8` 模型。

## 问题分析

### 上游依赖链

sherpa-onnx v1.13.2 中引入中文 Piper 支持的核心提交是 `d49e8f40`，但它依赖 5 个前置提交：

```
0b76a7df  Replace deprecated std::istrstream → std::istringstream
8bb52664  Replace deprecated std::wstring_convert → manual UTF-8 codec
926152b9  Refactor VITS to use the new Generate API (新增 GenerationConfig)
07c119a4  Use file path to initialize onnxruntime session (139个文件)
932f9164  Update openfst to v1.8.5 (新增 ReadFstsFromFar)
d49e8f40  Add two Piper Chinese TTS models  ← 目标提交
```

其中 `07c119a4` 改动 139 个文件，`926152b9` 引入 Generate API 重构。

### 方案选择

**不采用** 全量 cherry-pick，因为：
- 引入大量与中文 Piper 无关的变更
- 增加合并冲突和回归风险

**采用** 最小化 backport patch：将 G2PW 中文支持的核心逻辑直接移植到 v1.12.20。

## 模型原理

### 与传统 Piper 模型的差异

```
传统 Piper (英文等):
  输入文本 → espeak-ng 音素化 → PiperPhonemizeLexicon → token IDs → VITS

中文 Piper G2PW:
  输入文本 → 离线预生成 lexicon.txt → CharacterLexicon 查表 → token IDs → VITS
```

中文 Piper 不使用 espeak-ng，而是用 Python G2PW（基于 pypinyin）离线生成拼音 lexicon 映射表。运行时通过 `CharacterLexicon` 查表转换。

### BOS/EOS 机制

G2PW 模型训练时参照 [piper1-gpl/phonemize_chinese.py](https://github.com/OHF-Voice/piper1-gpl/blob/main/src/piper/phonemize_chinese.py#L296)，每句话首尾插入 BOS(`^`) / EOS(`$`) token：

```
传统 jieba:  [token_1, token_2, ..., token_n] [token_n+1, ...]
G2PW:        [^, token_1, token_2, ..., token_n, $] [^, token_n+1, ..., $]
```

## 代码变更

5 个文件，净增 **51 行**。

### 1. `offline-tts-vits-model-meta-data.h`

新增 `use_g2pw` 元数据字段：

```cpp
// for piper zh models using pinyin instead of espeak-ng
bool use_g2pw = false;
```

### 2. `offline-tts-vits-model.cc`

从 ONNX 模型元数据读取 `has_g2pw` 标记：

```cpp
SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.use_g2pw, "has_g2pw", 0);
```

### 3. `character-lexicon.h`

构造函数新增 `use_g2pw` 参数（默认 `false`，向后兼容）：

```cpp
CharacterLexicon(const std::string &lexicon, const std::string &tokens,
                 bool debug, bool use_g2pw = false);

template <typename Manager>
CharacterLexicon(Manager *mgr, const std::string &lexicon,
                 const std::string &tokens, bool debug,
                 bool use_g2pw = false);
```

### 4. `character-lexicon.cc`

四部分改动：

**a) Impl 构造函数** — 接收并存储 `use_g2pw_`：

```cpp
Impl(..., bool use_g2pw)
    : debug_(debug), use_g2pw_(use_g2pw) { ... }
```

**b) InitTokens** — 当 G2PW 模式时，从 tokens.txt 读取特殊符号 ID：

```cpp
if (use_g2pw_) {
    pad_id_ = token2id_.at("_");
    bos_id_ = token2id_.at("^");
    eos_id_ = token2id_.at("$");
}
```

**c) ConvertTextToTokenIds** — BOS/EOS 插入逻辑：

- 句首插入 `bos_id_`
- 遇到标点时，先插入 `eos_id_` 再切句，下一句以 `bos_id_` 开头
- 最后一句结尾插入 `eos_id_`

```cpp
if (use_g2pw_) {
    this_sentence.push_back(bos_id_);  // 句首
}
// ... 遍历 words ...
if (IsPunct(w)) {
    if (use_g2pw_) this_sentence.push_back(eos_id_);  // 标点前
    ans.emplace_back(std::move(this_sentence));
    if (use_g2pw_) this_sentence = {bos_id_};          // 下句开头
    else            this_sentence = {};
}
// ... 最后 ...
if (!this_sentence.empty()) {
    if (use_g2pw_) this_sentence.push_back(eos_id_);   // 结尾
    ans.emplace_back(std::move(this_sentence));
}
```

**d) 成员变量**：

```cpp
bool use_g2pw_ = false;
int32_t bos_id_ = 0;
int32_t eos_id_ = 0;
int32_t pad_id_ = 0;
```

### 5. `offline-tts-vits-impl.h`

Frontend 路由选择从仅检查 `jieba` 扩展为同时检查 `use_g2pw`，并将标记传入 `CharacterLexicon`：

```cpp
// 修改前
} else if (meta_data.jieba) {
    frontend_ = std::make_unique<CharacterLexicon>(
        ..., config_.model.debug);

// 修改后
} else if (meta_data.jieba || meta_data.use_g2pw) {
    frontend_ = std::make_unique<CharacterLexicon>(
        ..., config_.model.debug, meta_data.use_g2pw);
```

两处 `InitFrontend`（普通版和 Manager 模板版）均需修改。

## 编译

```bash
cd build-linux-x86_64
make -j$(nproc)
```

编译通过，0 错误。

## 验证

### 运行命令

```bash
MODEL_DIR="/path/to/vits-piper-zh_CN-chaowen-medium-int8"

./build-linux-x86_64/bin/sherpa-onnx-offline-tts \
  --vits-model="$MODEL_DIR/zh_CN-chaowen-medium.onnx" \
  --vits-tokens="$MODEL_DIR/tokens.txt" \
  --vits-lexicon="$MODEL_DIR/lexicon.txt" \
  --output-filename=/tmp/chaowen-test.wav \
  "你好，欢迎使用超文中文语音合成模型。"
```

注意：使用 `--vits-lexicon` 而非 `--vits-data-dir`（G2PW 模型不需要 espeak-ng）。

### Debug 输出确认

```
has_g2pw=1           ← 模型元数据正确读取
model_type=vits      ← 走 CharacterLexicon 路径
```

拼音查表输出：

```
你好 → n i 3 _ h ao 3 _
世   → sh i 4 _
界   → j ie 4 _
```

### 性能

| 输入 | 音频时长 | 耗时 | RTF |
|---|---|---|---|
| 你好，欢迎使用超文中文语音合成模型。 | 3.88s | 0.55s | 0.14 |
| 你好世界 | 1.05s | 0.16s | 0.15 |

RTF ≈ 0.14-0.15，CPU 推理速度约为实时的 7 倍。

## 使用说明

使用 G2PW 中文 Piper 模型时，需确保：

1. **lexicon.txt** 为 G2PW 拼音格式（可从 [tts-models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) 下载 `lexicon-zh-g2pw.txt`）
2. **tokens.txt** 包含 `^`、`$`、`_` 三个特殊 token
3. 命令行使用 `--vits-lexicon` 指定 lexicon 路径（**不要**使用 `--vits-data-dir`）

## 变更摘要

| 文件 | 改动行数 | 说明 |
|---|---|---|
| `offline-tts-vits-model-meta-data.h` | +3 | 新增 `use_g2pw` 元数据字段 |
| `offline-tts-vits-model.cc` | +1 | 读取 `has_g2pw` 标记 |
| `character-lexicon.h` | +3/−2 | 构造函数增加参数 |
| `character-lexicon.cc` | +36/−15 | BOS/EOS 逻辑 + ID 读取 + 成员变量 |
| `offline-tts-vits-impl.h` | +8/−0 | frontend 路由扩展 |

**总计: 5 files, +51/−17**
