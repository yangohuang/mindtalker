# MindTalker · 单卡 4090 跑通端到端 Omni LLM 数字人

> 把一个 113M omni LLM ([MiniMind-O](https://github.com/jingyaogong/minimind-o)) 和 一个 1.3B
> 数字人渲染器 ([SoulX-FlashHead Lite](https://github.com/Soul-AILab/SoulX-FlashHead))
> 在单张 RTX 4090 上端到端串起来，**绕过两次音频编解码往返**。

## 当前状态（2026-05-09）

| 阶段 | 状态 |
|---|---|
| Phase 1 朴素链路（MiniCPM-o + FlashHead）| 概念验证完成，B0 上限视频已生成 |
| H0 · 7M layer-gated Bridge MLP | ✅ 训练完成，per-layer cos +0.024 |
| H1 · 113M Talker → 7M head → 1.3B FlashHead 端到端 | ✅ **推理路径打通**（W2VHead 待训）|

## 一段视频对比

| 视频 | 链路 | 结果 |
|---|---|---|
| `data/flashhead_b0_001.mp4` | 真人音频 → wav2vec2 → FlashHead | 65s 上限 baseline |
| `data/flashhead_b1_001.mp4` | TTS → wav2vec2 → FlashHead（朴素）| 4s |
| `data/flashhead_b2_001.mp4` | TTS → wav2vec2 → **Bridge MLP** → FlashHead（H0）| 4s，phoneme 时刻嘴型更夸张化 |
| `data/flashhead_h1_001.mp4` | text → MiniMind-O → **W2VHead** → FlashHead（H1 plumbing）| 4s，链路通但未训 |
| `data/flashhead_b2_{vivian,ryan,uncle_fu}.mp4` | 多 voice B2 泛化 | 跨 voice 一致 |

抽帧：`data/video_frames/`

## 为什么这件事有意思

业界端侧 omni LLM 数字人的标准链路是：

```
LLM 输出 audio token → 解码成 24kHz 音频 → wav2vec2 重新编码成 features → 数字人渲染
                       ↑ 第 1 次往返          ↑ 第 2 次往返
```

两次"语义化—具象化"往返都是冗余的。我们想验证：**LLM 的中间层 hidden state 能否直接驱动视觉渲染器？**

## 双 finding 链 mechanism（H0 论证）

我们没有直接动手训模型，先做了两个 quantitative 探针：

**Probe 1（`run_006/008`）**：合成语音 vs 真人 wav2vec2 12 层 features 各层 cos：

| layer | 1 | 6 | 11 | 12 |
|---|---|---|---|---|
| cos (TTS vs real) | 0.85 | 0.95 | **0.996** | 0.97 |

→ 域漂移**集中在早层（声学）**，**晚层语义已饱和**。

**Probe 2（`run_007`）**：FlashHead AudioProjModel.proj1 各层 weight share：

| layer | 1 | 6 | 11 | 12 |
|---|---|---|---|---|
| weight share | **0.092** | 0.086 | 0.069 | 0.065 |

→ FlashHead **早层权重比晚层多 41%**。

**两个相乘**：FlashHead 最依赖的层正好是 TTS 跟真人差最大的层。所以：
- layer-gated Bridge MLP 应该自动学到"修早层、保留晚层"
- 7M 参数足够

**实证（`run_010`）**：
```
layer 1  cos +0.056  ← 早层最大修复
layer 11 cos +0.001  ← 晚层几乎不动
```

mechanism 预测的形态被复现 → H0 SUPPORTED。

## H1：真正"绕过两次编解码"

H0 BridgeMLP 是 features-level 修正，**没有省去 audio decoding**。H1 是真正的：

```
text/语义 → MiniMind-O Thinker (113M) → bridge_layer 3 hidden state
         → W2VHead (7.68M)
         → wav2vec2-style features (12 层 768D @ 25fps)
         → FlashHead Lite (1.3B, frozen)
         → 视频
```

**run_014 已打通推理路径**（10 分钟，单卡 4090）。剩下是 W2VHead 训练（4-5 周）。

## 一行复现

### B0 baseline（真音频 → 视频）
```bash
cd /path/to/yg/code/github/SoulX/SoulX-FlashHead
python generate_video.py --ckpt_dir models/SoulX-FlashHead-1_3B \
  --wav2vec_dir models/wav2vec2-base-960h --model_type lite \
  --cond_image examples/girl.png --audio_path examples/podcast_sichuan_16k.wav
```

### B2（H0 Bridge MLP 介入）
```bash
PYTHONNOUSERSITE=1 python src/generate_b2_video.py \
  --audio data/tts_b1_001_16k.wav \
  --bridge_ckpt experiments/H0-bridge-mlp-baseline/results/bridge_mlp_unpaired.pt \
  --save data/flashhead_b2_001.mp4
```

### H1 plumbing（端到端）
```bash
PYTHONNOUSERSITE=1 python src/generate_h1_video.py
# text → MiniMind-O → W2VHead → FlashHead → 视频
```

## 工作流（agentic）

整个研究通过 [orchestra-autoresearch](https://github.com/...) 两层循环 + GAN-style harness 双 agent fact-check 推进：

- **14 个 inner-loop iter** + **4 次 outer-loop reflection**
- **4 处 GAN harness fact-check 修订**（原方案的 wav2vec2 版本错误 / ASR 名称错误 / 训练资源错误 / 张量结构错误）
- **2 次研究方向修订**（A1 → A1.1 multi-head；H0 是 mechanism 论证不是产品）
- **完整 git 历史可复现**

## 文件结构

```
research/
├── README.md                          ← 本文件
├── paper/
│   └── draft.md                        ← arxiv tech report draft（8 sections）
├── src/
│   ├── bridge_mlp.py                  ← BridgeMLPLight 7.09M（含 layer_gate）
│   ├── bridge_mlp_train_unpaired.py   ← H0 unpaired domain matching 训练
│   ├── phase1_pipeline.py             ← wav2vec2 stage
│   ├── feature_distribution_probe.py  ← run_004 5-clip cos 矩阵
│   ├── h0_1_proj1_layer_norm.py        ← run_007 FlashHead weight introspection
│   ├── batch_extract_features.py       ← run_008 125 段 batch features
│   ├── multi_voice_compare.py          ← run_013 多 voice cos 对比
│   ├── syncnet_eval.py                  ← SyncNet 接口（stub）
│   ├── minicpm_o_stage.py              ← MiniCPM-o wrapper
│   ├── generate_b2_video.py            ← H0 推理：TTS+Bridge → FlashHead
│   ├── minimind_talker_probe.py        ← H1 step 1：113M LLM bridge hook
│   ├── h1_w2v_head.py                  ← H1 step 2：A1.1 W2VHead 7.68M
│   └── generate_h1_video.py            ← H1 step 3：端到端 H1 推理
├── experiments/
│   └── H0-bridge-mlp-baseline/
│       ├── protocol.md                 ← H0 实验 protocol
│       └── results/
│           ├── bridge_mlp_unpaired.pt   ← 7M trained Bridge MLP
│           └── training_log.json
├── data/
│   ├── flashhead_b0_001.mp4            ← B0 上限视频
│   ├── flashhead_b1_001.mp4            ← B1 朴素链路视频
│   ├── flashhead_b2_001.mp4            ← B2 H0 视频（dylan voice）
│   ├── flashhead_b2_{vivian,ryan,uncle_fu}.mp4
│   ├── flashhead_h1_001.mp4            ← H1 plumbing 视频
│   ├── tts_*_16k.wav                   ← 4 voice TTS 输入
│   ├── spike_*_features.npy            ← 真人 audio wav2vec2 features
│   ├── tts_batch_features/             ← 125 段 TTS features
│   └── video_frames/                   ← 抽帧对比
├── literature/                          ← vault 设计文档复制
├── to_human/                            ← progress reports 1-5
├── research-state.yaml                 ← 实验中央状态
├── research-log.md                     ← chronological 决策日志
└── findings.md                         ← 综合发现
```

## 性能数字

| 项 | 值 |
|---|---|
| 总参数 | MiniMind-O 113M + W2VHead 7.68M + BridgeMLP 7.09M + FlashHead 1.3B |
| 单段推理时间 | B0 65s 视频 / 总 GPU 时间 ~30s；B2 4s 视频 / 总 ~3s；H1 4s 视频 / 总 ~5s |
| 显存峰值 | ~10GB（4090 24GB 余量充足）|
| FlashHead Lite 帧率 | 96 FPS（论文公布；本机实测每 denoise step ~30ms）|

## 引用 / 致谢

本工作使用：
- [MiniMind-O](https://github.com/jingyaogong/minimind-o) (jingyaogong) — 113M omni LLM
- [SoulX-FlashHead Lite](https://github.com/Soul-AILab/SoulX-FlashHead) — 1.3B 数字人渲染器
- [Qwen3-TTS](https://github.com/QwenLM/Qwen-TTS) — TTS 用于生成 H0 训练数据
- [orchestra-autoresearch](https://github.com/...) — 两层循环 agentic 研究编排

## 详细 tech report

`paper/draft.md` 含 abstract / intro / method / mechanism / experiments / discussion 8 个章节，~3500 字。
