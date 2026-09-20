---
title: MindTalker · 端侧 Omni LLM 数字人对话系统立项
type: project
tags: [mindtalker, 数字人, minimind-o, flashhead, hidden-bridge, 求职项目]
created: 2026-05-08
revised: 2026-05-09 (v3 合并 - 吸收 11 份中间文档为单 spec)
status: 进行中（H0 done / H1 plumbing done / W2VHead 待 4000 段数据训完整版）
target_jd: 字节 Seedance/Seedream 推理加速 / 大疆 Insta360 智能创作 / 中金财富 AIGC
duration: 5 天面试弹药包（核心）+ 4-5 周完整 H1（可选）
related:
  - "[[01-EXPERIMENT-LOG]]"
  - "[[research/dji-world-model-uav/02-PROJECT-SPEC-v2.1-data-realigned|DJI 世界模型项目]]"
  - "[[job-prep/interviews/soulx-digithuman-cheatsheet]]"
---

# MindTalker · 端侧 Omni LLM 数字人对话系统

> 单卡 RTX 4090 上把 113M omni LLM ([MiniMind-O](https://github.com/jingyaogong/minimind-o)) 和 1.3B 数字人渲染器 ([SoulX-FlashHead Lite](https://github.com/Soul-AILab/SoulX-FlashHead)) 端到端串起来，**绕过 audio decoder + wav2vec2 重编码两次往返**。3-5 天交付面试弹药包（GitHub repo + arxiv tech report + demo 视频），4-5 周可选完整 H1 训练。

---

## 0. TL;DR

- **核心问题**：113M omni LLM 的 Talker 中间层 hidden state 能否旁路 audio decoder + wav2vec2 重编码，直接驱动 1.3B 视觉渲染器？
- **3 假设**：H0 layer-gated 7M Bridge MLP (✅ SUPPORTED) / H0.1 FlashHead 早层 weight bias (✅ SUPPORTED) / H1 A1.1 multi-head Talker（推理 ✅ 已打通，训练 4-5 周）
- **核心 mechanism**（双 finding 链）：(a) 合成 vs 真人 wav2vec2 早层 cos 0.85 / 晚层 0.99 + (b) FlashHead AudioProjModel 早层 weight 比晚层多 41% → "FlashHead 最依赖的层正好是 TTS 跟真人差最大的层" → layer-selective adapter 设计
- **当前产出**：8 段视频（B0/B1/B2×4/H1 random/H1 trained）+ 7 量化 metric + 22 git commits + tech report 8 sections + GitHub README
- **简历叙事**："单卡 4090 端到端 omni LLM 数字人系统，hidden bridge 直驱视觉，从 mechanism 分析到工程落地"

---

## 1. JD 对接

| 目标岗 | 命中点 |
|---|---|
| 字节 Seedance/Seedream 推理加速 | 单卡 4090 工程约束 / DiT × wav2vec2 introspection / layer-wise adapter 设计 / agentic workflow / 全 git 可复现 |
| 大疆 Insta360 智能创作 | 端侧实时性（96 FPS）/ 多模态对齐（audio→visual）/ 工程 + mechanism 双轮驱动 |
| 中金财富 AIGC | 独立完成端到端数字人系统（设计-实现-评测-写作）/ 稳重感 + 全栈能力 |

5 分钟讲法见 §11。

---

## 2. 研究问题（3 个可证伪假设）

### H0：layer-gated 7M Bridge MLP 在 wav2vec2 features 分布层修正合成-真人域漂移

- **训练**：125 段 Qwen3-TTS × 5 段真人 podcast 的 unpaired distribution matching loss
- **结果（run_010）**：50 epoch / 单卡 4090 / 5 min / per-layer cos B1 0.847 → B2 0.871（+0.024）
- **关键模式**：layer 1 +0.056 / layer 11 +0.001（早层修晚层不修，与 mechanism 预测一致）
- ✅ **SUPPORTED**

### H0.1：FlashHead AudioProjModel 对 wav2vec2 早层 weight 比晚层大

- **测量（run_007）**：解 `proj1.weight (512, 46080)` reshape 为 `(out, window=5, layer=12, dim=768)` 后 Frobenius norm 分组
- **结果**：layer 1 share 0.0918 (max) / layer 12 share 0.0651 (min) / **max/min = 1.41**
- ✅ **SUPPORTED**（H0 改进有 mechanism 头空间）

### H1：MiniMind-O Thinker bridge_states + W2VHead 7.68M 端到端驱动 FlashHead

- **推理路径（run_014）**：text → MiniMind-O 113M Thinker → bridge_layer 3 hidden → W2VHead → fake wav2vec2 features → FlashHead Lite → 视频。✅ **plumbing 已打通**
- **训练 spike（run_015）**：5 pseudo-pairs / 50 epoch / 5 min / L11 cos 0 → 0.93（接近真实 0.99）。第一段视频含 lip-sync 迹象
- **完整训练**：等 4000 段数字人口播视频到位（数据下载中），预计 ~15h GPU + ~3h agent

---

## 3. 数据集分层

### 3.1 主训练数据（即将到来）

- **4000 段网上数字人口播视频**（用户正在下载）
- **格式**：mp4/mkv/webm，预估 30-60s/段，720p
- **总大小预估**：~150-200 GB
- **域匹配优势**：数字人口播音频本身就是合成 / 工程化产出，与部署时输入分布一致（vs 真人录音是域错配）

### 3.2 域对照数据（建议混入）

- **真人 podcast/访谈**：500-1000 段，作为 "学到 TTS→TTS 还是语义→声学" 的 ablation 对照
- **现有 5 段 spike 数据**：podcast_sichuan / cantonese / scott / Wan2.2 talk / zero_shot — 跨集验证

### 3.3 训练数据构成

```
3000-3500 段数字人口播  (主训练 = 部署目标域)
+  500-1000 段真人音频  (域对照 / ablation)
+        5 段已有 spike  (held-out)
≈ 4000-4500 总段数
```

---

## 4. 技术路线（A1.1 multi-head）

### 4.1 H1 完整链路

```
text "今天天气不错…"
  ↓
MiniMind-O Thinker (113M, 8 层, 冻结)
  ↓ bridge_layer 3 hidden state (B, T_text, 768)
  │
  ├─ embed_proj × text_scale(3.0)             ┐
  │                                            ├ 双输入路径加权（保留原架构）
  ├─ audio_token_history × codec_proj × audio_scale(1.0) ┘
  ↓
MindTalker Talker (4 层 MiniMindBlock, 可训)
  ↓
  ├─ Head 1：原 Mimi codebook lm_head（保留 + λ_reg 衰减权重）→ 8 层 Mimi 码本（regularizer + 副产品仍能产语音）
  └─ Head 2（新加 ~7M）：W2VHead = pre LayerNorm+Linear+GELU + 12 个独立 Linear(768→768) + learnable layer_residual_scale
    ↓ 2× 线性插值（12.5Hz → 25fps）
    ↓ 推理时延迟 2 帧（80ms）补 audio_window=5 右侧
FlashHead Lite (1.3B, 冻结) — AudioProjModel + DiT 全部冻结
  ↓
512×512 视频 @ 25fps（96 FPS 推理）
```

### 4.2 关键参数（事实核查后）

| 项 | 值 |
|---|---|
| 可训参数 | Talker 4 层 + W2VHead 7.68M（+ Bridge MLP 7.09M for H0） |
| 冻结 | Thinker / AudioProjModel / DiT / wav2vec2 / Mimi decoder |
| `bridge_layer` | `num_hidden_layers // 2 - 1` = 第 3 层（默认 8 层 Thinker） |
| `text_scale, audio_scale` | `nn.Parameter(3.0), nn.Parameter(1.0)` 可学习 |
| 帧率 | wav2vec2 输出 25fps（FlashHead 内部已 linear_interpolation 对齐） |
| 上采样 | Talker 12.5Hz → 2× → 25fps（**不是 4×**，源码核查修正） |
| FlashHead AudioProjModel | flatten+Linear (5×12×768=46080→512)，**不是 cross-attention** |
| wav2vec2 模型 | `wav2vec2-base-960h` 英文版（**不是 chinese-wav2vec2-base**，源码核查修正） |
| MiniMind-O ASR | SenseVoice（**不是 Whisper**，源码核查修正） |
| MiniMind-O 训练资源 | 4×3090 / 4h（**不是单卡 3090 / 2h**，公众号核查修正） |

### 4.3 Loss 设计（multi-head）

```python
loss = 1.0 × loss_w2v_cos                                        # 主路径，per-layer cosine
     + 0.5 × loss_w2v_mse_norm  # var-normalized: ||pred - tgt||² / var(tgt)
     + λ_reg(t) × loss_mimi      # λ_reg 1.0 → 0.1 衰减；保留原 Mimi 输出能力 + 防过拟合
```

teacher-forcing 前半 → schedule sampling 后半（连续 features 累积误差对策）。

### 4.4 三档实现方案

| 方案 | Talker 输入 | Talker 输出 | 复杂度 |
|---|---|---|---|
| **A1.1（推荐）** | 保留 audio_token_history + bridge | 双 head：Mimi + wav2vec2 | low |
| A1.2 | history 改 wav2vec2 features | 双 head | high |
| A1.3 | 仅 thinker bridge（去 history） | 单 head wav2vec2 | mid |

---

## 5. 双 finding 链 mechanism（论文核心）

### Finding A · 合成 vs 真人 wav2vec2 per-layer cos

| Layer | 单段 (run_006) | 125 段 batch (run_008) |
|---|---|---|
| 1 | 0.596 | **0.847** |
| 6 | 0.808 | 0.950 |
| 11 | **0.996** | **0.996** |
| 12 | 0.942 | 0.973 |

→ 域漂移**集中在早层（声学）**，**晚层语义已饱和**。distribution-level gap 0.15（不是单段 0.40，诚实修正）。

### Finding B · FlashHead AudioProjModel 早层 weight bias

| Layer | 1 | 6 | 11 | 12 |
|---|---|---|---|---|
| weight share | **0.0918** | 0.0861 | 0.0685 | 0.0651 |

→ FlashHead **早层权重比晚层多 41%**。

### Finding A × B · effective domain exposure

| Layer | (1 - cos_dist) | weight share | exposure |
|---|---|---|---|
| 1 | 0.153 | 0.0918 | **0.0141** |
| 11 | 0.004 | 0.0685 | 0.00027 |

→ Layer 1 ~50× layer 11。**FlashHead 最依赖的层 = 合成-真人差最大的层**。

### Finding C · H0 实验复现 mechanism 预测形态

H0 真训后的 layer-selective 改善正是机制预测的形态（早 +0.06 / 晚 +0.001）→ mechanism 推理是真的，不是事后合理化。

---

## 6. GPU / 资源预算

### 6.1 当前状态

- 磁盘 824 GB free（不阻塞）
- GPU: 单卡 RTX 4090 24GB（间歇性独占可用，9-20GB free 看其他进程占用）
- 已有进程：gbrain TEI 1.5GB（保留）、其他用户进程

### 6.2 H1 完整流程预算（4000 段视频到位后）

| 阶段 | 物理时间 | Agent 编码 |
|---|---|---|
| 视频下载（已在跑）| 数小时 | 0 |
| ffmpeg 抽 16k audio（CPU 并行 8）| 1-2h | 0（已写好）|
| ASR 转写（SenseVoice on 4090）| 6-8h GPU | 0 |
| wav2vec2 features 抽取（fp16）| 1-2h GPU | 0 |
| W2VHead 训练（5000+ paired / 50-100 ep）| 3-10h GPU | 训练监控 30 min |
| 评测 + B2 / H1 视频生成 | 1-2h GPU | 30 min |
| GitHub repo 收尾 + tech report 填数 | — | 1-2h |
| **合计** | **~15h GPU + ~3h agent** | **<1 天连续** |

### 6.3 存储预算

```
原视频          ~150 GB
audio_16k         ~4 GB
wav2vec2 fp16   ~55 GB（4000 × 30s × 14 MB/s）
checkpoints      ~5 GB
合计           ~215 GB（远小于 824 GB free）
```

---

## 7. 5 天面试弹药包里程碑

| 天 | 工作 | 产出 | 状态 |
|---|---|---|---|
| **Day 1** | H0 真训 + H1 plumbing + W2VHead spike 训 | run_010-015，8 段视频 | ✅ 已完成 |
| **Day 2** | 等数据 + 写 Stage A pipeline 4 件 | preprocess / asr / build_paired / train_full | ✅ 已完成 |
| **Day 3** | 数据到位 → ASR + features + 真训完整版 W2VHead | 第一段 lip-sync 高质量 H1 视频 | ⏳ 等数据 |
| **Day 4** | 写 tech report 填 §5 实验数 / 录 demo 视频 | tech report PDF + B 站视频 | ⏳ |
| **Day 5** | 知乎 blog + Twitter thread + 5min 面试讲法默背 | 中英文传播 + 面试 ready | ⏳ |

**Day 3 在等用户的 4000 段数字人口播下载完毕**。

---

## 8. 三层评测协议

### Layer 1 · 重建 quality（features-level）
- per-layer cosine similarity (pred vs target wav2vec2 hidden_states)
- per-layer var-normalized MSE
- frame-level 音素边界对齐准确率（forced alignment 后）

### Layer 2 · 下游 lip-sync（视觉）
- SyncNet LSE-D / LSE-C（待装真模型，目前 stub + 视觉对比）
- FlashHead audio confidence threshold

### Layer 3 · 端到端用户感受
- B0 真音频 vs B1 朴素 TTS vs B2 + Bridge vs H1 multi-head 的 4-tier matrix
- MOS 主观评分（5-10 raters，可选）

### 4-tier baseline matrix

| Tier | 输入到 FlashHead | 角色 | 当前 |
|---|---|---|---|
| **B0** | Real audio → wav2vec2 | 上限 | ✅ 65s 视频 |
| B1 | TTS → wav2vec2 | 朴素 Phase 1 | ✅ 4s 视频 |
| B2 | + Bridge MLP 7M | H0 | ✅ 4 voice 视频 |
| **H1** | A1.1 multi-head | **主提案** | ✅ plumbing 视频，待真训 |

---

## 9. 工程环境

### Conda envs

| env | 用途 | 关键包 |
|---|---|---|
| `flashhead` | FlashHead 推理 / wav2vec2 / W2VHead 训练 | torch 2.7.1+cu128, librosa |
| `minicpm` | ASR (SenseVoice via funasr) / MiniMind-O 加载备用 | torch 2.6.0+cu124, funasr, onnxruntime |

**严格遵循 ~/.claude/CLAUDE.md pip 三步检查**。`PYTHONNOUSERSITE=1` 必带。

### 工作目录

以下为早期实验工作目录结构；本公开仓库对应其中的 `research/`，路径已改为通用示例。

```
<workspace>/mindtalker/
├── research/                       ← autoresearch workspace
│   ├── README.md                   ← GitHub repo entry
│   ├── src/                        ← 19 个 .py
│   │   ├── bridge_mlp.py           ← H0 BridgeMLPLight 7.09M (含 layer_gate)
│   │   ├── bridge_mlp_train_unpaired.py
│   │   ├── h1_w2v_head.py          ← H1 W2VHead 7.68M
│   │   ├── h1_w2v_head_train_full.py  ← 大数据训练（DataLoader + masking）
│   │   ├── minimind_talker_probe.py  ← H1 step 1
│   │   ├── generate_b2_video.py    ← H0 推理
│   │   ├── generate_h1_video.py    ← H1 推理
│   │   ├── feature_distribution_probe.py ← run_004
│   │   ├── h0_1_proj1_layer_norm.py ← run_007
│   │   ├── batch_extract_features.py ← run_008
│   │   ├── multi_voice_compare.py  ← run_013
│   │   ├── asr_transcribe.py       ← Stage A pipeline
│   │   ├── build_paired_dataset.py ← Stage A pipeline
│   │   ├── syncnet_eval.py         ← stub
│   │   ├── phase1_pipeline.py      ← wav2vec2 stage
│   │   └── ...
│   ├── scripts/preprocess_videos.sh ← Stage A pipeline
│   ├── data/                       ← 视频 / features / paired
│   ├── experiments/H0-bridge-mlp-baseline/  ← H0 ckpt
│   ├── experiments/H1-w2v-head/    ← H1 ckpt
│   └── paper/draft.md              ← arxiv tech report 8 sections
└── (Phase 1 实现，参考)
```

外部依赖（相对于本仓库根目录；也可通过 README 中的环境变量指定）：
- `../minimind-o/` — MiniMind-O 113M (216MB pytorch_model.bin from `jingyaogong/minimind-3o-pytorch`)
- `../SoulX-FlashHead/` — FlashHead Lite 1.3B + wav2vec2-base-960h

---

## 10. AI 接管 vs 用户介入

按 ~/.claude/CLAUDE.md "开发节奏" + 项目级 mindtalker/CLAUDE.md：

### Opus 4.7 完全接管
- 全部代码（pipeline / 训练 / 推理 / probe）
- 全部 vault 文档（含本 spec）
- 全部 git commit + progress report

### 用户介入（约 ~30 min 累计）
- 视频数据下载（物理 IO）
- 看 demo 视频做定性判断（"还行" / "有 lip-sync 迹象"）
- 关键决策（"H0 还是 H1 优先"、"投顶会还是 tech report"、"4000 段数据用真人还是数字人"）

### 估时公式

```
project_time = 物理过程（GPU/IO 不可压缩）
             + agent 编码（50-100× 加速）
             + 用户决策（~10 min/次）
```

→ 4000 段 H1 完整训练**不是 5-7 天**而是 **<1 天连续**（GPU 物理时间是真瓶颈）。

---

## 11. 面试 5 分钟讲法

```
30 秒：动机
"端侧 omni LLM × 数字人这条路被两次音频编解码绕远路。
我想验证 LLM 中间层能不能直接驱动视觉模型，绕过这两次往返。单卡 4090 上做完整研究和评测。"

1.5 分钟：核心是 mechanism 分析（双 finding 链）
"我没有先动手训模型。先做了两个 quantitative 探针：
  Probe 1: 合成语音 vs 真人 wav2vec2 12 层 features，发现早层 cos 0.85 / 晚层 0.996 ——
           TTS 跟真人差距集中在'声学'层，深层语义已饱和。
  Probe 2: FlashHead AudioProjModel 早层 weight 比晚层多 41%。
两个 finding 一乘 —— FlashHead 最依赖的层正好是 TTS 跟真人差最大的层。
这告诉我后续 adapter 应该 layer-selective 而不是 carpet-bombing。"

1.5 分钟：基于 mechanism 设计 + 验证
"基于 mechanism 设计了 7M layer-gated Bridge MLP 作为 sanity check baseline。
训练 50 epoch / 单卡 4090 / 5 min：layer 1 cos +0.056（最大修复）/ layer 11 +0.001（保持不动）。
这正是 mechanism 预测的形态 —— 早层修晚层不修。跨 4 voice 测试 pattern 一致（不是 cherry-pick）。
视觉上：B0 真音频 vs B1 TTS 朴素 vs B2 +Bridge —— frame-level 对比清晰，
Bridge MLP 在念字音节时让嘴型更夸张，安静时刻不动 —— phoneme-aware。"

1 分钟：诚实定位 + 工程
"BridgeMLP 不是 main contribution。FlashHead 训练用 782h 数据本身就有泛化。
BridgeMLP marginal 改善（+0.024 cos）反而是 mechanism 的硬证据 ——
能在 well-generalizing 模型上看到 layer-selective 小改善，说明双 finding 链是真的。
主线是 H1 multi-head Talker 端到端 —— 已打通推理路径，现在用 4000 段数字人口播训完整版。
工程：autoresearch 两层循环 + GAN harness 双 agent fact-check，15 实验 + 4 outer reflection，全 git 可复现。"

30 秒：开放性
"mechanism 推理在单 TTS 上 cos gap 0.40，125 段 batch 上稀释到 0.15 —— 两个数字都报了，没掩盖。
这教会我 distribution-level vs single-clip 测量的区别。
Reviewer 问的'是否 cherry-picked'，我用 4 voice 测试预先答了。"
```

---

## 12. 三个深挖问题预案

**Q1：Bridge MLP 7M 为什么不会过拟合？**
三层防御：(i) layer_gate init=0 起步是 identity 强 inductive bias；(ii) residual 连接保 baseline 下限；(iii) unpaired distribution-matching loss 是分布级而非样本级。

**Q2：为什么不直接 finetune FlashHead？**
计算量打不过（FlashHead 训练用 782h VividHead + 8×H800 + 1-2 周）。设计哲学是"在前向链路最便宜的位置做最小修正"。Bridge MLP 7M / W2VHead 7.68M 是这个最小修正点。

**Q3：agentic 工作流真的有用还是 hype？**
具体加速：GAN harness 找出原方案 4 处事实错误（人工 review 一周不一定查出）；autoresearch 两层循环让 GPU 阻塞期间持续推进 CPU 路径。trace 到 git commit。**取决于是否能给 evaluator 提供 ground truth** —— 能就有用，不能就退化。

---

## 13. 风险登记

| # | 风险 | 概率 | 应对 |
|---|---|---|---|
| 1 | 4000 段数字人口播 voice 单一（如全 ElevenLabs）→ W2VHead 学到 TTS→TTS | 中 | 混 500-1000 段真人 podcast 做域对照 |
| 2 | 视频带嵌字幕 / 水印 → face crop 包含字幕 | 低 | 检查样本，必要时手动 ROI 屏蔽 |
| 3 | ASR transcript 噪声 → paired data 错配 | 低 | 数字人音频 CER < 1%，比真人录音好 |
| 4 | Q3 数字人视频侵权风险 | 用户责任 | 用户自己评估来源 |
| 5 | W2VHead 不收敛（连续 features 累积误差）| 中 | teacher-forcing → schedule sampling 切换；fallback 到 per-frame contrastive |
| 6 | GPU 间歇被占，无法连续训 | 中 | 已有 kill GPU 进程权限 + 用户授权；checkpoint 每 5 epoch 保存 |
| 7 | bridge_layer 视觉最优 ≠ Mimi 最优 | 中 | Stage B' 重消融 layer 1-7（额外 0.5w） |

---

## 关联

### 实验细节
- [[01-EXPERIMENT-LOG]] — 15 inner-loop iter / 4 outer-loop reflection 完整时间轴
- [`paper/draft.md`](../paper/draft.md) — arxiv tech report 8 sections / 3500 字
- [`README.md`](../README.md) — GitHub repo entry

### 同生态
- [[research/digitalhuman/soulx]] — SoulX 三部曲合并报告（FlashHead 是其 Phase 2）
- [[github_learning/soulx-flashhead]] — FlashHead 源码学习笔记
- [[research/digitalhuman/pipeline/digithuman_v1.7.0_]] — 自研 v1.7.0 流式工程履历

### 求职 / 简历
- [[job-prep/interviews/soulx-digithuman-cheatsheet]] — SoulX 面试小抄
- [[research/dji-world-model-uav/02-PROJECT-SPEC-v2.1-data-realigned]] — DJI 平行项目（Phase 2 完成后启动）
- [[project-interview-prep]] — 求职备考主页

### 归档（中间过程文档，不更新）
- [[90-archive/]] — 含：phase2-final-plan / phase2-tldr / phase2-interview-strategy / phase2-feasibility-analysis-gan / phase2-feasibility-evaluator-report / phase2-joint-training / 14day-acceleration-plan / phase1-naive-pipeline / minimind-o-deep-analysis / 00-overview
