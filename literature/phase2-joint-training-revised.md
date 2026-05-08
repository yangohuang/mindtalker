---
title: MindTalker Phase 2 · MiniMind-O 联合重训
type: project
tags: [数字人, minimind-o, flashhead, 联合训练, 研究]
created: 2026-05-08
revised: 2026-05-08 (经 GAN harness 双轮事实核查修订)
status: 方案已修订，待评审
phase: 2
duration_estimate: 5-6 weeks（含 Phase 2.0 Bridge MLP baseline）
related:
  - "[[00-overview]]"
  - "[[phase1-naive-pipeline]]"
  - "[[phase2-feasibility-analysis-gan]]"
  - "[[phase2-feasibility-evaluator-report]]"
  - "[[SoulX三项目合并报告-事实核查版]]"
---

> **2026-05-08 修订说明**：本文档经 GAN harness 双轮（Generator-Evaluator）事实核查后修订，原稿 4 处事实错误已修正、维度 A 接口适配章节重写、新增 Phase 2.0 Bridge MLP baseline。完整 evaluator 报告见 [[phase2-feasibility-evaluator-report]]，深度分析见 [[phase2-feasibility-analysis-gan]]。

# Phase 2 · MiniMind-O 联合重训

> 把 MiniMind-O 113M 的 Talker 输出从 Mimi 码本切换到 wav2vec2 features，单卡 4090 全栈端到端训练，直接驱动 FlashHead。这是 MindTalker 的研究价值层。

## 核心研究问题

> Omni LLM 的内部表征能否旁路 audio decoder + 重编码，直接驱动外置视觉渲染器？

**反直觉钩子**：业界默认链路是 LLM → audio token → audio waveform → 重编码（wav2vec2）→ 视觉模型，做了两次"语义化—具象化"往返。MiniMind-O 因 0.1B 规模可全栈重训，是检验"能否绕过这层往返"的最佳载体。

## 为什么是 MiniMind-O 而不是 MiniCPM-o

| 维度 | MiniCPM-o 8B | MiniMind-O 113M |
|---|---|---|
| 全栈重训单卡 4090 | ❌ 显存放不下 | ✅ 1.5-3 小时 |
| 改训练目标 | LoRA only | 完全可改 |
| 训练数据可获 | 闭源 | 开源 + Parquet 格式公开 |
| 架构透明度 | 工业产品包装 | 教学级清晰（Thinker-Talker 两件套） |
| Hidden Bridge 已验证 | 没有 | ✅ 中间层（第 3 / 共 8）已 ablation |
| 短板 | 不可重训 | CER 8.97%（研究 demo OK） |

→ **MiniCPM-o 让你做工程师，MiniMind-O 让你做研究者**。Phase 2 的产出是 paper + 训练代码，不是 GitHub release。

## 训练目标设计

### 原 MiniMind-O 的 Talker

```
Thinker（MiniMind 主干，8 层）
    │
    └─ 抽第 3 层 hidden state ──┐
                                ▼
                           Talker（4 层）
                                │
                                ▼
                       8 层 Mimi 码本（离散）@ 12.5Hz
                                │
                                ▼
                          Mimi decoder
                                │
                                ▼
                           24kHz waveform
```

### 改造后的 MindTalker Talker（A1 推荐方案）

```
Thinker（保持不变）
    │
    └─ 抽第 3 层 hidden state ──┐
                                ▼
                       MindTalker Talker（4 层）
                                │
                                ▼
              wav2vec2 last_hidden_state 768D @ 12.5Hz
                                │
                                ▼  2× 线性插值上采样
                              25fps × 768D
                                │
                                ▼
                  12-Layer Up-Projector（新加 ~7M）
                  Linear(768→9216) + 12 个 small head
                                │
                                ▼
                12 层 × 768D × 25fps，per-frame
                                │
                                ▼  推理时延迟 2 帧（80ms）补 audio_window 右侧
            FlashHead Lite（AudioProjModel + DiT 全部冻结）
                                │
                                ▼
                          视频帧 @ 25fps
```

**关键改动（修订后，A1 推荐方案）**：
1. Talker 输出空间：**离散 Mimi 码本 → wav2vec2 last_hidden_state（1 层 768D）@ 12.5Hz**
2. 新增 12-layer up-projector：1 层 768 → 12 层 768，~7M 参数
3. 帧率对齐：12.5Hz → 25fps **2× 上采样**（非 4×；FlashHead 内部 wav2vec2 已对齐 25fps，见下文事实核查）
4. AudioProjModel 与 DiT 主干**全部冻结**
5. 推理时延迟 2 帧 / 80ms 补齐 audio_window=5 的右侧未来上下文

**为什么不是"12 层独立预测"**（事实核查后）：FlashHead 的 `AudioProjModel` 内部是 **flatten + Linear**（5×12×768=46080 → Linear → 512 → ... → 32×1536），不是 cross-attention，cross-attention 发生在外层 DiT block。Talker 直接输出 12 层等价于让单步回归出 9216 维向量（per frame），自回归累积误差风险极高。详见 [[phase2-feasibility-analysis-gan]] 维度 A 节的 A1/A2/A3 三档对比。

### Phase 2.0 · Bridge MLP baseline（新增，先做）

不动 MiniMind-O，**MiniCPM-o 直出 24kHz 音频**（同 Phase 1 链路），训一个 **5-15M Bridge MLP** 把 wav2vec2 features → "FlashHead 偏好分布" 修正后的 features：

- 训练量：单卡 4090 / 数小时
- 时长：1 周
- 角色：**主对照 baseline**，端到端 A1 必须打过它才能证明 Hidden Bridge 直驱视觉的研究价值
- 价值：先发的 short paper（即使 A1 失败，Bridge MLP 仍可独立成文）

### Loss 设计

由离散切到连续，loss 必须改：

**原 MiniMind-O loss**：
```
L_audio = sum over 8 codebooks of CrossEntropy(pred_logits, target_codes)
```

**MindTalker loss（修订后，A1 方案）**：
```python
# 一段语音同时跑两个编码器拿 paired data
# 注意：FlashHead 内部 wav2vec2 在 transformer 之前已经 linear_interpolation 到 25fps
# 所以 target hidden_states 全部是 25fps × 12 层 × 768
target_w2v_layers = wav2vec2(audio, fps=25, output_hidden_states=True).hidden_states[1:]
# shape: 12 × T_25fps × 768

# Talker 输出 12.5Hz × 768，经 2× 上采样 + 12-layer up-projector
pred_per_layer = up_projector(upsample_2x(MindTalkerTalker(thinker_hidden_l3)))
# shape: 12 × T_25fps × 768

# 按层方差归一化（早层声学/晚层语义尺度差异大）
loss = sum_over_layers(
    (1 - cosine(pred_l, target_l).mean()) +
    lambda * mse(pred_l, target_l) / target_l.var()
)
```

**关键修订点**：
1. target 是 **25fps**（FlashHead 内部已对齐），不是 50Hz
2. 12 层 loss 必须**按层方差归一化**，否则深层语义层会主导
3. 训练初期用 **teacher forcing**（喂 ground-truth wav2vec2 history），后期 schedule sampling 切 self-forcing — 连续 features 累积误差比离散码本风险显著高

**候选 B：per-frame contrastive loss**（替代原 KL distillation，因连续 features 无 logits）：
pred 与 target 同帧拉近 / 邻帧推开，作为 A 不收敛时的 fallback。

## 数据准备

### 配对数据生成

需要 (语音, MiniMind-O Thinker 中间层 hidden, wav2vec2 12 层 features) 三元组。

**数据源**（按公开度排序）：
1. MiniMind-O 自己的 T2A 训练集（45.7% 中文）— 用作内部一致性的最佳数据
2. AISHELL-3（中文，85 小时多说话人）— 公开免费
3. WenetSpeech（10000 小时但筛子集）— 公开免费

**目标规模**：5000-10000 段语音 × 平均 3-5 秒 ≈ 5-8 小时音频。

**事实修正**：MiniMind-O 真实预训练资源是 **4 块 3090 / 4 小时（≈16 GPU·h）**全栈，原文档"单卡 3090 / 2h"错误。本 Phase 2 只训 Talker（4 层）+ up-projector（~7M）+ 小数据集，不是全栈预训，单卡 4090 / 1.5-3h 仍合理但属于"改造微调"范畴。

**生成步骤（单卡 4090，预计 3-5 小时一次性）**：

```python
# 伪代码
for audio in dataset:
    # 1. 跑 MiniMind-O Thinker 抽中间层 hidden
    # MiniMind-O 实际用 SenseVoice 做 ASR（不是 Whisper）
    text = sensevoice(audio)
    thinker_hidden = minimindo.thinker(text, layer=3)  # T_text × 512

    # 2. 跑 wav2vec2，复用 FlashHead 同款 linear_interpolation 到 25fps
    # FlashHead 用的是 wav2vec2-base-960h（英文版），不是 chinese-wav2vec2-base
    video_length = int(len(audio) * 25 / 16000)
    w2v_features = flashhead_wav2vec2(audio, seq_len=video_length, output_hidden_states=True)
    # hidden_states[1:] = 12 层 × T_25fps × 768（已经是 25fps）

    # 3. 时间对齐：thinker text frames vs w2v 25fps audio frames
    # MiniMind-O 原 Mimi 是 12.5Hz；wav2vec2 target 是 25fps；text 是变长 token
    aligned = align_by_token_to_25fps(thinker_hidden, w2v_features)

    save_parquet({
        'audio_id': id,
        'thinker_hidden': aligned['thinker'],   # T × 512
        'w2v_features_25fps': aligned['w2v'],   # 12 × T × 768
    })
```

## 训练配置

### 硬件 & 显存

| 项 | 值 |
|---|---|
| GPU | 单卡 RTX 4090 24GB |
| Batch size | 16（按 MiniMind-O 单卡 3090 的配置缩放） |
| Mixed precision | bf16 |
| Gradient checkpointing | 是 |
| 预计显存 | ~18-20GB |
| 预计耗时 | 1.5-3 小时（参考 MiniMind-O 单卡 3090 / 2h） |

### Optimizer & schedule

```yaml
optimizer: AdamW
lr_thinker: 0  # 冻结
lr_talker: 1e-4
warmup_steps: 500
total_steps: 10000  # 约 5 epoch on 5000 段
weight_decay: 0.01
gradient_clip: 1.0
```

### 训练阶段（修订后）

按 MiniMind-O 三阶段渐进式策略改造：

**Stage A（1-2 小时）**：仅训新加的 12-layer up-projector
- 冻 Thinker + Talker 全部（保持原 Mimi 训练时的能力）
- 只训 up-projector（768 → 12×768）+ 2× 上采样器
- 目标：让 Talker last_hidden_state 1 层 → wav2vec2 12 层这件事先成立
- 配合 teacher forcing

**Stage B（1-2 小时）**：解冻 Talker 4 层 + up-projector
- 让 Talker 4 层重新对齐到 wav2vec2 last_hidden_state
- 这一步是真正的"Hidden Bridge → 视觉驱动"内化
- 训练后期 schedule sampling 切 self-forcing

**Stage B'（可选 0.5 周）**：Hidden Bridge layer 重消融
- 原 MiniMind-O 的 Bridge 第 3 层是为 Mimi 输出 ablation 出来的
- 改 wav2vec2 输出后，第 3 层是否仍最优？需要小消融（layer 1-7 各跑 100 step 看 loss）

**Stage C（可选）**：联合 LoRA Thinker
- 如果 Stage B 收敛但质量差，再放开 Thinker LoRA
- 风险：MiniMind-O 的语言能力可能受损

→ **推荐先做 A+B，B' 视消融成本决定，C 视效果决定**。

## 评测协议

三个层级 metrics，由近及远：

### Baseline 矩阵（修订后，新增主对照）

| Baseline | 角色 |
|---|---|
| 随机噪声 / mean features | 标定 SyncNet 下限 |
| **Phase 2.0 Bridge MLP（5-15M）** | **主对照** — 端到端 A1 必须打过它才证明 Hidden Bridge 直驱视觉的价值 |
| Phase 1 朴素链路（MiniCPM-o 直出） | 工程链路对照 |
| Phase 2.1 A1 端到端 | 主提案 |

### Layer 1 · 重建 quality

直接 metric：

- **Cosine similarity** (per layer)：pred vs target wav2vec2 features
- **Layer-wise reconstruction MSE**（按 target 方差归一化）：12 层每层独立 loss
- **Frame-level alignment accuracy**：音素边界对齐准确率

→ 验证模型学会了"映射"。

### Layer 2 · 下游唇形同步

接 FlashHead 推理（**冻结 AudioProjModel + DiT**），测：

- **SyncNet score** (LSE-D / LSE-C)：标准唇形同步指标
- **Confidence threshold**：FlashHead 自带的 audio confidence

→ 验证模型学到的 features 真的能驱动视觉。

### Layer 3 · 端到端用户感受

- **首帧延迟**：和 Phase 1 朴素链路对比（应有数百 ms 提升，但需扣除 audio_window 80ms 推理延迟）
- **MOS** 主观评分：邀请 5-10 人打分自然度
- **A/B 测试**：朴素链路 vs Bridge MLP vs A1 端到端，盲测哪个更像

→ 验证研究价值能转化为用户价值。

## 候选论文骨架

工作量评估：训练实验 1 周 + 评测 1 周 + 写作 1 周 = **约 3 周即可投 arxiv**。

**标题候选（修订后）**：

> *Bridging a 113M Omni-Modal Speech Model to a 1.3B Talking Head: Hidden-State Distillation Beats Audio Decoder Round-Trip on a Single Consumer GPU*

旧标题"Bypassing Audio Decoders"暗示完全替换音频解码，但 A1 方案实际是"用 wav2vec2 hidden-state 作为蒸馏 target"，新标题更准确。

**章节结构**：

1. Intro：业界数字人对话系统的"双重往返"问题（LLM token → audio → 重编码）
2. Related Work：MiniMind-O Hidden Bridge / SoulX FlashHead / Mimi 码本 / wav2vec2
3. Method：把 MiniMind-O Talker 改造成连续 features 输出器的具体方案
4. Experiments：单卡 4090 训练协议 + 三层评测
5. Discussion：什么 size 模型适合做这种"轻量蒸馏式重训"
6. Limitations & Future：CER 8.97% 限制 + Function Calling 缺失

**反直觉发现的可能形态**（未训练完前是猜测）：
- "MiniMind-O 的中间层 hidden 携带了足够的发音信息——证据是接到 FlashHead 后唇形同步分数与朴素链路相当甚至更高"
- 反向："MiniMind-O 的 Whisper 编码已把音频信号语义化，丢失了驱动唇形所需的低层声学信息——证据是 Stage B 收敛但 SyncNet 分数明显低于朴素链路"

→ **两种结果都是论文**。这是研究的最佳形态——你不知道答案，但你的实验能给出答案。

## 风险登记（修订后）

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | Stage B 不收敛（连续 features 累积误差比离散码本高） | 高 | teacher forcing → schedule sampling 切 self-forcing；fallback 到 per-frame contrastive loss |
| 2 | wav2vec2 features 和 FlashHead 训练分布有 gap | 中 | 用 FlashHead 实际指定的 **`wav2vec2-base-960h`（英文版）**（不是 chinese-wav2vec2-base）抽 features，且复用 FlashHead 同款 linear_interpolation 到 25fps |
| 3 | MiniMind-O CER 8.97% 影响整体 demo 可看性 | 中 | 训练时音频文本对齐用 ground truth 文本，不依赖 ASR |
| 4 | 单卡 4090 OOM | 低 | gradient checkpointing + batch 减半 |
| 5 | MiniMind-O 训练数据不公开导致无法精确复现 | 中 | 用公开 AISHELL-3 + WenetSpeech 子集替代 |
| 6 | 联合 LoRA Thinker 损害语言能力 | 中 | Stage C 默认不做 |
| 7 | A1 端到端打不过 Bridge MLP（thesis 反转） | 中 | Bridge MLP 已是 Phase 2.0 主对照，反转本身就是有价值的负结果论文 |
| 8 | audio_window=5 推理延迟 80ms 拖慢首帧 | 低 | 论文 limitation 章节明示，对话场景可接受 |

## 备选 Plan B（修订：已升格为 Phase 2.0 主对照）

修订前 Bridge MLP 是 fallback；修订后已升格为 **Phase 2.0 主对照 baseline**（详见前文）。理由：
- A1 端到端必须打过 Bridge MLP 才证明 Hidden Bridge 直驱视觉的研究价值
- 即使 A1 失败，Bridge MLP 仍可独立成一篇 short paper
- 1 周工作量，先发结果，降低 sunk cost 风险

## 与 Phase 1 的关系

Phase 1 是 baseline + 工程价值；Phase 2 是研究价值。

Phase 2 的所有评测都需要和 Phase 1 朴素链路做 A/B：
- 同硬件（单卡 4090）
- 同评测语料
- 同 FlashHead 权重
- 同前端

→ Phase 1 的 benchmark.py 直接复用为 Phase 2 评测脚本。

## 不在 Phase 2 范围内

明确**不做**：

- ❌ FlashHead 主干微调（SoulX 训练资源要求超出单卡 4090）
- ❌ 多说话人 / 多语种扩展（论文范围内做单语 + 单 voice clone）
- ❌ 视频质量优化（不是研究问题，留 Phase 3）
- ❌ 实时全双工对话（这是 MiniCPM-o 4.5 路径的事）

## 关联笔记

### 直接依赖
- [[00-overview]] — 项目总览
- [[phase1-naive-pipeline]] — Phase 1 的 baseline 代码 + benchmark 复用

### 研究上游
- [[SoulX三项目合并报告-事实核查版]] — SoulX 三部曲（FlashHead 是 P2，需要它训练资源数据反向证明全栈重训不可行）
- [[SoulX-LiveAct_项目分析]] — LiveAct 单项目深挖（ConvKV 思想可能在 Talker 也用得上）

### 概念页索引
- [[MiniMind-O]] — 概念页（Thinker-Talker 架构 + Hidden Bridge L3 + 三阶段训练）

### 待补充链接
- [[Mimi-Codebook]] — Moshi/Kyutai 的离散音频表示
- [[wav2vec2-base]] / [[chinese-wav2vec2-base]] — 12 层 features 编码器
- [[Hidden-Bridge]] — MiniMind-O 中间层桥接的设计哲学
