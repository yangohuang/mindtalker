---
title: MindTalker Phase 2 · 最终结合改造方案
type: project
tags: [数字人, mindtalker, minimind-o, flashhead, 联合训练, 最终方案]
created: 2026-05-08
status: 最终方案（GAN harness 双轮事实核查后）
phase: 2
related:
  - "[[00-overview]]"
  - "[[phase1-naive-pipeline]]"
  - "[[phase2-joint-training]]"
  - "[[phase2-feasibility-analysis-gan]]"
  - "[[phase2-feasibility-evaluator-report]]"
---

# MindTalker Phase 2 · 最终结合改造方案

> 经 [[phase2-feasibility-evaluator-report|GAN harness 双轮事实核查]] 后凝练的 [[MiniMind-O]] × [[github_learning/soulx-flashhead|SoulX-FlashHead]] 最终结合方案。详细推演见 [[phase2-feasibility-analysis-gan]]，原始方案见 [[phase2-joint-training]]。

## 一句话

**[[MiniMind-O]] Talker 学 wav2vec2 last_hidden_state（1 层 768D @ 12.5Hz）→ 2× 上采样 → 12-layer up-projector → 喂冻结的 [[github_learning/soulx-flashhead|FlashHead Lite]]。[[Bridge-MLP]] 当主对照 baseline 先发。**

---

## 完整改造链路

```
用户语音
    ▼
MiniMind-O Thinker（冻结，8 层 Transformer）
    │
    └─ 抽第 3/8 层 hidden state（Hidden Bridge，公众号 ablation 已验证）
    ▼
MindTalker Talker（4 层，可训）
    │
    ▼
wav2vec2 last_hidden_state · 768D @ 12.5Hz
    │
    ▼  2× 线性插值上采样
25fps × 768D
    │
    ▼
12-Layer Up-Projector（新加 ~7M 参数）
Linear(768→9216) + 12 个独立 small head
    │
    ▼
12 层 × 768D × 25fps，per-frame
    │
    ▼  推理时延迟 2 帧（80ms）补 audio_window=5 右侧上下文
FlashHead Lite（AudioProjModel + DiT 全部冻结）
    │ 内部：(B, 33, 5, 12, 768) → flatten+Linear → 32×1536 → DiT cross-attention
    ▼
512×512 视频帧 @ 25fps（96 FPS 推理）
```

---

## 关键参数

| 项 | 值 |
|---|---|
| 可训参数 | Talker 4 层 + up-projector ~7M |
| 冻结 | Thinker、AudioProjModel、DiT、wav2vec2 |
| 帧率对齐 | 12.5Hz → 25fps（**2× 上采样**，非 4×） |
| wav2vec2 模型 | `wav2vec2-base-960h`（英文版） |
| MiniMind-O ASR | SenseVoice |
| 推理延迟 | +80ms（audio_window=5 右侧 2 帧） |
| 显存峰值 | ~18-20GB（单卡 4090） |

---

## 训练流程

### Phase 2.0 · Bridge MLP baseline（1 周，先做）
- 不动 MiniMind-O，[[research/clawteam/audio-llm-digithuman-flow/minicpm-o-deep-analysis|MiniCPM-o]] 直出 24kHz 音频
- 训 5-15M Bridge MLP：wav2vec2 features → "FlashHead 偏好分布"
- **主对照**，A1 必须打过它才证明研究价值

### Phase 2.1 · A1 端到端（4-5 周）

| Stage | 时长 | 训练对象 |
|---|---|---|
| A | 1-2h | up-projector（Talker 冻结），teacher forcing |
| B | 1-2h | Talker 4 层 + up-projector，schedule sampling → self-forcing |
| B'（可选 0.5w） | — | Hidden Bridge layer 重消融（layer 1-7） |
| C（可选） | — | Thinker LoRA（仅 B 收敛但质量差时） |

**Loss**（按层方差归一化）：
```python
loss = sum_l [(1 - cos(pred_l, target_l)) + λ·MSE(pred_l, target_l)/var(target_l)]
```

**数据**：5000-10000 段 × 3-5s（AISHELL-3 + WenetSpeech 子集）

---

## 评测 baseline 矩阵

| Baseline | 角色 |
|---|---|
| 随机噪声 | SyncNet 下限 |
| **Bridge MLP（Phase 2.0）** | **主对照** |
| [[phase1-naive-pipeline|Phase 1 朴素链路]] | 工程对照 |
| A1 端到端（Phase 2.1） | 主提案 |

---

## 三种结果 → 三种论文

| A1 vs Bridge MLP | 论文价值 |
|---|---|
| 端到端 > Bridge > 朴素 | **强论文**：Hidden Bridge 直驱视觉假说成立 |
| 端到端 ≈ Bridge | 中性论文："5M Adapter 已足够"，工业界有用 |
| 端到端 < Bridge | 负结果论文：Bridge MLP 升主推荐，arxiv preprint |

→ 三种结果都能发，Bridge MLP 已先发兜底。

**论文标题候选**：

> *Bridging a 113M Omni-Modal Speech Model to a 1.3B Talking Head: Hidden-State Distillation Beats Audio Decoder Round-Trip on a Single Consumer GPU*

---

## 工作量

**5-6 周单人**（含 Phase 2.0 + 数据 + 训练 + 评测 + 写作 + 不含 Stage C）

---

## 立即下一步

跑 Phase 2.0 Bridge MLP，1 周内出对比结果，再决定是否上 Phase 2.1。

---

## 关联笔记

### 阅读路线
> [[00-overview]] → [[phase1-naive-pipeline]] → [[phase2-joint-training]] → [[phase2-feasibility-analysis-gan]] → **本文**

### 直接依赖
- [[00-overview]] — 项目总览
- [[phase1-naive-pipeline]] — Phase 1 朴素串联（工程对照）
- [[phase2-joint-training]] — 原始 Phase 2 方案文档（已被本方案修订）

### 事实核查证据链
- [[phase2-feasibility-analysis-gan]] — Generator Round 2 终稿，A1/A2/A3 三档对比 + 维度 A/B/C 拆解
- [[phase2-feasibility-evaluator-report]] — Evaluator 14 条断言事实核查报告

### 上游决策依据
- [[research/clawteam/audio-llm-digithuman-flow/feasibility-conclusion]] — Qwen2.5-Omni × DINet 同思路可行性
- [[research/clawteam/audio-llm-digithuman-flow/06-kill-criteria]] — 端到端 LLM 全军 FC 缺失的 kill criteria
- [[research/clawteam/audio-llm-digithuman-flow/minicpm-o-deep-analysis]] — MiniCPM-o 显存/延迟分析

### 同生态项目
- [[research/digitalhuman/soulx]] — SoulX 三部曲合并报告（FlashHead 是 Phase 2，训练资源 8×H800 反向证明全栈不可行）
- [[github_learning/soulx-flashhead]] — FlashHead 源码学习笔记（含 wav2vec2-base-960h、AudioProjModel 结构等关键事实）
- [[github_learning/soulx-flashtalk]] — FlashTalk 14B 完整版（FlashHead 前代）
- [[research/digitalhuman/pipeline/soulx-liveact_]] — LiveAct 单项目深挖（ConvKV 思想）

### 我方项目沉淀
- [[research/digitalhuman/pipeline/digithuman_v1.7.0_]] — v1.7.0 流式工程履历（vecmfcc 流式编码器思路可复用）

### 面试武器
- [[job-prep/interviews/soulx-digithuman-cheatsheet]] — 把 MindTalker 包装成"研究范式工程履历"

### 待补充链接（vault 中尚无）
- [[MiniMind-O]] — 待写概念页（Thinker-Talker 架构 + Hidden Bridge L3 + 三阶段训练）
- [[Bridge-MLP]] — 待写概念页（5-15M Adapter 模式）
- [[Hidden-Bridge]] — 待写概念页（Thinker 第 3 层 ablation 设计哲学）
- [[Mimi-Codebook]] — 待写概念页（Moshi/Kyutai 12.5Hz × 8 RVQ 离散音频表示）
- [[wav2vec2-base-960h]] — 待写概念页（英文 ASR finetune 版，FlashHead 实际使用）
