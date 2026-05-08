# Protocol H0 · Bridge MLP Baseline

> **Lock时间**：2026-05-08
> **状态**：locked, awaiting execution
> **类型**：CONFIRMATORY（在 phase2-final-plan.md 已规划）

## 假设

H0：训练一个 5-15M 的 Bridge MLP，把 [[MiniCPM-o]] 直出音频经过 wav2vec2-base-960h 编码后的 features，进一步映射到"FlashHead 偏好分布"。该 baseline 在 SyncNet 指标上**优于直接喂裸 wav2vec2 features**（即 Phase 1 朴素链路）。

## 预测（具体到指标）

- **强预测**：Bridge MLP 后 SyncNet LSE-C 提升 ≥ 0.3，LSE-D 下降 ≥ 0.5
- **弱预测**：至少不劣于裸 wav2vec2（LSE-C 持平 ± 0.1）
- **反预测**（refute H0）：Bridge MLP 反而劣化 SyncNet → 说明 wav2vec2 features 已经是 FlashHead 训练时的目标分布，无 domain gap

## 所需输入

1. **配对数据**：(音频片段, FlashHead-style wav2vec2 features 25fps × 12 层 × 768D, 真实数字人视频)
2. **音频源**：MiniCPM-o 2.6 INT4 直出 24kHz waveform → 重采样 16kHz → wav2vec2-base-960h
3. **target distribution**：FlashHead 训练数据（VividHead 子集）的 wav2vec2 features 分布

## 设计

```
MiniCPM-o 直出 24kHz audio
    │ (resample 24k → 16k)
    ▼
wav2vec2-base-960h（冻结）
    │ output_hidden_states=True
    ▼
12 层 × T_25fps × 768D（裸 features，作为 baseline）
    │
    ├─ 路径 A（baseline）: 直接喂 FlashHead → 测 SyncNet
    │
    └─ 路径 B（H0 候选）: Bridge MLP → 修正 features → 喂 FlashHead → 测 SyncNet

Bridge MLP 架构：
  Per-frame: (12, 768) → flatten → 9216
                       → Linear(9216, 4096) + GELU + Dropout(0.1)
                       → Linear(4096, 4096) + GELU + Dropout(0.1)
                       → Linear(4096, 9216)
                       → reshape (12, 768) + residual connection
  ~ 75M 参数 (large) / 收紧到 hidden=2048 时 ~14M (target)

  alt：Per-frame Per-layer 独立 Linear(768→768) × 12 + 1×1 conv across layers
       ~ 7M 参数（轻量版）
```

**初版选轻量 7M 版**，重型版作为 fallback。

## Loss

由于 target = "FlashHead 偏好的 wav2vec2 features"，但 ground-truth 的 "FlashHead 偏好" 不可知，先用**自监督代理**：

```python
# 路径 1: identity reconstruction warmup（让 Bridge MLP 学到接近恒等映射）
loss_warmup = MSE(bridge(x), x) × 100 step

# 路径 2: SyncNet adversarial（用 SyncNet 作为冻结 critic）
# pred_video = FlashHead(bridge(wav2vec2_features), reference_image)
# loss_main = -SyncNet_confidence(pred_video, audio)
# 因 FlashHead 不可微 → 用 reinforcement-style + paired data 替代
```

**实际可行 loss（Iter 1 决定）**：
- **Iter 1**：直接 supervised — 用真实数字人视频对应的 wav2vec2 features 作为 target，Bridge MLP 输出尽量接近。这退化为 identity reconstruction，应改为：
- **Iter 1 修订**：训练数据用 MiniCPM-o 输出和 FlashHead 训练集 paired audio 的 wav2vec2 features 之间的 distribution shift 学一个映射。但 FlashHead 训练集不公开 → 用 AISHELL-3 真实人声的 wav2vec2 features 作为 target，MiniCPM-o "复述" 同句 → 学映射。

→ **训练 protocol 待 Iter 1 实验前用小规模 30 段 spike 验证设计可行性**。

## 评测

| 指标 | 计算方式 | 目标值 |
|---|---|---|
| SyncNet LSE-D | 100 段 held-out | < Phase 1 baseline |
| SyncNet LSE-C | 100 段 held-out | > Phase 1 baseline |
| FlashHead audio confidence | 100 段平均 | > Phase 1 baseline |
| MOS（可选） | 5 人盲测 5 段 | > Phase 1 baseline |

**Baseline 来源**：Phase 1 朴素链路（即"裸 wav2vec2 features 直喂 FlashHead"）的 SyncNet 分数。**必须在 H0 之前测出 baseline**，否则无对照。

## 工作量分解

| 步骤 | 时长 | 阻塞依赖 |
|---|---|---|
| S1 · 环境搭建（conda env mindtalker，PyTorch 2.6 + CUDA 12.4） | 1 day | 无 |
| S2 · 模型下载（MiniCPM-o INT4 7GB + FlashHead 1.3B 3GB + wav2vec2-base-960h 0.4GB） | 0.5 day | S1 |
| S3 · Phase 1 朴素链路打通（一次性 demo 跑通） | 1 day | S2 |
| S4 · SyncNet baseline 测量（100 段验证集） | 0.5 day | S3 |
| S5 · Bridge MLP 训练数据准备（500-2000 段配对，GPU 抽 features） | 1 day | S2 |
| S6 · Bridge MLP 训练（轻量 7M 版，2-4 epoch） | 0.5 day | S5 |
| S7 · 接 FlashHead 推理 + 重测 SyncNet | 0.5 day | S6 |
| **合计** | **5 天**（中间会有空隙） | |

## 风险

1. **MiniCPM-o INT4 输出质量不达标** → 切 4.5 或回退到 BF16
2. **wav2vec2 features 分布与 FlashHead 训练分布无 domain gap** → H0 反预测，仍是有价值结果
3. **SyncNet 对 features 修正不敏感** → 改用 LSE-D / FaceShape 复合指标
4. **配对数据规模不够** → 用 wav2vec2 self-supervised pretrain 风格 augmentation

## 不在 H0 范围内

- ❌ MiniMind-O 任何重训（属 H1）
- ❌ FlashHead 任何 finetune
- ❌ 多说话人 / 多语言

## Definition of Done

- [ ] Phase 1 baseline SyncNet 已记录到 research-state.yaml
- [ ] Bridge MLP 7M 版训练 loss 收敛
- [ ] Bridge MLP 加入推理链路后 SyncNet 重测，结果记录到 results/
- [ ] analysis.md 写完，CONFIRMATORY 还是 EXPLORATORY 标注清楚
- [ ] 若 H0 SUPPORTED，进入 H1 protocol；若 REFUTED，做 outer loop 决定 PIVOT
