# Research Findings — MindTalker Phase 2

## Research Question

能否让一个 113M 参数的 Omni-modal LLM（[[MiniMind-O]]）的 Talker 中间层 hidden-state，旁路 audio decoder + wav2vec2 重编码两次"语义化—具象化"往返，直接驱动一个 1.3B 的 talking-head DiT（[[SoulX-FlashHead]] Lite）—— 单卡 RTX 4090 上完成全部训练与推理？

## Current Understanding

**截至 outer loop cycle 1（2026-05-08）**：

研究处于 **scaffolding 阶段**，3 个 inner-loop iter 完成。所有代码 skeleton 就位（BridgeMLP 模型 + Phase 1 wav2vec2 stage + 训练循环 + SyncNet stub），但**主指标 (SyncNet) 数据为零** — 阻塞在 GPU contention（24GB / 24.5GB used 跨两个 wakeup 周期）。

下一个关键 epistemic 问题：**MiniCPM-o-derived 音频的 wav2vec2 features 与真人音频的 wav2vec2 features 是否存在 measurable distribution gap?** 这是 H0 能否 supported 的根本前提。如果 gap ≈ 0，Bridge MLP 没有学习空间，H0 必然 refuted；如果 gap 显著，则进入"Bridge MLP 能否填上"的二阶问题。

**inner loop 的隐藏价值**：3 iter 中产生了 **2 个非平凡设计修订**（4-tier baseline matrix、BridgeMLP identity-init），都来自实操中遇到的具体问题，非空想推演。说明 GAN harness 双轮事实核查后的方案仍有改进空间。

**已知事实（来自 GAN harness 双轮核查）**：

1. **接口接口适配比想象的轻**：FlashHead 内部已用 linear_interpolation 把 wav2vec2 50Hz 卷积特征对齐到 25fps，再送入 transformer encoder。所以 12 层 hidden states 全部是 25fps × 768D。MiniMind-O Talker 原生 12.5Hz 只需 2× 上采样，不是 4×。
2. **AudioProjModel 内部是 flatten+Linear**（5×12×768=46080 → Linear → 512 → ... → 32×1536），不是 cross-attention。Cross-attention 发生在外层 DiT block。这意味着改 Talker 学单层 last_hidden_state 配合 12-layer up-projector 是最经济方案。
3. **autoregressive on continuous features 风险高于 discrete codebook**：Mimi 离散码本有 quantization "保护"，连续 features 的累积误差是开放问题。Teacher-forcing → schedule-sampling 切换是必要 mitigation。
4. **audio_window=5 推理含 ±2 帧未来上下文**：自回归生成时需要 80ms 延迟/padding 补齐右侧。

## Key Results

### run_004 · Wav2Vec2 Layer-12 Features Cross-Clip Cosine (5 real audios)

| | 001 | scott | canton | talk | zero_shot |
|---|---|---|---|---|---|
| 001 (Chinese podcast 65.9s) | 1.000 | 0.903 | 0.962 | **0.601** | 0.917 |
| scott (English 28.7s) | 0.903 | 1.000 | 0.928 | 0.848 | 0.933 |
| cantonese (37.5s) | 0.962 | 0.928 | 1.000 | 0.700 | 0.961 |
| talk (Wan2.2 example 5.0s) | **0.601** | 0.848 | 0.700 | 1.000 | 0.773 |
| zero_shot_prompt (3.5s) | 0.917 | 0.933 | 0.961 | 0.773 | 1.000 |

**off-diagonal mean cos = 0.8527, min = 0.601 (001 vs talk)**

Layer-12 norm range: 4.36 (001) — 7.53 (talk). **talk 显著 outlier**：layer-12 norm 高 70% + cos drop。Wan2.2/talk.wav 5s 短样本，可能含背景音乐 / 韵律 atypical。

**观察**：
- 同语种家族（001 podcast / cantonese 中文族）cos 0.962 — 几乎不变
- 跨语种（English ↔ Chinese）cos 0.90-0.93 — 小变化
- atypical audio （talk）vs normal speech：cos 跌至 0.60-0.85 — 大变化

**对 H0 的含义**：wav2vec2 layer-12 对说话人/语种弱敏感、对 audio-quality atypicality 强敏感。如果 MiniCPM-o 输出听上去是"普通合成语音"（无明显 artifact），它的 features 与真人 podcast features 的 cos sim 估计 > 0.93 — Bridge MLP 可学习 gap 偏小，但非零。如果有合成 artifact，gap 显著。**关键 next experiment：跑一次 MiniCPM-o 输出 vs 同句真人录音的 features 对比**（需 GPU）。

### run_005 · MiniMind-O 源码深度解析（关键设计修订）

读 `minimind-o/model/model_omni.py:288-312` 后**新发现**：

1. **Hidden Bridge 是计算式不是固定数**：`bridge_layer = num_hidden_layers // 2 - 1`（line 29）。8 层 Thinker → 第 3 层（与公众号一致）；如果换更深 Thinker，bridge 自动调整。设计支持 H3 ablation。

2. **Talker 不仅吃 thinker bridge，还吃历史 audio_ids embedding**（line 301）：
   ```python
   hidden_states = embed_proj(bridge_states) * text_scale(=3.0)
                 + codec_proj(talker_emb) * audio_scale(=1.0)
   ```
   两条路并行输入 Talker，learnable scalar 加权融合。**text 权重显著大（3.0 vs 1.0）**。

3. **Talker config**：4 层 MiniMindBlock + RMSNorm + TalkerHead + RoPE，hidden_size=768，8 codebook 共享 vocab 2112 (=2048 + pad/stop/spk specials)。

4. **TalkerHead 是低秩 adapter**：`audio_vocab_size=2112` 共享基座，每码本独立小 adapter（与 vault 公众号"低秩 adapter 共享基座"一致）。

**对 H1 (A1 方案) 的设计影响**：

原 phase2-final-plan A1 = "Talker 输出从 8 codebook 切到 wav2vec2 last_hidden_state"。但**忽略了输入端**——Talker 当前还以历史 audio token embedding 作为输入。改输出空间后，自回归输入怎么处理？

新增子方案：

| 方案 | Talker 输入 | Talker 输出 | 复杂度 |
|---|---|---|---|
| **A1.1** | 保留原 audio_ids embedding (即 Mimi token history) | last_hidden_state 768D | low — 双输出头并行 |
| **A1.2** | 改成 history wav2vec2 features (768D direct) | last_hidden_state 768D | high — 输入空间也变 |
| A1.3 | 仅 thinker bridge + 0 audio history | last_hidden_state 768D | mid — 无自回归 history |

→ **A1.1 最优 first try**：保留 audio token history（Talker 仍能并行预测 Mimi 码本作为 regularization / sanity），只额外加一路 wav2vec2 head。这是 phase2-final-plan 的 "joint head Y 路线" 与 A1 的合并版。

phase2-final-plan **下一次 vault 修订** 应该把 A1 拆成 A1.1/A1.2/A1.3，并默认推 A1.1。

### run_006 · 首个真实 B1-B0 metric — Per-layer 域漂移 pattern

**实验**：Qwen3-TTS Dylan 合成"今天天气不错，适合出门散步…"4.16s 中文 → wav2vec2 features (104, 12, 768)，与 podcast_sichuan_001 真人中文对比。

**Layer-12 整体 cos sim**：

| Pair | cos |
|---|---|
| TTS Dylan vs 真人中文 podcast | **0.9421** |
| TTS Dylan vs 粤语真人 | 0.9513 |
| TTS Dylan vs 英文 scott | 0.9378 |
| Reference 同语种真人 (run_004) | > 0.96 |
| Reference 跨语种真人 (run_004) | 0.90-0.93 |

→ TTS 落在跨语种和同语种之间，是"normal-sounding synthetic speech"，不像 atypical talk.wav (0.60)。

**核心 finding — Per-layer 渐变模式**：

| Layer | cos (TTS vs real_zh) | 解读 |
|---|---|---|
| 1 | **0.5961** | **早层声学层域差巨大** — TTS 与真人完全不同的低层 acoustic profile |
| 3 | 0.8052 | 音素层 |
| 6 | 0.8085 | 音节层 |
| 9 | 0.8763 | 中间层 |
| 11 | **0.9956** | **近完全一致** — 深层语义抽象 TTS≈real |
| 12 | 0.9421 | last layer，整体表征 |

**Insight**：合成 vs 真人 wav2vec2 域漂移**集中在早层**（acoustic / 韵律），**深层近 identity**（semantic）。FlashHead 的 AudioProjModel **flatten-mixes 全部 12 层**（5×12×768=46080 → Linear 512）→ **继承分层域漂移**。

**对 H0 (Bridge MLP) 的预判修正**：

| 预判版本 | 依据 | 预测 |
|---|---|---|
| 旧（pre-run_006）| layer-12 cos 0.94 | Bridge gap 偏小，H0 难显著提升 |
| **新**（post-run_006）| per-layer 渐变 | **早层有 0.4 cos gap，layer-gated Bridge MLP 7M 集中修复早层后可能显著提升** |

**Bridge MLP 设计修正**（已实现）：`BridgeMLPLight` 加 `layer_gate`（12 sigmoid 标量，init=0 → start as residual identity，让模型自动选择"修哪些层"）。早层 gate 学到大值，深层 gate 学到 ~0（pass through）。

**新的次假设 H0.1**：检查 FlashHead `AudioProjModel.proj1.weight`（46080→512 Linear），按 layer 分组算 norm。若早层 norm 大 → H0 + layer-gate Bridge 有清晰改进空间；若晚层 norm 大 → H0 大概率 refute。**这是下次 GPU window 第一个跑的实验**。

### run_007 · H0.1 决定性核查 — FlashHead 实际依赖哪些层？

加载 `SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors` 中的 `audio_proj.proj1.weight`（shape `(512, 46080)`），reshape 为 `(out=512, window=5, layers=12, dim=768)` 后 Frobenius 范数分组：

| Layer | Weight Norm Share | 趋势 |
|---|---|---|
| 1 | **0.0918** | **MAX** ← 声学层 |
| 2 | 0.0918 | |
| 3 | 0.0906 | |
| 6 | 0.0861 | |
| 9 | 0.0812 | |
| 11 | 0.0685 | |
| 12 | **0.0651** | **MIN** ← 语义层 |

Per-window uniform (~0.20 each)，单调早→晚 weight share 衰减，**max/min = 1.41**（早层比晚层多 41%）。

**核心结论 — H0.1 SUPPORT**：

把 run_006 + run_007 两个 finding 乘起来：

| Layer | TTS-vs-real cos (run_006) | FlashHead weight share (run_007) | "FlashHead 暴露给的有效 domain shift" |
|---|---|---|---|
| 1 | 0.596 | 0.0918 | **大 × 大 = 高暴露** |
| 6 | 0.808 | 0.0861 | 中 × 中 = 中等 |
| 11 | 0.996 | 0.0685 | ~0 × 小 = 几乎无 |
| 12 | 0.942 | 0.0651 | 小 × 小 = 几乎无 |

→ **FlashHead 主要受 TTS 早层 domain shift 影响**。layer-gate Bridge MLP 对早层做修正，理论上能直接降低 FlashHead 的 effective domain shift。**H0 不再被预测 refute，反而有清晰 headroom**。

**研究路线确认**：
- ✅ H0 (Bridge MLP) 值得训
- ✅ layer-gate 设计是必要的（不只是 nice-to-have）
- ✅ 如果训出来 H0 SyncNet 显著好于 Phase 1，就有定量解释（per-layer 双 finding）
- ✅ 如果训出来 H0 没好——说明上面机制有未发现的反作用（仍是论文 finding）

## Patterns and Insights

**截至 outer loop cycle 3（2026-05-08，after run_006）**：

1. **Per-layer 探针 >> 单 aggregate cos**：layer-12 cos 0.94 暗示"小 gap"，但 per-layer 显示早层 0.60 大 gap。如果只看 aggregate，会错失关键设计 insight。**Methodological lesson**：所有 cross-distribution 测试都应做 per-layer 分解，不能只看 last_hidden_state。

2. **一段 4 秒 TTS 改变研究方向**：从 "Bridge MLP 大概率 refute" 到 "可能显著提升 + 设计修正为 layer_gate"。这违反"实验需要大量样本"直觉——**当假设是关于 distribution shape 的，1-5 个样本就足以提供高信息密度的设计指导**。

3. **BridgeMLP 设计与 wav2vec2 layer 语义耦合**：早层 = 声学，深层 = 语义。合成语音的 domain shift 是"非平稳的"——深层 ≈ 真人，早层 ≠ 真人。任何 single-MLP 等权处理这 12 层都会浪费容量。layer_gate 是 minimal 修正。

4. **GPU contention 是间断的**：3 ticks 全占 → 1 tick 20GB free。要有"random-access GPU window"准备：所有 5min 内能跑的实验代码就绪。run_006 抓住这个 window。

## Lessons and Constraints

**从原 Phase 2 文档的 4 处事实错误学到的**（评审前自检清单）：

- FlashHead 用的是 `wav2vec2-base-960h`（英文版），不是 `chinese-wav2vec2-base` —— 始终从源码 `models/wav2vec2-base-960h/` 验证
- MiniMind-O ASR 是 SenseVoice，不是 Whisper —— 从公众号 / 论文一手信息验证
- MiniMind-O 真实训练资源是 4×3090 / 4h —— 不要用"按比例算"做粗糙推断
- 任何"直接喂"叙述都要展开到具体张量形状 + 内部 module 结构，否则容易掩盖工程难点

**从 inner loop iter 1-3 学到的**（autoresearch 实战）：

- **MiniCPM-o-2.6-int4 在 HF cache 里只有 1.1GB（不完整）**：去年某次中断的下载留下的尸体，不可用。改用 4.5-awq（12GB 完整）。**lesson**: 启动新研究第一步要先 `du -sh` 验证模型完整性，别假设 HF cache hit 就是好的。
- **flashhead env 的 pip 来自 ~/.local**：不能直接 `flashhead/bin/pip install`，会污染所有 py3.10 env（CLAUDE.md 明确警告）。**lesson**: 安装新依赖前用 `pip --version` 看 "from XXX" 路径。需要新依赖时建独立 conda env 或用 `PYTHONNOUSERSITE=1`。
- **flashhead env 没有 torchaudio**：但有 librosa + soundfile，可替代。**lesson**: 不要假设深度学习 env 都有 torchaudio。
- **GPU 全占跨 wakeup 持久化**：minicpm/flashhead/digithuman 进程是用户其他工作产物，**不能擅杀**。需要 wait-or-CPU 策略。**lesson**: research state 应该明确记录"GPU is shared with other work"作为 environment constraint，不要默认独占。
- **BridgeMLP identity-init 对 cos loss 起步效果显著**：dry-run 第一步 cos loss 仅 0.0506（vs 随机 init 通常 0.5+）。**lesson**: 当 task 是 "learn small distribution shift" 时，identity init + residual 是几乎免费的良好先验。
- **inner loop 自然产出设计反思**：4-tier baseline matrix 的 insight 出现在 iter 2 中，不是 outer loop。**lesson**: 不必死板等 outer loop 才反思——好的设计修订会从执行细节里冒出来。

## Open Questions

- H0 Bridge MLP 5-15M 是否足以打过朴素链路？
- H1 A1 端到端能否打过 Bridge MLP？
- H3 Hidden Bridge 视觉驱动最优层是否仍是 layer 3？
- 12.5Hz → 25fps 2× 上采样是用线性插值还是 learned conv？
- 训练数据：MiniMind-O 自有 T2A 集 vs AISHELL-3 vs WenetSpeech，哪个最适合？

## Optimization Trajectory

| run_id | hypothesis | type | metric | wall_time | change |
|---|---|---|---|---|---|
| run_001 | H0 | design-verify | n/a | 1 min | BridgeMLPLight 7.09M smoke test ✓ |
| run_002 | H0 | stage-spike | n/a | 2 min | wav2vec2 stage CPU spike ✓ (1648, 12, 768) |
| run_003 | H0 | scaffolding | loss 0.43→0.43 dry-run | 5 min | training loop + syncnet stub + val plan |
| run_004 | H0 | distribution-probe | cos sim 0.8527 | 3 min | 5-clip wav2vec2 layer-12 cosine matrix; talk outlier |
| run_005 | H0+H1 | B0 video + source | qualitative "OK" | 12 min | First real video B0; A1→A1.1 multi-head redesign |
| run_006 | H0 | **first real B1-B0 metric** | **0.9421 (layer-12)** | 6 min | Qwen3-TTS Dylan vs real_zh; per-layer pattern (L1=0.60, L11=0.996); BridgeMLP layer_gate added |
| run_007 | H0.1 | **decisive probe** | weight ratio 1.41 | 4 min | FlashHead AudioProjModel.proj1: early-layer share 0.0918 (max) vs late 0.0651 (min). H0.1 SUPPORT — H0 has clear headroom. |

primary metric (SyncNet) 仍未测出，受 GPU 阻塞。**Trajectory plot 留待首个真实 metric run 后绘制**。

## 关联 vault 文档

- [[phase2-final-plan]] — 最终方案凝练
- [[phase2-feasibility-analysis-gan]] — GAN harness 深度分析（A1/A2/A3 三档）
- [[phase2-feasibility-evaluator-report]] — Evaluator 14 条断言事实核查
- [[phase2-joint-training]] — 原始详细方案（已修订）
- [[phase1-naive-pipeline]] — Phase 1 朴素链路（H1 baseline 来源）
- [[github_learning/soulx-flashhead]] — FlashHead 源码学习笔记
