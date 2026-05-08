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

## Patterns and Insights

（待 outer loop cycle 1 后填充。）

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

primary metric (SyncNet) 仍未测出，受 GPU 阻塞。**Trajectory plot 留待首个真实 metric run 后绘制**。

## 关联 vault 文档

- [[phase2-final-plan]] — 最终方案凝练
- [[phase2-feasibility-analysis-gan]] — GAN harness 深度分析（A1/A2/A3 三档）
- [[phase2-feasibility-evaluator-report]] — Evaluator 14 条断言事实核查
- [[phase2-joint-training]] — 原始详细方案（已修订）
- [[phase1-naive-pipeline]] — Phase 1 朴素链路（H1 baseline 来源）
- [[github_learning/soulx-flashhead]] — FlashHead 源码学习笔记
