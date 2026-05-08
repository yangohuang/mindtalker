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

（无实验，待 H0 Bridge MLP baseline 跑出。）

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

primary metric (SyncNet) 仍未测出，受 GPU 阻塞。**Trajectory plot 留待首个真实 metric run 后绘制**。

## 关联 vault 文档

- [[phase2-final-plan]] — 最终方案凝练
- [[phase2-feasibility-analysis-gan]] — GAN harness 深度分析（A1/A2/A3 三档）
- [[phase2-feasibility-evaluator-report]] — Evaluator 14 条断言事实核查
- [[phase2-joint-training]] — 原始详细方案（已修订）
- [[phase1-naive-pipeline]] — Phase 1 朴素链路（H1 baseline 来源）
- [[github_learning/soulx-flashhead]] — FlashHead 源码学习笔记
