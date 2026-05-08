# Research Findings — MindTalker Phase 2

## Research Question

能否让一个 113M 参数的 Omni-modal LLM（[[MiniMind-O]]）的 Talker 中间层 hidden-state，旁路 audio decoder + wav2vec2 重编码两次"语义化—具象化"往返，直接驱动一个 1.3B 的 talking-head DiT（[[SoulX-FlashHead]] Lite）—— 单卡 RTX 4090 上完成全部训练与推理？

## Current Understanding

（截至 bootstrap，暂无实验数据。当前理解全部来自 vault 已有的事实核查文档。）

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

## Open Questions

- H0 Bridge MLP 5-15M 是否足以打过朴素链路？
- H1 A1 端到端能否打过 Bridge MLP？
- H3 Hidden Bridge 视觉驱动最优层是否仍是 layer 3？
- 12.5Hz → 25fps 2× 上采样是用线性插值还是 learned conv？
- 训练数据：MiniMind-O 自有 T2A 集 vs AISHELL-3 vs WenetSpeech，哪个最适合？

## Optimization Trajectory

（无实验，待第一个 run 后开始填表。）

| run_id | hypothesis | metric | delta vs baseline | wall_time | change |
|---|---|---|---|---|---|
| (pending) | | | | | |

## 关联 vault 文档

- [[phase2-final-plan]] — 最终方案凝练
- [[phase2-feasibility-analysis-gan]] — GAN harness 深度分析（A1/A2/A3 三档）
- [[phase2-feasibility-evaluator-report]] — Evaluator 14 条断言事实核查
- [[phase2-joint-training]] — 原始详细方案（已修订）
- [[phase1-naive-pipeline]] — Phase 1 朴素链路（H1 baseline 来源）
- [[github_learning/soulx-flashhead]] — FlashHead 源码学习笔记
