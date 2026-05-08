# MindTalker Phase 2 · Progress Report #1

> 时间：2026-05-08
> 状态：autoresearch workspace 已 bootstrap，H0 第一步 design verification 通过

## 我现在在做什么

按 [[orchestra-autoresearch]] skill 的两层循环架构跑 [[phase2-final-plan|MindTalker Phase 2 研究]]。

研究问题（一句话）：**113M 参数的 [[MiniMind-O]] Talker 中间层 hidden-state，能否旁路 audio decoder + wav2vec2 重编码两次"语义化—具象化"往返，直接驱动 1.3B 的 [[SoulX-FlashHead]] talking-head DiT —— 单卡 4090 上完成？**

## 已完成

| 项 | 状态 | 文件 |
|---|---|---|
| Workspace 初始化 | ✅ | `/home/yg/yg/code/mindtalker/research/` |
| 5 个假设 H0-H4 定义 | ✅ | `research-state.yaml` |
| 文献库导入（5 份 vault 文档） | ✅ | `literature/` |
| H0 protocol locked + git committed | ✅ | `experiments/H0-bridge-mlp-baseline/protocol.md` |
| BridgeMLPLight 7M 实现 + smoke test | ✅ | `src/bridge_mlp.py` |

## 5 个假设

| ID | 假设 | 优先级 | 估时 |
|---|---|---|---|
| **H0** | 5-15M Bridge MLP 把 wav2vec2 features → "FlashHead 偏好分布" 后，SyncNet 优于裸 features | high | 4-8h |
| H1 | A1 端到端（Talker 学 last_hidden_state + 12-layer up-projector）SyncNet ≥ 朴素链路 | high | 6-12h |
| H2 | A1 SyncNet > Bridge MLP（核心 thesis） | high | 取决于 H0+H1 |
| H3 | Hidden Bridge 视觉驱动最优层 ≠ Layer 3（MiniMind-O 的 Mimi 最优层） | medium | 3-5h |
| H4 | Teacher-forcing → schedule sampling 切换 vs 纯 self-forcing 收敛对比 | medium | 2-4h |

## 第一个 run 结果

**run_001（design-verification）**：
- BridgeMLPLight params = 7.09M（达到 ~7M 目标）
- BridgeMLPHeavy params = 92.29M（fallback）
- I/O shape (B, T, 12, 768) 与 FlashHead 输入兼容 ✓
- 复用现有 `flashhead` conda env（torch 2.7.1+cu128，无需新建 env）

## 下一步（优先级排序）

1. **跑 Phase 1 朴素链路 + 测 SyncNet baseline** — 没 baseline 就没对照，必须先做（S3+S4，预计 1.5 天）
2. **下载 MiniCPM-o 2.6 INT4** — 7GB，本地慢网络
3. **准备 Bridge MLP 训练配对数据**（500-2000 段，AISHELL-3 子集）
4. **训 BridgeMLPLight 2-4 epoch**（轻量，单卡 ~2h）
5. **接 FlashHead 推理重测 SyncNet** → H0 SUPPORTED / REFUTED

## 三种结果都有论文价值（重申）

| H0 vs naive | H1 vs H0 | 论文形态 |
|---|---|---|
| H0 win | H1 win | **强论文** — Hidden Bridge 直驱视觉假说成立 |
| H0 win | H1 ≈ H0 | 中性论文 — "5M Adapter 已足够" |
| H0 lose | H1 win | "端到端必要性" 论文 |
| H0 lose | H1 lose | "wav2vec2 features 已是最优" 负结果 |

## 风险预警

- **网络慢**：模型下载是阻塞主路径，预留 0.5 天
- **训练数据无 ground truth**：Bridge MLP 的"FlashHead 偏好分布"target 不可知，已设计 self-supervised + paired audio shift 学法，但 Iter 1 前需用 30 段 spike 验证
- **SyncNet 灵敏度**：若指标对 features 修正不敏感，需切复合指标

## Agent Continuity

下一次 wakeup 会：
1. 检查这份报告是否仍 valid（research-state.yaml）
2. 推进下一步（首选：开始下载 MiniCPM-o INT4 in background）
3. 若长任务在跑就监控，不在跑就发起新任务

---

**对应 vault 笔记**：[[phase2-final-plan]] · [[phase2-feasibility-analysis-gan]] · [[phase2-feasibility-evaluator-report]]
