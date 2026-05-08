# Literature Survey — MindTalker Phase 2

## 内部已有研究材料（vault 复制）

| 文件 | 角色 |
|---|---|
| `phase2-final-plan.md` | 最终方案凝练（A1 路线 + Bridge MLP baseline） |
| `phase2-feasibility-analysis-gan.md` | GAN harness Generator Round 2 深度分析（A1/A2/A3 三档） |
| `phase2-feasibility-evaluator-report.md` | Evaluator 14 条断言事实核查 |
| `phase2-joint-training-revised.md` | 原始详细方案（已修订） |
| `minimind-o-tech-report-summary.md` | MiniMind-O 微信公众号深度解读 |

## 外部论文（待补 — 进入 inner loop 后第一时间补充）

由于网络受限（实验环境网络），优先用本地源 + 已有 vault 笔记。需要新增 literature 时通过 Exa MCP / arXiv 检索。

### 关键论文优先级（按 fetch 顺序）

1. **MiniMind-O Technical Report** — `arxiv` 暂未公开，用公众号 + GitHub README 替代
2. **SoulX-FlashHead arxiv 2602.07449** — 已 vault 有学习笔记
3. **Mimi / Moshi (Kyutai)** — 12.5Hz × 8-layer RVQ 离散音频表示
4. **Wav2Vec 2.0 原论文** — 12 层 hidden states 语义层级
5. **Distillation 类**：DMD2、Self-Forcing、Neighbor-Forcing — vault 已有 SoulX wiki

## Gaps（重申）

1. 没人做过"sub-200M Omni LLM × >1B talking-head DiT 单卡联合训练"
2. 没人做过 Hidden Bridge 视觉驱动侧的 layer ablation
3. 没人做过 connected continuous-feature autoregressive 在 talking-head 链路的 publishability 评估

## 关联 vault

- [[research/digitalhuman/soulx]] — SoulX 三部曲合并报告
- [[research/digitalhuman/mindtalker/phase2-final-plan]] — 最终方案
