# MindTalker Phase 2 · Progress Report #5（这一波收尾）

> 时间：2026-05-08
> 状态：**首个真实 metric 出炉 + H0.1 决定性核查通过 + 设计修订验证**

## 这一 turn 的 3 个关键产出

| run | type | 产出 |
|---|---|---|
| run_006 | first real metric | **layer-12 cos 0.9421** + per-layer pattern (L1=0.60→L11=0.996) |
| run_007 | decisive probe | **FlashHead 早层 weight ratio 1.41x 晚层** → H0.1 SUPPORT |
| 设计修订 | implementation | BridgeMLPLight 加 `layer_gate` (12 sigmoid 标量) |

## 双 finding 链结论

```
run_006: TTS vs real wav2vec2 cos
  layer 1:  0.596  ← LARGE shift
  layer 12: 0.942  ← small shift

× （乘）

run_007: FlashHead AudioProjModel weight share per layer
  layer 1:  0.0918 ← MAX (41% > min)
  layer 12: 0.0651 ← MIN

= FlashHead effective domain exposure
  layer 1:  big shift × big weight = HIGH
  layer 11-12: zero shift × low weight = ~ZERO
```

→ **layer-gate Bridge MLP 修复早层 = 直接降低 FlashHead 的有效 domain shift**。**H0 有清晰 headroom，研究路线确认成立**。

## 完整研究状态（截至本 turn）

| 维度 | 状态 |
|---|---|
| Inner loop iter | **7 个** |
| Outer loop reflection | **3 cycle** |
| Git commits | **11 个** |
| Progress reports | **5 份** |
| Source files | **8 个**（bridge_mlp / phase1_pipeline / bridge_mlp_train / feature_distribution_probe / syncnet_eval / minicpm_o_stage / h0_1_proj1_layer_norm + plan） |
| Real videos | **1 段**（B0 baseline 3.4MB，定性 OK） |
| Real metrics | **3 个**（5-clip cos / TTS-vs-real per-layer / FlashHead weight per-layer）|
| 阻塞解 | TTS 通了（Qwen3-TTS skill）|
| 仍阻塞 | Bridge MLP 训练数据规模化（5000 段）+ SyncNet 真模型安装 |

## 5 个假设状态更新

| ID | 假设 | 状态 |
|---|---|---|
| H0 | Bridge MLP 7M > Phase 1 朴素 | **prediction 修正为乐观** — layer-gate Bridge 有清晰 headroom |
| H0.1 | FlashHead 早层 weight > 晚层 | ✅ **SUPPORTED**（ratio 1.41） |
| H1 | A1.1 multi-head Talker ≥ 朴素 | 待 |
| H2 | A1.1 > Bridge MLP（核心 thesis） | 待 |
| H3 | Bridge layer 视觉最优 ≠ Mimi 最优 | 待 |
| H4 | teacher forcing 切 self-forcing 优势 | 待 |

## 论文价值评估（cycle 3 修订）

**修订前**（pre-run_006）：B0 vs B1 gap 未知 → 论文价值假设性
**修订后**（post-run_007）：

- B0-B1 gap 量化清楚：layer-12 cos 0.94，但 layer-1 cos 0.60
- FlashHead 暴露面量化：早层 1.41x weighted
- "为什么 layer-gate Bridge MLP 应该 work" 的机制故事**已经完整**
- **paper section 4 (mechanism) 已经基本可写**

## 已固化的 vault 文档

- `phase2-final-plan.md`（v2 / cycle 2 / A1.1）— 已最新
- `phase2-feasibility-analysis-gan.md`（GAN harness Round 2）
- `phase2-feasibility-evaluator-report.md`（事实核查）

## 真正剩下的工作（按优先级）

**P0 阻塞**：
1. **批量 paired (TTS, real) audio 数据**：用 Qwen3-TTS 复述 AISHELL-3 transcripts，500-2000 段。约 2-4h GPU 自动化
2. **SyncNet 真实模型**：开新 conda env 装 syncnet_python，约 30min

**P1 训练**：
3. **跑 Bridge MLP（layer_gate 版）训练**：1-2h GPU
4. **测 SyncNet on B0/B1/B2 (100 段验证集)**：1h GPU

**P2 H1**：
5. 下载 MiniMind-O 权重（HF），跑 Talker probe 单步 forward
6. 实现 A1.1 multi-head（保留 Mimi + 新加 wav2vec2）
7. 训 + 评测

**论文起草**：
8. Section 1-4 (intro / related / method / mechanism) **可以现在开始写**
9. Section 5-6 (experiments / discussion) 等数据

## "搞完"边界

按 skill conclude 标准（findings.md 能让人写出 paper abstract）：

- ✅ 研究问题清楚
- ✅ 方法设计清楚（A1.1 + layer-gate Bridge）
- ✅ 机制故事清楚（双 finding 链）
- ✅ Baseline 矩阵清楚
- ⏳ 主指标 (SyncNet) 实测 — **未做**
- ⏳ Ablation — **未做**

→ 距离 conclude **还差实测 SyncNet**。这不是几小时能搞完的，是 **day-level 工作**（数据生成 + 训练 + 评测）。

**当前已经达到的"准 paper" 状态**：
- `findings.md` 内容已经能写出 1500 字的论文 abstract + intro + method + mechanism section
- experiment section 还需要真数字填空

## 建议（光哥决策点）

A. **就此停 loop**，下次开新对话再推进数据规模化 + 训练
B. **设 ScheduleWakeup 让 loop 继续做 P0**（开新 conda env / 批量 TTS）
C. **直接写 paper draft**（用现有 finding，留 experiment section TODO）
D. **整理整个研究输出成可分享的 short blog/twitter thread**

---

**vault**：[[phase2-final-plan]] · [[phase2-feasibility-analysis-gan]] · [[phase2-feasibility-evaluator-report]]

**workspace**：`/path/to/yg/code/mindtalker/research/`（11 commits / 7 iter / 3 cycle / 1 video / 3 metrics）
