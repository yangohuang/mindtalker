# MindTalker Phase 2 · Progress Report #2

> 时间：2026-05-08（loop tick #1）
> 状态：H0 inner loop iter 2 完成，GPU 阻塞中

## 本轮增量

| run | type | 结论 |
|---|---|---|
| run_001 | design-verification | BridgeMLPLight **7.09M params** smoke test 通过 ✓ |
| run_002 | stage-spike | Phase 1 wav2vec2 stage 跑通（CPU），65.92s audio → (1648, 12, 768) features，25fps 完美对齐 ✓ |

## 关键发现

### 1. Protocol 修订：单 baseline → 4-tier baseline matrix

原 protocol 只有"Phase 1 朴素链路"作为对照。run_002 后意识到这不够，改成 4 层：

| Tier | 输入 | 角色 |
|---|---|---|
| **B0** | Real audio → wav2vec2 | FlashHead 自身天花板 |
| **B1** | MiniCPM-o output → wav2vec2 | Phase 1 朴素链路 |
| **B2** | + Bridge MLP（H0） | 主假设 |
| **B3** | Bridge trained vs B0 features | Bridge 上限 proxy |

**核心洞察**：B0 vs B1 的 gap 定义了 Bridge MLP 真正可补偿的空间。**如果 B0 ≈ B1，H0 实验在开始前就 dead**——MiniCPM-o 输出的 wav2vec2 features 已经是 FlashHead 训练分布同源，无 domain gap 可补。

### 2. 模型可用性盘点

| 模型 | 状态 | 路径 |
|---|---|---|
| FlashHead Lite 1.3B | ✅ 完整 | `../SoulX-FlashHead/models/SoulX-FlashHead-1_3B` |
| wav2vec2-base-960h | ✅ 完整 (英文版) | `../SoulX-FlashHead/models/wav2vec2-base-960h` |
| MiniCPM-o-2_6-int4 | ❌ **不完整**（仅 1.1GB，无 safetensors） | HF cache |
| MiniCPM-o-4_5-awq | ✅ 完整 12GB | HF cache |
| MiniMind-O | ⏳ 未下载 | github.com/jingyaogong/minimind-o |

→ Phase 2 后续切 4.5-awq 而非 INT4。phase2-final-plan 也建议 4.5（530ms 首帧 vs 1.1-1.6s）。

## 阻塞

**GPU 全占**（24GB used，700MB free）：
- minicpm 进程 14GB（用户其他工作，不能动）
- flashhead 进程 5.7GB
- digithuman 1.5GB
- TEI embedding router 1.5GB

→ FlashHead 推理（需 ~3GB） + MiniCPM-o-4.5 推理（需 12GB）必须等 GPU 空闲。当前 turn 用 CPU 跑 wav2vec2 是 workaround。

## 下一步（按 GPU 可用性分支）

### 若下次 wakeup GPU 空闲
1. 跑 FlashHead 推理 (real audio → 视频，B0 上限测试)
2. 跑 MiniCPM-o-4.5 (输出 24kHz audio, B1)
3. 出第一个数字（哪怕只 5 段验证集）

### 若 GPU 继续占
1. 写 SyncNet 集成（CPU 也能跑，模型 ~50MB）
2. 写 AISHELL-3 子集下载脚本（100 段验证集准备）
3. 写 BridgeMLP 训练 loop skeleton

## 三条线 commits

```
5cbb580 research(results): run_002 + 4-tier baseline redesign
b13f608 research(results): run_001 design verification
010867a research(protocol): H0 locked
eca7d6c research(init): MindTalker Phase 2
```

## 反思（pre-outer-loop）

按 skill 标准，3 个 inner-loop iter 后做 outer loop。已 2 个，下一个完成后会做正式反思。当前预判：

- **进展 vs 流程**：iter 1 是 design check（轻量），iter 2 是真实代码 + 真实 audio，价值在线
- **意外发现**：4-tier baseline 矩阵的设计修订是 outer-loop 级别的洞察提前到了 inner loop 里。说明 inner loop 应当包含设计反思，而不仅是机械执行
- **风险**：GPU 阻塞如果跨多次 wakeup 不解，研究进展会卡在 CPU-side prep，无法出真数字

下次 wakeup 后会更新这份反思。

---

**vault 关联**：[[phase2-final-plan]] · [[phase2-feasibility-analysis-gan]] · [[orchestra-autoresearch]]
