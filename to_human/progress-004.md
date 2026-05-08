# MindTalker Phase 2 · Progress Report #4

> 时间：2026-05-08（loop tick #3）
> 状态：**首个 quantitative finding 出炉** — 即使 GPU 持续阻塞

## 本轮增量

| run | type | 时长 | 价值 |
|---|---|---|---|
| run_004 | distribution-probe | 3 min | **首个量化数据** — 5 段真实音频 wav2vec2 layer-12 跨剪辑 cos 矩阵 |

并行任务：
- ✅ git clone `github.com/jingyaogong/minimind-o`（13MB）→ `/home/yg/yg/code/github/minimind-o/`
- ✅ wav2vec2 features × 4 段（cantonese / scott / zero_shot / talk）
- ✅ Cross-clip cosine + PCA 分析

## 关键 quantitative 发现

### Wav2Vec2 Layer-12 跨剪辑相似度矩阵（5 段真音频）

|  | 001 | scott | canton | talk | zero |
|---|---|---|---|---|---|
| 001 (中文 podcast 65.9s) | — | 0.903 | **0.962** | **0.601** | 0.917 |
| scott (英文 28.7s) | 0.903 | — | 0.928 | 0.848 | 0.933 |
| canton (粤语 37.5s) | 0.962 | 0.928 | — | 0.700 | 0.961 |
| talk (Wan2.2 5s) | **0.601** | 0.848 | 0.700 | — | 0.773 |
| zero_shot (3.5s) | 0.917 | 0.933 | 0.961 | 0.773 | — |

**off-diagonal mean cos = 0.8527**

### 三档敏感度

1. **同语种家族**（中文 podcast ↔ 粤语）cos **0.96+** — 几乎不变
2. **跨语种**（中 ↔ 英）cos **0.90-0.93** — 小变化
3. **atypical audio**（Wan2.2 talk.wav，疑似含音乐 / 异常韵律）cos **0.60-0.85** — 大变化
   - 它的 layer-12 norm = 7.53（其他 4.36-5.52，高 70%）→ 明显 out-of-distribution

### 对 H0 假设的初步推断

**核心 epistemic 进展**：

> 如果 MiniCPM-o 输出听上去是"普通合成语音"（无明显 artifact），它的 wav2vec2 features 与真人 podcast features 的 cos sim 估计 **>0.93**（类似跨语种程度）。**Bridge MLP 可学习 gap 偏小但非零**。
>
> 如果 MiniCPM-o 输出有合成 artifact / 韵律异常，gap 可能像 talk 那样大幅扩大。

**直接含义**：
- H0 (Bridge MLP) 不大可能像 SyncNet 5 个点那样大幅提升 — 可学习空间有限
- H0 仍可能小幅改善 - 如果 cos 0.93 → 0.99 转化为 ~0.5 SyncNet 提升
- H1 (端到端) 的优势可能体现在"绕过两次重编码"的延迟，而非 lip-sync 质量本身

→ phase2-final-plan 提到的"3 种结果各有论文"中，第 2 种（"H1≈H0：5M Adapter 已足够"）的概率被这个数据**轻微上调**。

## 三 ticks 阻塞统计

| Tick | GPU free (MB) | 决策 |
|---|---|---|
| #1 | 200 | wav2vec2 stage CPU spike + scaffolding |
| #2 | 200 | scaffolding (training loop + syncnet stub + val plan) |
| #3 | 192 | distribution probe + minimind-o clone |

**GPU contention 已成定式**。所有 CPU-side 工作每次 wakeup 仍能产出 1 个 inner-loop iter。

## 后续 GPU 解锁后（按优先级）

1. **跑 MiniCPM-o-4.5 复述** podcast_sichuan 同句话 → 24kHz audio → wav2vec2 → 与真人 features 计算 cos sim → **真正的 B1-B0 gap 数字**（~5min experiment）
2. **跑 FlashHead 推理** real audio → 视频 → 肉眼判断 lip-sync 上限（B0 视频版）
3. **抽 Bridge MLP 训练数据** ~500 段 paired (src, tgt) features

## 下次 wakeup 决策

GPU 未空 → 继续可做的 CPU 工作：
1. **解析 minimind-o 源码**：理解 Talker 4 层结构 + Hidden Bridge 取层位置（H1 起手必备）
2. **写 audio→features 批处理脚本**：为 Bridge MLP 训练数据生成铺路
3. **写 SyncNet 替代实现 plan**：对比 LatentSync sync_score / joonson SyncNet / 自实现轻量版

## 研究节奏感

第 4 个 inner-loop iter，**首个量化数据出来**。skill 推荐"3-5 iter 后做 outer loop"，下次 (run_005) 完后做 cycle 2 reflection。

cycle 2 重点会是：
- 我们对 H0 的预判从"未知"变成"小 gap 偏多"，研究方向是否需调整？
- 如果真是小 gap，**是否应当把 H1 提前优先做**（绕过 Bridge MLP 中间步骤）？
- 可能 PIVOT 决策点

---

**vault 关联**：[[phase2-final-plan]] · [[phase2-feasibility-analysis-gan]]

**git log**:
```
b12f0df research(results): run_004 first quantitative finding
39cfd8c research(reflect): outer-loop cycle 1 DEEPEN
5cbb580 research(results): run_002 + 4-tier baseline
b13f608 research(results): run_001 design verify
010867a research(protocol): H0 locked
eca7d6c research(init): MindTalker Phase 2
```
