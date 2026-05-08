# MindTalker Phase 2 · Progress Report #3

> 时间：2026-05-08（loop tick #2）
> 状态：H0 scaffolding 完成 + outer-loop cycle 1 反思

## 本轮增量（run_003 + reflection）

### Scaffolding 完成

| 文件 | 内容 | 验证 |
|---|---|---|
| `src/syncnet_eval.py` | 3 backend 接口（stub/librosa/syncnet） | smoke test pass |
| `src/bridge_mlp_train.py` | 全训练循环（cos+mse_norm+identity-warmup loss） | dry-run 5 步 loss 单减 0.4317→0.4257 ✓ |
| `data/val_set_plan.md` | AISHELL-3 / WenetSpeech / spike 三档方案 | 文档 |

### 重要决策

- **SyncNet 真正模型暂缓装**：flashhead env 的 pip 来自 `~/.local/`，按 CLAUDE.md 这是污染源。改用 stub backend 走通 plumbing，等开新 conda env 后再装真 SyncNet
- **MiniCPM-o INT4 路径放弃**：本地只有 1.1GB（不完整），切 4.5-awq（12GB 完整）

## Outer Loop Cycle 1 反思

### 模式
- Inner loop 自然产出 **2 个非平凡设计修订**（4-tier baseline matrix + BridgeMLP identity-init）— 不是空想，来自实操
- GPU 阻塞跨 2 个 wakeup tick 持续，CPU prep 路线积累清晰

### 已 ruled out
- torchaudio 在 flashhead env（无）→ 用 librosa + soundfile
- MiniCPM-o-2.6-int4 路径 → 切 4.5-awq
- flashhead env 直接 pip install → 走 ~/.local 污染源

### 仍 open（核心 epistemic）
**MiniCPM-o 输出音频的 wav2vec2 features 与真人音频的 features 之间，存在多大 distribution gap？**

这是 H0 supported/refuted 的根本前提：
- gap ≈ 0 → Bridge MLP 没空间，H0 必然 refuted（仍是论文，但属负结果）
- gap 显著 → 进入 "Bridge MLP 能否填上" 二阶问题
- 当前数据 = 0，**完全卡在 GPU**

### Decision: DEEPEN on H0

不 pivot（H0 well-scoped）、不 broaden（premature）、不 conclude（无 metric）。

下一里程碑：**首个 B0 SyncNet 数字**。需要 ~5GB free GPU + SyncNet 装好。

## 数据状态

| run | type | 时长 | 价值 |
|---|---|---|---|
| run_001 | design-verify | 1 min | params count |
| run_002 | stage-spike | 2 min | (1648, 12, 768) features 形态确认 |
| run_003 | scaffolding | 5 min | 全 pipeline 代码就绪 |

**累计有效进度：8 分钟实操，6 个 git commit，3 份 progress report。**

## GPU 阻塞分析（Phase 2 进度的关键变量）

跨 2 wakeup tick 持续：
```
minicpm 进程 14GB（用户其他 session）
flashhead 进程 5.7GB
digithuman 1.5GB
TEI router 1.5GB
total 23.9GB / 24.5GB
```

**对光哥的请求**：如果其中某些进程已经不需要（特别是 minicpm 那 14GB），考虑收一下。
否则后续 wakeup 仍只能做 CPU prep，无法出 metric。一种解法：**搬到 H100 云** 或 **抢 4090 独占窗口**。

## 下一步（下次 wakeup）

**优先级**：
1. 检查 GPU 是否空了 → 跑 B0 baseline（即使没装 SyncNet 也可以先生成视频，肉眼看 lip-sync）
2. 否则 → 准备 minimind-o git clone（用户 vault 下未下，github.com/jingyaogong/minimind-o）
3. 或 → 跑 wav2vec2 batch 抽 features（podcast_sichuan 65s 已抽，可扩展到本地 5 段全部）

---

**vault 关联**：[[phase2-final-plan]] · [[phase2-feasibility-analysis-gan]] · [[orchestra-autoresearch]]

**git log**:
```
39cfd8c research(reflect): outer-loop cycle 1 — DEEPEN on H0
5cbb580 research(results): run_002 + 4-tier baseline redesign
b13f608 research(results): run_001 design verification
010867a research(protocol): H0 locked
eca7d6c research(init): MindTalker Phase 2
```
