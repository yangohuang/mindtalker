---
title: Phase 2 可行性分析 · Evaluator 事实核查报告
type: evaluator-report
tags: [数字人, mindtalker, gan-harness, 事实核查]
created: 2026-05-08
related:
  - "[[phase2-feasibility-analysis-gan]]"
  - "[[phase2-joint-training]]"
---

# Phase 2 可行性分析 · Evaluator 事实核查报告

## 评分汇总

| # | 断言（截短） | 评级 | 证据 |
|---|---|---|---|
| 1 | wav2vec2-base-960h（英文版）非 chinese-wav2vec2-base | CORRECT | `SoulX-FlashHead/models/wav2vec2-base-960h/`（实物） |
| 2 | MiniMind-O ASR = SenseVoice 非 Whisper | CORRECT | 公众号 L26 |
| 3 | MiniMind-O 真实是 4×3090 / 4h | CORRECT | 公众号 L50 |
| 4 | FlashHead 输入 (B, 81, 5, 12, 768) | MISLEADING | `flash_head_model.py:413,430` |
| 5 | Talker 输出 8 层 Mimi 码本 → 24kHz | CORRECT | 公众号 L26 |
| 6 | AudioProjModel intermediate=512, output=1536, ctx=32 | CORRECT | `flash_head_model.py:378-391` |
| 7 | wav2vec2 输出 50Hz | WRONG | `flash_head_pipeline.py:218` + `wav2vec2.py:32` |
| 8 | linear_interpolation 对齐到视频帧（25fps） | CORRECT | `flash_head_pipeline.py:206-218` |
| 9 | audio_window=5 自回归推理只有左侧上下文 | CORRECT (推理含义) | `flash_head_model.py:441-465` |
| 10 | Talker 通过 Hidden Bridge 读 Thinker 第 3 层 | CORRECT | 公众号 L34 |
| 11 | 8 个 Mimi 码本用低秩 adapter 共享基座 | CORRECT | 公众号 L40 |
| 12 | 12 层"同时使用"非 mean pool | CORRECT | `flash_head_model.py:485-547` (concat→Linear) |
| 13 | VAE 压缩 81→9，vae_scale=4 | CORRECT | `flash_head_model.py:411,431,446` |
| 14 | 训练数据 782h VividHead | UNVERIFIABLE | README 仅指 dataset 名 VividHead，未给小时数 |

## 详细核查

### #1 wav2vec2-base-960h
**评级**：CORRECT
**证据**：`models/` 目录直接列出 `wav2vec2-base-960h`（英文 ASR finetune 版），无 chinese-wav2vec2-base。`flash_head_pipeline.py:14` import `Wav2Vec2Model`，由 `wav2vec_dir` 参数加载。

### #2 ASR=SenseVoice
**评级**：CORRECT
**证据**：公众号 L26 "输入端的音频（SenseVoice）、图像（SigLIP2）、文本（Tokenizer）"。

### #3 4×3090 / 4h
**评级**：CORRECT
**证据**：公众号 L50 "全部训练在四块RTX 3090上四小时内完成"。

### #4 输入张量 (B, 81, 5, 12, 768)
**评级**：MISLEADING
**证据**：
- `flash_head_model.py:413` 注释 `context: torch.Tensor #(5, 33, 12, 768)`（无 B 维，T=33 frame_num）
- `flash_head_model.py:430` 注释 `# 输入: context (bsz, 81, 5, 12, 768)`
- `infer_params.yaml:1` `frame_num: 33`

**判定**：两处注释互相矛盾。Generator 文档照搬了 :430 的"81"但没说明这与 :413 的"33"差异。实际：`generate_video.py:141` 切 chunk 长度=frame_num=33，:413 注释才是单 chunk 真实形状；:430 的"81"可能是该函数被多 chunk 拼接调用时的 cache 形状，需更深入追踪。Generator 没意识到此矛盾，**应在文档中标注两种形状的语境**。

### #6 AudioProjModel 维度
**评级**：CORRECT
**证据**：`flash_head_model.py:376-391`：`audio_window=5 / intermediate_dim=512 / output_dim=1536 / context_tokens=32`，与文档完全一致。

### #7 wav2vec2 输出 50Hz
**评级**：**WRONG**
**证据**：`flash_head_pipeline.py:206-218`：`video_length = len(speech_array)*fps/sr`，`audio_encoder(..., seq_len=int(video_length))`；`wav2vec2.py:32` 在 feature_extractor 之后立即 `linear_interpolation(extract_features, seq_len=seq_len)`。即 wav2vec2 在送入 transformer encoder 之前就**先把 50Hz 卷积特征插值到 25fps**。后续 12 层 hidden states 全部是 25fps。
**判定**：文档把 "wav2vec2 原生 50Hz" 当成了 FlashHead 接到的输入，这是 wrong。FlashHead 接的就是 25fps × 12 层 × 768D。这个错误反过来推翻了文档"4× 上采样到 50Hz"的方案——根本不需要 50Hz。

### #8 linear_interpolation 对齐到 25fps
**评级**：CORRECT
**证据**：见 #7。`fps=25` 由 `infer_params.yaml:3 tgt_fps:25` 喂入。

### #9 audio_window=5 推理时无右侧
**评级**：CORRECT（作为 Talker 自回归生成时的隐含约束）
**证据**：`flash_head_model.py:441-465` 的 first/middle/last_of_group 切片说明 FlashHead **训练**用了双向 ±2 邻域（mid_idx=2，last_of_group 取 `[mid_idx:]` 即未来 3 帧）。Talker 自回归每步只能产生当前帧 features，未来 2 帧未生成，**结构上确实需要延迟或 padding**。

### #10 Hidden Bridge 第 3 层
**评级**：CORRECT
**证据**：公众号 L34 "中间层（默认第3层，总8层）恰好平衡"。

### #11 低秩 adapter 共享基座
**评级**：CORRECT
**证据**：公众号 L40 "嵌入层用共享的嵌入表加上每个码本的低秩Adapter，输出头用共享线性头加上每个码本的低秩Adapter"。

### #12 12 层是否同时使用 / 是否 mean pool
**评级**：CORRECT（同时使用，非 mean pool）
**证据**：`flash_head_model.py:485-547` `class AudioProjModel`：
- `:502 self.input_dim = seq_len * blocks * channels` (5×12×768=46080)
- `:509 self.proj1 = nn.Linear(self.input_dim, intermediate_dim)`
- `:520-522` `view(batch, window_size*blocks*channels)` 后过 proj1

**判定**：12 层和 5 帧 window 一起 flatten 成 46080 维向量过单个 Linear，**不是 mean pool，也不是 cross-attention，是"拼接 + Linear 投影"**。Generator 说"同时使用（不是 mean pool）"对，但应进一步明确"是 flatten+Linear，不是 attention"。这对 Phase 2 改造有重要影响：Talker 需要预测的是同一帧 12 层的 12×768=9216 维 + 5 帧 window = 46080 维，**单层 768 替代不能直接复用现有 proj1 权重**。

### #13 VAE 压缩 81→9
**评级**：CORRECT
**证据**：
- `flash_head_model.py:411` `x: (1, 16, 9, 64, 64)` 注释（latent 形状第3维=9）
- `:431` 注释 `81 帧 = 1 (第一帧) + 80 (后续帧, 每4帧对应VAE压缩后的1帧)`
- `:446` `vae_scale=4`

但注意：`infer_params.yaml:1 frame_num: 33`，对应 `(33-1)/4+1 = 9` latent。所以 frame_num=33 / latent=9 才是默认配置；81 帧是该函数支持的另一种规模。

### #14 782h VividHead
**评级**：UNVERIFIABLE
**证据**：README 多次提及 VividHead dataset（L15、L30）但未给具体小时数。Technical Report PDF 在 `assets/SoulX_FlashHead.pdf` 但本次未读。Generator 数字 782h 来自 Vault 已有笔记 `[[research/digitalhuman/soulx]]`，本次核查范围内无独立验证。

## 关键发现 / 新事实（Generator 未提）

1. **wav2vec2 50Hz 是错误**：FlashHead 内部已经把 wav2vec2 输出插值到 25fps，文档第 43 行"wav2vec2 输出 50Hz"和"50Hz 还是 12.5Hz 后插值"的整段讨论建立在错误前提上。**正确叙述**：FlashHead 期待的是 25fps × 12 层 × 768 hidden states。Talker 原生 12.5Hz，需要 2× 上采样到 25fps，不是 4×。

2. **AudioProjModel 是 flatten+Linear 不是 attention**：`proj1` 把 (window=5, layers=12, dim=768)=46080 维直接 Linear 到 512。这意味着改成"只学某一层"需要重设 input_dim=5×1×768=3840 + 重新初始化 proj1，**不能继承权重**。Generator 文档维度 A 的最小改动方案"Talker 输出限定为某一层"暗含此代价但没明说。

3. **frame_num=33 才是默认**：`infer_params.yaml` 定的是 33 帧/chunk（latent=9），不是 81 帧。Generator 把 :430 注释当主形状，但实际 inference 默认走 33。81 帧路径在 `:441-465` 的 latter_frames_audio 处理逻辑中处理 32 帧（去掉第一帧）的多 latent 拼接情况，需进一步确认是否被默认 inference 触发。

4. **`audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1)`**：`flash_head_pipeline.py:224` 取 hidden_states[1:]，即 12 层 transformer hidden（跳过 embedding 层），所以 12 层指 wav2vec2 transformer encoder 12 层（这是 wav2vec2-base 的层数，符合 base 配置）。

## Generator 报告应补充 / 修正的清单

按重要性排序：

1. **【必须修正】wav2vec2 帧率叙述**：删除"wav2vec2 输出 50Hz"，改为"wav2vec2 transformer 12 层 hidden states 经 linear_interpolation 输出 25fps × 768D × 12 层"。维度 A 的"4× 上采样"改为"2× 上采样"（12.5Hz → 25fps）。
2. **【必须修正】frame_num/81 的语境矛盾**：明确说 `infer_params.yaml` 默认 frame_num=33，:430 的 81 是另一种 chunk 配置；论文里说"81 帧 = 1+80"是该函数支持的最大形状但非默认。
3. **【应补充】AudioProjModel 是 flatten+Linear**：在维度 A 节点出"投影路径"时明确不是 cross-attention，是 (5×12×768)→Linear→512→Linear→512→Linear→32×1536。"cross-attention 32 tokens × 1536D" 的措辞会让人误以为 AudioProjModel 内部就是 cross-attention，实际 cross-attention 发生在 DiT block（`flash_head_model.py:230,277`）。
4. **【应澄清】"12 层独立预测" vs flatten+Linear**：维度 A "12 层独立预测"的方案不是简单"加 12 个 head"，而是要让 Talker 输出 12×768=9216 维向量，单步 self-regress 复杂度极高。最小方案应是"Talker 输出 last_hidden_state 1 层 + 接一个 12-layer Up-projector head"。
5. **【可选】782h VividHead 应另查**：标记为 UNVERIFIABLE，等读 `assets/SoulX_FlashHead.pdf` 后再补。
6. **【保留】文档其余结构（维度 B/C、方案 X/Y/Z、三条总评）逻辑成立**，事实层面无新错误。

## 关联笔记

- [[phase2-feasibility-analysis-gan]] — 本核查针对的 Generator 输出
- [[phase2-joint-training]] — 原 Phase 2 文档（被 Generator 指出 4 处错误，本核查全部 CORRECT）
- [[github_learning/soulx-flashhead]] — 与本次源码读取一致
- [[链接文章/公众号/MiniMind-O开源]] — 公众号 ground truth
