# Validation Set Plan — 100 段 held-out

## 目标

为 H0/H1/H2 评测准备 100 段 (audio, reference_image, ground_truth_video) 验证三元组。
SyncNet 测分需要 (生成视频, 原音频) 对，所以最低要 (audio, reference_image) 100 对。

## 数据源选择

### 选项 A · AISHELL-3（推荐）

- **链接**：https://www.openslr.org/93/
- **规模**：85h，218 说话人，~88k clips
- **语言**：普通话
- **采样**：16kHz 单声道 — **直接喂 wav2vec2 不用重采样**
- **许可**：Apache 2.0
- **下载**：~12GB tar.gz

抽取协议：
1. 取前 100 个 clip，去掉 < 1s 和 > 10s 的（保留 100 段 1-10s 范围）
2. 每段配一张固定 reference image（用 FlashHead `examples/girl.png`）
3. 输出 `data/val_set/clip_{000-099}.wav`

### 选项 B · WenetSpeech 子集

更大但中文更多样，需要 sample 一遍。10000h 全量太大，子集 ~50GB。

### 选项 C · 现有本地音频（spike-only）

| 文件 | 时长 | 用途 |
|---|---|---|
| `SoulX-FlashHead/examples/podcast_sichuan_16k.wav` | 65.92s | spike#1 已用 |
| `SoulX-FlashTalk/examples/cantonese_16k.wav` | TBD | spike#2 |
| `PantoMatrix/examples/audio/2_scott_0_103_103_28s.wav` | 28s | spike#3 |
| `Wan2.2/examples/zero_shot_prompt.wav` | TBD | spike#4 |
| `Wan2.2/examples/talk.wav` | TBD | spike#5 |

→ 5 段，**够 spike 不够 baseline 测量**。

## Bridge MLP 训练数据生成（独立于验证集）

需要 5000-10000 段配对 (src_feat, tgt_feat)：
- src: MiniCPM-o-4.5-awq 复述句子 → 24k audio → resample 16k → wav2vec2 → src_feat
- tgt: 真人录音 → wav2vec2 → tgt_feat
- 句子来源：AISHELL-3 自带的文本 transcripts

伪代码：
```
for sentence_text, real_audio_16k in aishell3_subset:
    minicpm_audio_24k = minicpmo.tts(sentence_text)         # GPU stage
    minicpm_audio_16k = librosa.resample(minicpm_audio_24k, 24000, 16000)
    src_feat = wav2vec2(minicpm_audio_16k)                   # (T, 12, 768)
    tgt_feat = wav2vec2(real_audio_16k)                      # (T, 12, 768)
    save_pair(src_feat, tgt_feat)
```

GPU 阻塞期可暂缓 src 生成（MiniCPM-o 12GB），先准备 tgt（real audio → wav2vec2）。
CPU 跑 wav2vec2 → tgt 生成 ~实时（已验证：65.92s audio CPU 处理几秒）。

## 立即可做（不依赖 GPU/网络）

1. ✅ Plan 已写
2. ⏳ 待 GPU 空 → 下载 AISHELL-3（12GB，预计 1-2h，取决于网络速度）
3. ⏳ 待 GPU 空 → 跑 MiniCPM-o-4.5 复述生成 src audio
4. ✅ 现可做：CPU 处理 wav2vec2 抽 tgt features（一次性 5-8h CPU）

## 风险

- **AISHELL-3 下载速度受限**：用 hf-mirror.com 镜像或 modelscope
- **MiniCPM-o 复述质量受 prompt 影响大**：需要 prompt engineering 让其严格按 transcript 念
- **真人音频 vs MiniCPM-o 音频长度差异**：同一句子 MiniCPM-o 可能更慢/更快，需要 DTW 对齐或 trim 到等长
