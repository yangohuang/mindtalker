# Research Log — MindTalker Phase 2

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-05-08 | bootstrap | Initialized autoresearch workspace at `/path/to/yg/code/mindtalker/research/`. Imported Phase 2 plan from vault: `phase2-final-plan.md` (final), `phase2-feasibility-analysis-gan.md` (GAN harness Round 2), `phase2-feasibility-evaluator-report.md` (14-claim fact-check). Defined 5 hypotheses (H0-H4): H0 Bridge MLP baseline, H1 A1 end-to-end, H2 A1 vs Bridge MLP, H3 Hidden Bridge layer ablation, H4 teacher-forcing schedule. Proxy metric: SyncNet (LSE-D / LSE-C). Compute: single RTX 4090. Known baseline (Phase 1 naive) to be measured. |
| 2 | 2026-05-08 | inner-loop | H0 run_001 (design-verification): Implemented `src/bridge_mlp.py` with two variants. Smoke test on `flashhead` conda env (torch 2.7.1+cu128). BridgeMLPLight 7.09M params ✓, BridgeMLPHeavy 92.29M params, both forward (2,33,12,768) → same shape. Decision: use Light as default, Heavy as fallback if underfits. Reuse `flashhead` env (skip mindtalker env creation, all deps present). |
| 3 | 2026-05-08 | inner-loop | H0 run_002 (stage-spike): Implemented `src/phase1_pipeline.py` (wav2vec2 stage). Switched torchaudio→librosa+soundfile (flashhead env). GPU OOM (24GB occupied by 4 prior processes — minicpm 14GB, flashhead 5.7GB, digithuman 1.5GB, TEI router 1.5GB). Ran on CPU. Result: 65.92s `podcast_sichuan_16k.wav` → (1648, 12, 768) features, 25fps alignment perfect. Mean -0.0032 std 0.3700 normal. FlashHead-input format confirmed. Wrote `src/minicpm_o_stage.py` skeleton for Stage 1 (deferred until GPU free). |
| 4 | 2026-05-08 | design-revision | Updated H0 protocol: 1 baseline → **4-tier baseline matrix**. B0 (real audio upper bound) is prerequisite for evaluating H0, not just B1 (MiniCPM-o-derived Phase 1). Δ(B0-B1) defines the actual gap Bridge MLP must close. Found via reflection: "if B0 ≈ B1 the experiment is dead before it starts." |

<!-- Entry types:
  bootstrap, inner-loop, outer-loop, pivot, report, conclude
-->
