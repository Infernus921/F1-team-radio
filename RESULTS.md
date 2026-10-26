# Results

Every number below is produced by a script and stored in `results/` (JSON/CSV). Each finding is
labelled **strong**, **weak** or **null** against the criterion fixed before its data existed
(`results/stage_a/PREREGISTRATION.md`, `results/stage_d/PREREGISTRATION_T2.md`). Deviations from
`BUILD_BRIEF.md` are listed at the end.

## Ranking: what survives scrutiny (Checkpoint 8)

| # | Finding | Verdict | Key number | Source |
|---|---|---|---|---|
| 1 | Claim type is identifiable from driver speech (T1) | **strong** vs context; **weak** vs TF-IDF | DeBERTa 0.426 vs context 0.320 macro AUC-PR (CV), +0.106 [0.039, 0.179]; 2025 test +0.187 [0.073, 0.381]; LODO 0.475 | `results/stage_e/text_metrics.json` |
| 1b | A zero-shot LLM given the category definitions is the best claim classifier | **strong** | Qwen2.5-1.5B-Instruct 0.629 [0.578, 0.691]; vs DeBERTa +0.203 [0.118, 0.271]; vs TF-IDF +0.164 [0.088, 0.230] (paired, same 281 items; one prompt, no tuning) | `results/stage_e/llm_zeroshot.json` |
| 1c | Fine-tuning vs TF-IDF with more data (pre-declared learning curve) | **weak** | DeBERTa 0.272→0.426, TF-IDF 0.390→0.466 (58→225 examples/fold); DeBERTa steeper but flattens; no evidence it overtakes | `results/stage_e/learning_curve.json` |
| 2 | Partial fine-tuning of WavLM recovers arousal (T3) | **strong** | 0.554 [0.495, 0.624] vs frozen 0.442; paired +0.113 [0.028, 0.192]; LODO 0.513 | `results/stage_e/wavlm_ft_metrics.json`, `wavlm_ablation.json` |
| 3 | Arousal is readable from text (T3) | **strong**, with caveat (annotator saw transcript) | DeBERTa multi-task 0.544 [0.485, 0.618] | `results/stage_e/text_metrics.json` |
| 4 | Lap-time residual model is adequate (Stage D) | **strong** | held-out RMSE 0.70 s vs 1.12 s baseline; R² 0.75; fuel slope −0.057 s/lap | `results/stage_d/residual_model_diagnostics.json` |
| 5 | Broadcast timestamps lag speech by ~30–60 s | **strong** (descriptive) | 81.6% of 163 "box" calls stamped after pit entry (median +31.5 s) | `results/stage_d/timestamp_offset.json` |
| 6 | Fusing audio + text beats audio alone (frozen features) | **weak** | +0.056 [0.021, 0.093]; not better than text alone | `results/stage_e/arousal_metrics.json` |
| 7 | Speaker attribution (Stage A, Checkpoint 1) | **null against gate** | 0.670 [0.610, 0.726] vs 0.85; voice 0.646, text 0.391 | `results/stage_a/attribution_eval.json` |
| 8 | Whisper prompt biasing (Stage B, Checkpoint 2) | **weak** | WER 0.290 vs 0.298 dataset (−2.7% rel., CI −3.2% to +7.6%); bar 15% | `results/stage_b/wer.json` |
| 9 | Telemetry corroborates claims (T2 rules) | **null** | no claim type differs from its base rate; INCIDENT 0.11 vs base 0.05 but placebo 0.10 | `results/stage_e/t2_rules.json` |
| 10 | Telemetry identifies the claim (T2 neural) | **null** | gated 0.312 = placebo 0.310; ≈ context 0.291; gated − concat +0.013 [−0.021, 0.053] | `results/stage_e/t2_neural.json` |
| 11 | Arousal → subsequent lap pace (T4, Checkpoint 7) | **null** | +0.147 s [−0.020, 0.315], MDE 0.26 s; hand-label check −0.005 [−0.193, 0.183] | `results/stage_g/t4_results.json`, `t4_mde.json` |
| 12 | Driver heterogeneity in arousal effect | **null** | random-slopes LRT p = 0.19 / 0.55 | `results/stage_g/t4_results.json` |

Placebo (Checkpoint 6): in both T2 analyses the window three laps *before* the message is as
informative as the window at the message. The apparent signal reflects broadcast selection of eventful
moments, not the driver's report.

## Checkpoints

| CP | Stage | Result | Verdict |
|---|---|---|---|
| 1 | Speaker attribution | 0.670 accuracy (n = 330), gate 0.85; test–retest 0.90 (n = 20) | failed; proceeded with human-verified driver turns (logged deviation) |
| 2 | Re-transcription | prompted WER 0.290, plain 0.302, dataset 0.298 (41 clips, 859 words) | below 15% bar; prompts not adopted corpus-wide |
| 3 | Annotation | 450 labelled, 317 driver; NO_CLAIM 123, GRIP_DEG 49, NO_SENSOR 45, REL_PACE 32, INCIDENT 32, TRAFFIC 25, POWER 11 | single annotator, no κ; classes < 30 dropped |
| 4 | Telemetry + residual model | RMSE 0.70 s; timestamp lag 30–60 s | adequate |
| 5 | Models vs baselines | T1 beats context on every split; T3 beats majority/chance; T2 neural ≈ context | mixed |
| 6 | Placebo | placebo ≈ real window in T2 | signal predates the message |
| 7 | T4 | null, MDE 0.26 s/lap | null |

## Deviations from the brief (all logged before the relevant outcome was seen)

1. **Telemetry source:** FastF1 instead of OpenF1, because OpenF1 has no data before 2023.
2. **Attribution gate failed (CP1), project continued:** downstream labelled data uses only utterances the annotator verified as the driver; unlabelled T4 utterances still contain ~30% non-driver speech.
3. **Class threshold 30 instead of 100:** with 317 labels only NO_CLAIM reaches 100. A threshold of 20 (adds TRAFFIC) is reported as sensitivity; the conclusion is unchanged.
4. **Taxonomy:** added `NO_CLAIM` for acknowledgements and questions.
5. **One annotator:** no Cohen's κ; 20-item test–retest for attribution only.
6. **WER set:** 41 clips instead of 100 (the CI on the relative gain already excludes the 15% bar).
7. **Neural T2 target:** the human claim label (predicted from telemetry), not the rule-derived consistency label, which a model could only reproduce.
8. **Leave-one-driver-out** is crossed with race folds so held-out folds are disjoint by both driver and race.
9. **LLM zero-shot baseline:** a local open model (Qwen2.5-1.5B-Instruct) instead of an API model.
10. **Attribution naming constraint** (every channel has ≥ 1 driver and ≥ 1 engineer cluster) was added after a structural check, before any label existed; the original rule is reported alongside it.

## Reproduce

`scripts/` run in order a0–a6, b0–b2, c0, d0–d5, e0–e8, g0, h0–h2 (GPU steps: `scripts/run_all_gpu.ps1`).
Tests: `.venv/Scripts/python -m pytest` (race-leakage assertions).
