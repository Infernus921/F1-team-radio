# Claude Code — Build Prompt (Full Pipeline)

> Paste as the opening message. Keep in the repo as `BUILD_BRIEF.md`.

---

## 0. How to work

You are building a DL + NLP research system. **Do not build it all in one pass.** Work to the numbered `🛑 CHECKPOINT` gates. At each one: stop, show what you have, explain in plain language what you did and what you propose next, and **wait for my reply**. Between gates, work autonomously.

At every checkpoint, state explicitly:
- whether the result is **genuinely good**, **weak**, or **a compromise** — do not dress up a null as "an interesting finding"
- whether the work is still **aligned with this brief** or has drifted
- what the number would need to be for the next stage to be worth building

Never fabricate a number, a citation, or a result. If something fails, say it failed. Python 3.11+, fixed seeds, config in one YAML, git commit at each checkpoint, `requirements.txt` maintained.

---

## 1. What we're building

For every Formula 1 team-radio message: identify what the **driver** claimed, estimate how aroused they sounded, and check the claim against the car's telemetry at that moment. Then test whether arousal predicts being wrong, and whether it predicts subsequent lap performance.

| ID | Task |
|---|---|
| T1 | Claim-type classification from driver-side utterances |
| T2 | Telemetry grounding — does the sensor data corroborate the claim? |
| T3 | Arousal estimation from text + audio |
| T4 | Arousal → veridicality, and arousal → subsequent lap performance |

T2 is the centrepiece. T4 is mixed-effects statistics, not deep learning — intentionally.

**Claim taxonomy** (defined backwards from available sensors): `GRIP_DEG`, `TRAFFIC`, `REL_PACE`, `INCIDENT`, `POWER`, `NO_SENSOR`.

**Data:** `MikCil/f1-team-radio` (14,681 clips, 2018–2025, 43 drivers, 149 GPs, audio + ASR transcripts, CC BY 4.0) joined to OpenF1 via `racing_number` + UTC `message_timestamp`.

**Public telemetry has:** speed, throttle %, brake (binary), rpm, gear, DRS at ~3.7 Hz; lap/sector times; gaps; tyre compound and age; pit timing; track position; race control decisions; weather.
**It does not have:** brake pressure, brake or tyre temperatures, steering angle, ride height, fuel load. Claims needing those go to `NO_SENSOR`.

---

## 2. Splits — apply everywhere, no exceptions

- **Group by `race_id`.** All messages from one grand prix land in exactly one fold. Never split at message level; messages from one race share track, weather, compounds and incidents, and a random split inflates every number you will report.
- **Primary:** chronological — train 2018–2023, val 2024, test 2025. Test set is touched once, at the end.
- **Secondary:** 5-fold grouped CV over races, for stable estimates given small per-class counts.
- **Third:** leave-one-driver-out on the top 8 drivers. If performance collapses on unseen drivers, the model learned driver identity, not claim semantics. Report it either way.
- Write a leakage assertion into the test suite that fails if any `race_id` appears in two folds.

---

## Stage A — Speaker attribution

`driver_id` is **whose channel it is**, not who is speaking. Most broadcast clips are the engineer. Many clips contain both speakers in one recording with no turn markers.

1. pyannote diarization over the audio.
2. Cluster speaker embeddings **within each `driver_id`** across the whole corpus — the driver's voice and their engineer's are each consistent across hundreds of clips, so 2-way clustering per channel is tractable.
3. Fallback/ensemble: linguistic-cue classifier (imperatives, "box", target lap times, third-person driver names → engineer).
4. Hand-label 300 utterances; report attribution accuracy.

🛑 **CHECKPOINT 1** — report accuracy. **Below ~85% and we stop and rethink rather than building on sand.** Say plainly which it is.

---

## Stage B — Re-transcription

Existing ASR mangles exactly the vocabulary we need ("SuperSalt" for supersoft, "virtual safeguard" for virtual safety car, driver names garbled).

Whisper large-v3 with an `initial_prompt` seeded with season driver names, compounds and F1 jargon, plus a post-hoc normalization lexicon. Try prompt-only biasing before anything heavier.

Report WER before/after on 100 hand-transcribed clips.

🛑 **CHECKPOINT 2** — WER delta. If the improvement is under ~15% relative, say so; it may not be worth the compute.

---

## Stage C — Annotation

Build a minimal labelling UI. Label 800–1200 driver-side utterances for claim type + arousal (3-point). Two annotators on an overlapping subset; report Cohen's κ per category.

κ below 0.6 on a category means the category definition is broken — say so rather than proceeding.

🛑 **CHECKPOINT 3** — κ and label distribution. Flag any class under ~100 instances; those get dropped.

---

## Stage D — Telemetry join and lap-time residual model

Windows of −60 s to +120 s around each message, resampled to a fixed grid, normalized within driver-session.

Build the expected-lap-time model: fuel-corrected (lap number proxy), tyre-age- and compound-corrected, traffic-corrected. Outcome is the **residual**. This same model serves both `GRIP_DEG` grounding and T4's outcome — report its own error, because everything downstream inherits it.

Quantify the radio timestamp offset on a sample. Timestamps are broadcast times and may lag the utterance; use lap-level windows, not second-level.

🛑 **CHECKPOINT 4** — residual model diagnostics and measured timestamp offset.

---

## Stage E — Models

**T1/T3 text:** DeBERTa-v3-base, shared trunk, claim-type head + arousal head.
**T3 audio:** WavLM or wav2vec2 pooled → arousal head. Normalize within driver-session; control for season (codec differs 2018 vs 2025). Report as **arousal**, never "stress" — vocal arousal here is confounded with g-load, breathing and broadcast compression.
**T2 grounding:** telemetry encoder (1D CNN or small transformer) × claim representation → cross-attention → binary consistency score.

**On fusion:** naive concatenation of text into a time-series backbone is known to underperform constrained fusion (arXiv:2603.22372). Implement a constrained/gated fusion variant and ablate against naive concat. Do not default to concatenation.

**Baselines that must be beaten:** majority class; context-only (lap, position, stint age, gap — no text); text-only; telemetry-only; and an LLM zero-shot prompt for T1.

Metrics: **AUC-PR with bootstrap CIs, per claim type.** Not overall accuracy — classes are imbalanced and test folds are small.

🛑 **CHECKPOINT 5** — full results table vs every baseline. Tell me directly whether we beat context-only. **If we didn't, that is the result and we report it — do not go hunting for a configuration that does.**

---

## Stage F — Controls

All three are mandatory, not optional:

1. **Reverse-time placebo.** Run the same prediction on the window *before* each message. Equal performance backwards means we measured the broadcast editor's selection, not the driver's foresight.
2. **Context-only baseline** as above.
3. **Leakage audit** across race-grouped folds.

🛑 **CHECKPOINT 6** — placebo result. This is the single most important number in the project for credibility.

---

## Stage G — T4

Mixed-effects model: lap-time residual, **and** residual variance over the following 3–5 laps, regressed on lagged arousal.

- **Lag it.** Arousal at lap *L* → residual at *L+1..L+3*, **controlling for the residual at lap L**. Without that control the result is circular: drivers are angry *because* the lap was slow.
- Random slopes on arousal by driver; test heterogeneity via **random-slopes vs random-intercepts likelihood ratio test**. Report the **variance component**, not a per-driver leaderboard. Per-driver slopes to an appendix, labelled underpowered.
- Use predicted arousal **probabilities**, not hard labels — measurement error attenuates coefficients. Re-run on the hand-labelled subset as an attenuation check.
- **Compute and report the minimum detectable effect before looking at results.**
- Note in the write-up: in-race positive affect is rare (cheerful radio is mostly post-race, with no following lap), so the axis is arousal/negative valence vs calm baseline, not happy-to-angry.

🛑 **CHECKPOINT 7** — model, MDE, result. **If null, we report the null with the MDE.** If you find yourself trying specifications after seeing outcomes, stop and tell me — that is p-hacking and I would rather have a clean null.

---

## Stage H — Outputs

- All results to `results/` as JSON + CSV, every number traceable to a script
- Figures to `reports/figures/`
- A `RESULTS.md` that states, for each finding, whether it is strong, weak, or null

🛑 **CHECKPOINT 8** — before any write-up, give me an honest ranking of which findings survive scrutiny and which don't.

---

## Standing rules

- Never report a number you have not computed
- Call T2's target **telemetry-consistency**, never "truth" or "lie detection" — there is no veracity ground truth except FIA verdicts on `INCIDENT`
- Pre-register corroboration thresholds before looking at outcomes; include a sensitivity sweep
- F1 audio is copyrighted despite the dataset's CC BY 4.0 tag — coursework use only, no redistribution
- If I ask for something contradicting this brief, point at the conflict before doing it

**Start at Stage A and stop at Checkpoint 1.**
