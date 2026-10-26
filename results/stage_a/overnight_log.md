# Overnight log (2026-09-29)

Plain-language status of what ran overnight. Newest entries at the bottom.
(Times before ~10:00 are approximate; the machine clock read 11:04 when the attribution step finished.)

## Morning summary (11:10)
**All overnight compute is done.** Nothing has been evaluated against labels; no checkpoint has been passed.
- ✅ Diarization: all 14,370 clips (fixed a 30× slowdown along the way; output identical)
- ✅ Whisper large-v3 with word timestamps: all 14,370 clips
- ✅ Driver/Engineer guess for every turn (29,134 turns ≥ 0.5 s; 34% Driver, 61% Engineer, 5% unattributed). **Unscored until your labels.**
- ✅ Telemetry: 148/149 races; lap-time model held-out error 0.70 s/lap
- ✅ Radio→lap join: 10,229 clips matched to a racing lap
- ✅ Your labelling page is ready and tested → **double-click `START_LABELLING.bat`**
- ⚠️ Two things to know before you label: (1) diarization split only 14% of clips into ≥ 2 speakers, which may be too few, and your "Mixed" labels will tell us; (2) I added a logged, pre-label constraint so every channel has at least one Driver voice cluster (Lando Norris's channel had none).
- ❓ Four decisions waiting for you (see "When you wake up" at the bottom).

## Rules I'm following tonight
- No checkpoint gets passed and no result gets evaluated without you.
- Only "no-regret" work: diarization, Whisper, per-turn Driver/Engineer guesses (unscored), labelling pages, telemetry download.

## Log
- 02:45 Started. Diarization currently 2.3x realtime (~19 h for everything); testing faster settings first.
- 03:20 Found why diarization is slow: pyannote peaks at 10.9 GB GPU memory on this 8 GB card, so Windows spills to system RAM. Bigger window steps are faster but change the output by 11–15% DER, so rejected. Testing smaller batch sizes (should give identical output).
- 03:20 Telemetry download running via FastF1 (~41 s/race, ~1.7 h total). **Decision for you:** FastF1 for all seasons instead of OpenF1, because OpenF1 has nothing before 2023 (checked).
- 03:20 Built the Stage B hand-transcription page and drew its 100-clip sample (15.8 min of audio, 14–15 clips per season).
- 03:20 Wrote results/stage_a/PREREGISTRATION.md: attribution method fixed before any labels exist.
- ~03:00 **Diarization fixed: 65x faster, identical output.** Batch size 4 instead of 32 keeps pyannote inside the 8 GB card (0.25 GB vs 10.9 GB peak; DER vs default = 0.0000 on 40 clips). The full run was launched at 03:04 (log: data/interim/diarize.log, ~1 h).
- ~03:05 **Telemetry:** the first attempt hit FastF1's 500-requests/hour limit. The fetcher now waits and retries (log: data/interim/fetch_telemetry.log, ~3–4 h). 2018 Italian GP has no timing data in the source at all, so it gets no telemetry. 2018 Bahrain has laps but no car data.
- ~03:10 **Whisper large-v3 (plain, word timestamps)** launched on all clips (log: data/interim/asr_plain.log). Shares the GPU with diarization; expect it to finish mid/late morning.
- ~03:30 Written, not yet run: scripts/a4_attribute.py (Driver/Engineer per turn), a5_sample_label_items.py (your 300-turn sample), a6_evaluate_attribution.py (Checkpoint 1 numbers).
- ~03:40 Stage C prepared: labels/ANNOTATION_GUIDE.md and the claim/arousal labelling page (scripts/claim_server.py). **Decision for you:** I added a `NO_CLAIM` category for "Copy"/"Thanks"/questions. The brief's taxonomy has no slot for them.
- ~03:45 The auto-mode permission checker started failing on shell commands (a service outage, not a refusal). If this log stops here, that's why: the jobs above keep running in the background regardless.

- ~04:00 Whisper was crawling at ~0.5x realtime while sharing the GPU with diarization, so I stopped it (79 clips kept). It restarts **automatically** when diarization finishes. The 300-turn label sample is also queued to draw itself automatically at that point.
- ~04:10 **Timestamp mystery solved:** `message_timestamp` is UTC; the time inside each clip ID is *local* track time (Abu Dhabi 17:27 local = 13:27 UTC), hence the whole-hour offsets. `message_timestamp` is the one to use.
- ~04:15 **Radio→lap join works** (scripts/d2_join_radio_laps.py). On the 47 races downloaded so far: 74% of clips fall during the race, 14% after the flag (median 3.5 min after the last lap), 11% before the start (median 17.5 min before). Implication for T4: about a quarter of clips have no following racing lap.
- ~04:15 Also written, not yet run: scripts/d1_lap_residuals.py (expected-lap-time model), b1_asr_prompted.py and b2_wer.py (Stage B).

- ~04:15 First test of the lap-time model (48 races): held-out error 0.77 s/lap (median race) vs 1.18 s for a driver-average baseline; fuel slope −0.059 s/lap (right sign and size). Full numbers come at Checkpoint 4.
- ~05:05 **Diarization finished: all 14,370 clips.** Speakers found per clip: 0 → 129, 1 → 12,396, 2 → 1,686, 3 → 159. Overlap is 2.5% of speech. 77 of the 129 "no speech" clips have the transcript "Thank you.", which confirms that the ASR invents that phrase on silence.
- ~05:05 ⚠️ **Honest concern:** 86% of clips came out single-speaker, but reading transcripts I estimated ~25% mix driver and engineer. Diarization may be under-splitting (radio compression makes the two voices similar). **Your labels will measure this**: the "Mixed" button exists for exactly this, and the Mixed rate is reported at Checkpoint 1.
- ~05:10 **Your labelling sample is ready:** 380 items (360 + 20 repeats), ~51 per season, 37 channels, 135 races. I tested the page in a browser (audio plays; nothing was written).
- ~05:10 Queued to run automatically: attribution (a4) when Whisper finishes; re-run of the lap join and residual model when the telemetry download finishes.

- ~05:40 **Telemetry download complete:** 148/149 races have lap data (147 with car data). Missing: 2018 Italian GP (nothing in the source); 2018 Bahrain has laps only.
- ~05:45 **Lap join on the full data:** 10,229 clips fall during a race with a matched lap (2025 contributes only 531). About 28% of clips are pre- or post-race.
- ~05:45 **Lap-time model on the full data** (147 races, 140k clean laps): held-out error **0.70 s/lap** (median race; IQR 0.59–0.88) vs 1.12 s for a driver-average baseline; R² 0.75; fuel slope −0.057 s/lap. (Checkpoint 4 material, reported properly then.)
- ~05:50 Whisper plain run: ~38 clips/min, **ETA ~11:30**. Then automatically: attribution (a4), then the prompted-Whisper run on the 100 WER clips.
- ~05:50 Driver glossary for the transcription page built from official race results (44 drivers).

- ~06:00 Dry-ran attribution on partial data to catch bugs before the automatic run: pyannote emits all-zero voice fingerprints for 2,208 speakers, which crashed clustering. Now treated as "no fingerprint" → text fallback (logged in the pre-registration). Nothing was scored.
- ~06:10 Stage C sampling script written and dry-run (output deleted so it can't be used by mistake). It draws 800 driver utterances, ~80% in-race, equal per season, with a 100-item overlap set for a second annotator.
- ~06:15 **Pre-registered the T2 corroboration rules** (results/stage_d/PREREGISTRATION_T2.md) before any claim label exists, as the brief requires. **For discussion at Checkpoint 4:** T2's target comes from these telemetry rules, so a neural T2 model can at best reproduce them. The only external ground truth is FIA decisions on `INCIDENT`.

- ~06:30 Whisper sped up to ~75 clips/min once the CPU was free. **New ETA ~08:30.** Then attribution runs automatically.
- ~06:30 Prepared (not run): shared driver-utterance definition (src/f1radio/utterances.py) and the WavLM/prosody audio feature extractor for T3 (scripts/e0_audio_features.py). The remaining Stage E design waits for Checkpoint 3 label counts.

## ☀️ When you wake up
1. **Double-click `START_LABELLING.bat`** in the project folder. A browser page opens (refresh it if it's blank at first).
2. Label until the counter says **300 driver/engineer labels** (roughly 1–1.5 h; take breaks, it autosaves).
   Keys: `1` Driver, `2` Engineer, `3` Mixed, `4` Other voice, `5` No speech, `6` Unsure, `Space` replay, `C` whole clip.
   Judge by the voice. The driver usually has engine/wind noise behind them.
3. Tell me when you're done (or tell me "continue" if my session has stopped), and I'll compute Checkpoint 1.
4. Also tell me your answers to the four decisions (FastF1, NO_CLAIM, Stage C size + second annotator, WER set size).
