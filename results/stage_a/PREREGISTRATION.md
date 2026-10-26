# Stage A — pre-registered attribution method

Written 2026-09-29, before any diarization output was inspected for attribution and before
any hand label existed. Changes after this point are logged at the bottom with a reason.

## Unit
One diarized turn: a contiguous span of one local speaker from pyannote's *exclusive*
diarization, with same-speaker gaps ≤ 0.3 s merged. Turns < 0.5 s are not attributed
(reported as coverage loss).

## Components (each reported separately, then the ensemble)
1. **Voice (per channel).** For each `driver_id` with ≥ 25 clips: take each (clip, local speaker)
   centroid embedding from pyannote; cluster with agglomerative clustering (cosine, average
   linkage), k chosen by silhouette over k = 2..6. Each cluster is named DRIVER or ENGINEER by the
   duration-weighted mean text-cue score of its members (> 0 → ENGINEER). Aggregating over
   hundreds of turns makes a noisy per-turn cue reliable at cluster level.
2. **Voice (global, small channels).** Channels with < 25 clips: logistic regression on embeddings,
   trained on the cluster-derived labels of the large channels.
3. **Text cues.** `f1radio.attribution.text_cue_score` on the turn's Whisper words. The rules were
   fixed in code before any label existed.

## Ensemble rule
Use the voice label, unless the unit's cosine similarity to its cluster centroid is in the bottom
10% for its channel **and** the turn's text score is non-zero. In that case use the sign of the text score.

## Evaluation
- 300 evaluable hand labels (DRIVER/ENGINEER). MIXED / OTHER / NOSPEECH / UNSURE are excluded
  from accuracy and reported as rates. The MIXED rate measures diarization failure.
- Accuracy with a 95% bootstrap CI, resampled by race (turns cluster within races).
  Also per-class recall and accuracy by season.
- Test-retest agreement on 20 repeated items.
- Gate: accuracy ≥ 0.85 (BUILD_BRIEF). No tuning of any component on the hand labels.
  If anything is changed after seeing them, it is logged here and the change is evaluated
  with race-grouped CV over the labels, not reported on the same labels.

## Changes log
- 2026-09-29 03:50, before any attribution output or label existed (clarifications, not changes):
  (a) A turn whose local speaker got no embedding from pyannote uses its text score; if that is 0,
      the turn is UNATTRIBUTED and counted as coverage loss (never silently assigned).
  (b) The brief's literal variant (k = 2 per channel) is reported alongside the silhouette-k variant.
      The silhouette-k ensemble remains the primary number.
- 2026-09-29 ~06:00, still before any label existed: pyannote returns all-zero vectors for some speakers
  (2,208 units). These are treated as "no embedding" under (a). Found via a dry run that crashed on
  them; the dry run used partial Whisper text and was not scored against anything.
- **Change 2 — 2026-09-29 11:04, verified before any hand label existed (labels/attribution_labels.csv
  absent).** Structural check of the full run: 3 channels (LANNOR01 with 1,109 clips, KEVMAG01,
  NYCDEV01) had *no* cluster named DRIVER, and BREHAR01 had no ENGINEER cluster. So every voice turn
  on those channels got one label, which is logically impossible for a two-party radio channel.
  New constraint: each channel with ≥ 2 clusters gets at least one DRIVER and one ENGINEER cluster.
  If the text rule names none, the lowest-score (most driver-like) / highest-score cluster takes that
  name. The constrained variant becomes primary. **The original rule is still reported** as
  `ensemble_silk_raw`, so the effect of this change is visible at Checkpoint 1.
