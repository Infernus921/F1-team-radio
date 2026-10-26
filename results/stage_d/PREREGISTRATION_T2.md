# T2 telemetry-consistency: pre-registered corroboration rules

Written 2026-09-29 ~06:15, before any claim label existed and before any radio clip was joined to
a consistency outcome. The target is **telemetry-consistency**: whether public telemetry shows
evidence compatible with the claim. It is never "truth" or "lie detection".

## Time reference
Lap-level windows (broadcast timestamps may lag). L = the lap containing `message_timestamp`
(scripts/d2_join_radio_laps.py). Only in-race claims (phase = race) are grounded; pre/post-race
claims are excluded from T2 and counted.

## Rules (primary threshold; sensitivity sweep in brackets)

| Claim | Consistent if… | Primary | Sweep |
|---|---|---|---|
| `GRIP_DEG` | mean lap-time residual (d1 model, which already corrects for typical tyre-age deg) over clean laps L−2..L is ≥ τ | τ = +0.30 s | 0.10, 0.20, 0.30, 0.50, 0.80 |
| `TRAFFIC` | gap to car ahead at the end of lap L−1 or L is < g, **or** a blue-flag race-control message names the car in L−1..L+1 | g = 1.5 s | 1.0, 1.5, 2.0, 3.0 |
| `REL_PACE` | a pace differential exists: the gap to the nearest car (ahead or behind) changes by ≥ p s/lap on average over L−2..L | p = 0.30 | 0.15, 0.30, 0.50 |
| `INCIDENT` | a race-control message names the car (number) within laps L−1..L+2 | window L−1..L+2 | L..L+1, L−2..L+3 |
| `POWER` | on lap L or L+1, the car's max speed is ≥ v km/h below its median max speed over its previous 5 clean laps, **or** max RPM is ≥ r below the same baseline | v = 8 km/h, r = 500 | v 4/8/15; r 300/500/1000 |
| `NO_SENSOR`, `NO_CLAIM` | not grounded | — | — |

A claim with missing inputs (no car data, not enough prior laps) is **ungradeable** and counted, never imputed.

## Known limitations (stated in advance)
- `REL_PACE` has no direction ("he's faster" vs "I'm faster"); the rule only checks that *some* pace
  differential exists. It is weak by construction.
- Base rates differ a lot (cars are often within 1.5 s), so every rule is reported next to its base
  rate on matched non-claim laps and next to the reverse-time placebo (Stage F).
- **Circularity:** the T2 label is rule-derived from telemetry, so a neural T2 model can at best
  reproduce the rules. The only external ground truth is FIA outcomes for `INCIDENT`
  (race-control decision messages: penalty / no further action / noted). This goes to Checkpoint 4 for discussion.

## Changes log
- (none yet)
