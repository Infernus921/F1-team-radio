# Stage C annotation guide: claim type + arousal (driver-side utterances)

You label **what the driver asserts** and **how aroused they sound**. Listen to the audio; the
transcript is only an aid (it contains ASR errors).

## 1. Claim type (pick one: the main claim)

Categories are defined by what public telemetry can check (BUILD_BRIEF §1).

| Label | The driver asserts something about… | Examples | Checked against |
|---|---|---|---|
| `GRIP_DEG` | tyre grip, degradation, tyres "gone", sliding, understeer/oversteer from tyres | "No grip at the rear." "These tyres are dead." | lap-time residual vs tyre age/compound |
| `TRAFFIC` | cars around them: blocked, stuck behind, dirty air, blue flags, being held up | "I'm stuck behind him." "Blue flag!" | gaps, positions, intervals |
| `REL_PACE` | own pace *relative to* others, or others' pace | "He's much faster." "I can go quicker than this." | lap/sector times vs rivals |
| `INCIDENT` | contact, being pushed off, a rival's illegal move, track limits, unsafe release | "He pushed me off!" "That's a penalty." | race-control messages / FIA decisions |
| `POWER` | engine/power-unit: loss of power, harvesting, gearbox, RPM, DRS | "I've lost power." "No DRS." | speed, RPM, throttle, gear, DRS |
| `NO_SENSOR` | a factual claim public telemetry can't check: brakes temp, steering, balance setup, fuel, drinks, seat, radio, visor, weather feel | "Brakes are overheating." "My drink isn't working." | — |
| `NO_CLAIM` | **no assertion**: acknowledgements, questions, thanks, swearing without content, requests | "Copy." "What's the gap?" "Box this lap?" "Thanks guys." | — |

Rules
- **One label per utterance**: the claim that carries the message. If two claims are equally central, pick the one earlier in the utterance and tick *multi-claim*.
- A **question is not a claim** ("Are my tyres OK?" → `NO_CLAIM`). A complaint phrased as a question that asserts something ("Why is he so fast?") → `REL_PACE`.
- If the utterance is actually the **engineer** (attribution error), tick *not driver* and skip the rest. This is recorded, not wasted: it feeds back into the attribution error rate.
- **Both driver and engineer in one item:** label only the driver's part (claim from what the driver asserts,
  arousal from the driver's voice; a bare "copy"/"yeah" is `NO_CLAIM`). If the driver's part is too short or
  unclear to judge, tick *not driver*. Don't use *multi-claim* for this; it means two claims by the driver.

`NO_CLAIM` was added to the brief's taxonomy (pending your approval): without it, acknowledgements
would be forced into a sensor category. It is excluded from the T1/T2 targets.

## 2. Arousal (3-point, from the voice)

Judge **the voice**, not the words, relative to **this driver's usual radio voice**.

| Label | Sounds like |
|---|---|
| `1 LOW` | calm, flat, conversational, measured |
| `2 MID` | raised, urgent, clipped, noticeably energetic |
| `3 HIGH` | shouting, strained, very fast or breaking voice, audible anger or elation |

- Engine noise and radio compression make everything sound louder; don't count loudness alone.
- This is **arousal**, not "stress": effort, g-force and breathing also raise it.

## 3. Mechanics
- Keys: claim `1`–`7` in table order, arousal `q` `w` `e`, `n` = not driver, `Space` = replay, `Enter` = save and next.
- Annotator 2 labels the overlap set without seeing annotator 1's labels.
