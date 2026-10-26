# Stage D.4: how far do radio message_timestamps lag the actual speech? Two anchors

# Imports
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Main
cfg = load_config()
tel = path(cfg, "raw") / "telemetry"
clips = pd.read_parquet(path(cfg, "interim") / "clips.parquet")
clips["ts"] = pd.to_datetime(clips["message_timestamp"], utc=True, format="ISO8601").dt.tz_localize(None)
asr = {r["id"]: r["text"] for r in map(json.loads, open(path(cfg, "interim") / "asr_plain_words.jsonl", encoding="utf-8"))}
clips["text"] = clips["id"].map(asr).fillna("")

# Settings
J = pd.read_parquet(path(cfg, "processed") / "clip_laps.parquet")[["id", "phase"]]
clips = clips.merge(J, on="id")
BOX = re.compile(r"\bbox,? box\b|\bbox (this|now|the end of this)\b|\bbox,? box,? box\b", re.I)

a_rows, b_rows = [], []
for race_id, g in clips.groupby("race_id"):
    f = tel / race_id / "laps.parquet"
    if not f.exists():
        continue
    L = pd.read_parquet(f, columns=["DriverNumber", "LapNumber", "LapStartDate", "LapTime", "PitInTime", "Time"])
    L["DriverNumber"] = L["DriverNumber"].astype(str)
    meta = json.loads((tel / race_id / "meta.json").read_text())
    t0 = pd.Timestamp(meta["t0_date"]) if meta.get("t0_date") not in (None, "NaT", "None") else None
    for r in g.itertuples():
        car = L[L["DriverNumber"] == str(r.racing_number)]
        if car.empty:
            continue
        if BOX.search(r.text) and t0 is not None:
            pit = (t0 + car["PitInTime"].dropna()).to_numpy()
            if len(pit):
                d = (pit - np.datetime64(r.ts)) / np.timedelta64(1, "s")
                k = np.argmin(np.abs(d))
                if abs(d[k]) < 300:
                    a_rows.append({"race_id": race_id, "msg_minus_pitin_s": -float(d[k])})
        if r.phase == "post_race":
            end = (car["LapStartDate"] + car["LapTime"]).dropna().max()
            b_rows.append({"race_id": race_id, "car": r.racing_number, "after_line_s": (r.ts - end).total_seconds()})

A = pd.DataFrame(a_rows)
B = pd.DataFrame(b_rows)
Bfirst = B.groupby(["race_id", "car"])["after_line_s"].min() if len(B) else pd.Series(dtype=float)
q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
out = {
    "A_box_calls": {"n": int(len(A)),
                    "msg_minus_pitin_s_quantiles": {str(k): round(float(v), 1) for k, v in A["msg_minus_pitin_s"].quantile(q).items()} if len(A) else None,
                    "share_after_pit_entry": round(float((A["msg_minus_pitin_s"] > 0).mean()), 3) if len(A) else None,
                    "reading": "negative = call precedes pit entry (expected). Positive values imply timestamp lag (or a call made in the pit lane)."},
    "B_first_post_race_clip": {"n_cars": int(len(Bfirst)),
                               "seconds_after_line_quantiles": {str(k): round(float(v), 1) for k, v in Bfirst.quantile(q).items()} if len(Bfirst) else None,
                               "reading": "lower tail approximates the minimum broadcast lag after the flag"},
}
(path(cfg, "results") / "stage_d" / "timestamp_offset.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
