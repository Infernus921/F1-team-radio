# Stage D.2: join each radio clip to the car's lap at message_timestamp

# Imports
import json
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

rows = []
for race_id, g in clips.groupby("race_id"):
    f = tel / race_id / "laps.parquet"
    if not f.exists():
        rows += [dict(id=i, phase="no_laps") for i in g["id"]]
        continue
    L = pd.read_parquet(f, columns=["DriverNumber", "LapNumber", "LapStartDate", "LapTime", "Time"])
    L = L.dropna(subset=["LapStartDate"])
    L["DriverNumber"] = L["DriverNumber"].astype(str)
    L["end"] = L["LapStartDate"] + L["LapTime"].fillna(pd.Timedelta(0))
    by_car = {k: v.sort_values("LapNumber") for k, v in L.groupby("DriverNumber")}
    for r in g.itertuples():
        car = by_car.get(str(r.racing_number))
        if car is None or car.empty:
            rows.append(dict(id=r.id, phase="no_laps"))
            continue
        t = r.ts
        if t < car["LapStartDate"].iloc[0]:
            rows.append(dict(id=r.id, phase="pre_race",
                             secs_to_start=(car["LapStartDate"].iloc[0] - t).total_seconds()))
            continue
        if t >= car["end"].iloc[-1]:
            rows.append(dict(id=r.id, phase="post_race", last_lap=int(car["LapNumber"].iloc[-1]),
                             secs_after_end=(t - car["end"].iloc[-1]).total_seconds()))
            continue
        hit = car[(car["LapStartDate"] <= t) & (t < car["end"])]
        if hit.empty:
            prev = car[car["LapStartDate"] <= t].iloc[-1]
            rows.append(dict(id=r.id, phase="race", lap=int(prev["LapNumber"]), lap_frac=np.nan))
            continue
        h = hit.iloc[0]
        dur = (h["end"] - h["LapStartDate"]).total_seconds()
        rows.append(dict(id=r.id, phase="race", lap=int(h["LapNumber"]),
                         lap_frac=(t - h["LapStartDate"]).total_seconds() / dur if dur > 0 else np.nan))

# Settings
J = clips[["id", "race_id", "driver_id", "racing_number", "year", "ts"]].merge(pd.DataFrame(rows), on="id")
J.to_parquet(path(cfg, "processed") / "clip_laps.parquet", index=False)
out_dir = path(cfg, "results") / "stage_d"
out_dir.mkdir(exist_ok=True)
with_laps = J[J["phase"] != "no_laps"]
summary = {
    "n_clips": int(len(J)),
    "phase_counts": J["phase"].value_counts().to_dict(),
    "phase_share_where_laps_exist": with_laps["phase"].value_counts(normalize=True).round(3).to_dict(),
    "pre_race_minutes_before_start_quantiles": (with_laps["secs_to_start"].dropna() / 60).quantile([.1, .5, .9]).round(1).to_dict(),
    "post_race_minutes_after_end_quantiles": (with_laps["secs_after_end"].dropna() / 60).quantile([.1, .5, .9]).round(1).to_dict(),
    "race_phase_by_year": J[J["phase"] == "race"].groupby("year").size().to_dict(),
}
(out_dir / "join_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
