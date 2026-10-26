# Stage A.1: remove duplicate clips before any split is drawn

# Imports
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Main
cfg = load_config()
interim, results = path(cfg, "interim"), path(cfg, "results") / "stage_a"
results.mkdir(exist_ok=True)

m = pd.read_parquet(interim / "metadata.parquet").merge(pd.read_parquet(interim / "audio_md5.parquet"), on="id")
m = m.sort_values("id").reset_index(drop=True)
m["drop_reason"] = ""

races_per_hash = m.groupby("md5")["race_id"].transform("nunique")
m.loc[races_per_hash > 1, "drop_reason"] = "dup_audio_cross_race"

dup_within = m.duplicated(["md5"], keep="first") & (m["drop_reason"] == "")
m.loc[dup_within, "drop_reason"] = "dup_audio_within_race"


# Functions
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


m["_norm"] = m["transcription"].fillna("").map(norm)
for _, g in m[m["drop_reason"] == ""].groupby(["driver_id", "race_id"]):
    g = g.sort_values("duration_s")
    texts = list(zip(g.index, g["_norm"]))
    for i, (ia, ta) in enumerate(texts):
        if len(ta) < 20:
            continue
        if any(ta in tb and ia != ib for ib, tb in texts[i + 1:]):
            m.loc[ia, "drop_reason"] = "near_dup_contained"

id_ts = pd.to_datetime(m["id"].str.extract(r"_(\d{8}_\d{6})$")[0], format="%Y%m%d_%H%M%S", utc=True)
m["message_ts"] = pd.to_datetime(m["message_timestamp"], utc=True, format="ISO8601")
m["id_ts_offset_s"] = (id_ts - m["message_ts"]).dt.total_seconds()
m["ts_shared"] = m.duplicated(["driver_id", "message_timestamp"], keep=False)

m["keep"] = m["drop_reason"] == ""
clips = m[m["keep"]].drop(columns=["_norm", "drop_reason", "keep"])
clips.to_parquet(interim / "clips.parquet", index=False)

off = m["id_ts_offset_s"]
report = {
    "n_input": int(len(m)),
    "n_kept": int(m["keep"].sum()),
    "dropped_by_reason": m.loc[~m["keep"], "drop_reason"].value_counts().to_dict(),
    "races_kept": int(clips["race_id"].nunique()),
    "clips_per_year_kept": clips.groupby("year").size().to_dict(),
    "ts_shared_rows_kept": int(clips["ts_shared"].sum()),
    "id_ts_offset_s": {
        "n_parsed": int(off.notna().sum()),
        "quantiles": {str(q): round(float(off.quantile(q)), 1) for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]},
        "note": "id time minus message_timestamp; interpretation deferred to Stage D",
    },
}
(results / "dedup_report.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
