# Stage A.5: draw the hand-labelling sample for attribution accuracy

# Imports
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Main
N_MAIN, N_REPEAT = 360, 20
cfg = load_config()
rng = np.random.default_rng(cfg["seed"])
interim = path(cfg, "interim")
min_s = cfg["stage_a"]["min_segment_s"]

clips = pd.read_parquet(interim / "clips.parquet").set_index("id")
rows = []
with open(interim / "diarization.jsonl") as f:
    for line in f:
        r = json.loads(line)
        for k, (s, e, spk) in enumerate(r["turns"]):
            if e - s >= min_s:
                rows.append((r["id"], k, s, e))
turns = pd.DataFrame(rows, columns=["clip_id", "turn_idx", "start", "end"])
turns = turns[turns["clip_id"].isin(clips.index)]
turns["year"] = clips.loc[turns["clip_id"], "year"].values
turns["race_id"] = clips.loc[turns["clip_id"], "race_id"].values
turns["driver_id"] = clips.loc[turns["clip_id"], "driver_id"].values
print("eligible turns:", len(turns), turns.groupby("year").size().to_dict())

years = sorted(turns["year"].unique())
per = [N_MAIN // len(years) + (i < N_MAIN % len(years)) for i in range(len(years))]
main = pd.concat([turns[turns["year"] == y].sample(k, random_state=int(rng.integers(1 << 31)))
                  for y, k in zip(years, per)])
main = main.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
main["item_id"] = [f"A{i:03d}" for i in range(len(main))]
main["repeat_of"] = ""

src = main.iloc[rng.choice(150, N_REPEAT, replace=False)].copy()
src["repeat_of"] = src["item_id"]
src["item_id"] = [f"R{i:03d}" for i in range(N_REPEAT)]
items = main.copy()
for _, r in src.iterrows():
    pos = int(rng.integers(200, len(items) + 1))
    items = pd.concat([items.iloc[:pos], r.to_frame().T, items.iloc[pos:]], ignore_index=True)

cols = ["item_id", "clip_id", "start", "end", "year", "race_id", "driver_id", "turn_idx", "repeat_of"]
items[cols].to_csv(path(cfg, "labels") / "attribution_items.csv", index=False)
print(f"{len(items)} items ({N_MAIN} main + {N_REPEAT} repeats); per season:",
      main.groupby("year").size().to_dict(), "| distinct channels:", main["driver_id"].nunique(),
      "| distinct races:", main["race_id"].nunique())
