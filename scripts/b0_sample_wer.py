# Stage B.0: draw the hand-transcription sample for WER, before any new ASR output exists

# Imports
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Main
cfg = load_config()

# Settings
N = 100
clips = pd.read_parquet(path(cfg, "interim") / "clips.parquet")
pool = clips[clips["duration_s"].between(2, 30)]
years = sorted(pool["year"].unique())
per = [N // len(years) + (i < N % len(years)) for i in range(len(years))]
s = pd.concat([pool[pool["year"] == y].sample(k, random_state=cfg["seed"]) for y, k in zip(years, per)])
s = s.sample(frac=1, random_state=cfg["seed"])
out = pd.DataFrame({"item_id": [f"W{i:03d}" for i in range(len(s))], "clip_id": s["id"].values, "year": s["year"].values})
out.to_csv(path(cfg, "labels") / "wer_items.csv", index=False)
print(out.groupby("year").size().to_dict(), "total audio min:", round(s["duration_s"].sum() / 60, 1))
