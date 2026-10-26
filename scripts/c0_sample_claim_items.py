# Stage C.0: draw driver-side utterances for claim-type + arousal labelling

# Imports
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path
from f1radio.utterances import driver_utterances

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--overlap", type=int, default=100)
ap.add_argument("--race-share", type=float, default=0.8)
ap.add_argument("--out", default=None, help="override output path (plumbing tests only)")
args = ap.parse_args()

cfg = load_config()
rng = np.random.default_rng(cfg["seed"] + 3)

# Settings
T = pd.read_parquet(path(cfg, "processed") / "turns_attributed.parquet")
J = pd.read_parquet(path(cfg, "processed") / "clip_laps.parquet")[["id", "race_id", "year", "phase", "lap"]]

D = driver_utterances(T)
D = D[D["text"].str.len() > 0]
D = D.merge(J, left_on="clip_id", right_on="id", how="left").drop(columns="id")
D["phase"] = D["phase"].fillna("no_laps")
print("driver utterances:", len(D), "| by phase:", D["phase"].value_counts().to_dict())

years = sorted(D["year"].dropna().unique())
per = [args.n // len(years) + (i < args.n % len(years)) for i in range(len(years))]
picks = []
for y, k in zip(years, per):
    g = D[D["year"] == y]
    race, other = g[g["phase"] == "race"], g[g["phase"] != "race"]
    k_race = min(len(race), int(round(k * args.race_share)))
    k_other = min(len(other), k - k_race)
    k_race = min(len(race), k - k_other)
    picks.append(race.sample(k_race, random_state=int(rng.integers(1 << 31))))
    picks.append(other.sample(k_other, random_state=int(rng.integers(1 << 31))))
S = pd.concat(picks).sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
S["item_id"] = [f"C{i:04d}" for i in range(len(S))]
S["overlap_set"] = 0
per_ov = [args.overlap // len(years) + (i < args.overlap % len(years)) for i in range(len(years))]
for y, k in zip(years, per_ov):
    idx = S.index[S["year"] == y][:k]
    S.loc[idx, "overlap_set"] = 1
cols = ["item_id", "utt_id", "clip_id", "start", "end", "text", "year", "race_id", "driver_id", "phase", "lap", "overlap_set"]
S[cols].to_csv(args.out or (path(cfg, "labels") / "claim_items.csv"), index=False, encoding="utf-8")
print(len(S), "items | per season:", S.groupby("year").size().to_dict(),
      "| race phase share:", round((S["phase"] == "race").mean(), 2), "| overlap:", int(S["overlap_set"].sum()))
