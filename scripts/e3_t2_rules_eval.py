# Stage E.3 / F: T2 telemetry-consistency of labelled claims, with controls

# Imports
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--labels-dir", default=None)
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

# Settings
PLACEBO_SHIFT = 3
N_BOOT = 2000
PRIMARY = {"GRIP_DEG": "GRIP_DEG@0.3", "TRAFFIC": "TRAFFIC@1.5", "REL_PACE": "REL_PACE@0.3",
           "INCIDENT": "INCIDENT@L-1..L+2", "POWER": "POWER@8kmh_500rpm"}
cfg = load_config()
rng = np.random.default_rng(cfg["seed"])
labels_dir = Path(args.labels_dir) if args.labels_dir else path(cfg, "labels")
out_dir = Path(args.out_dir) if args.out_dir else path(cfg, "results") / "stage_e"
out_dir.mkdir(parents=True, exist_ok=True)

rules = pd.read_parquet(path(cfg, "processed") / "lap_rules.parquet")
rules["DriverNumber"] = rules["DriverNumber"].astype(str)
rule_cols = [c for c in rules.columns if "@" in c]
items = pd.read_csv(labels_dir / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(labels_dir / "claim_labels.csv", encoding="utf-8", dtype=str)
lab = lab[lab["annotator"] == "A1"].drop_duplicates("item_id", keep="last")
df = items.merge(lab[["item_id", "claim", "not_driver"]], on="item_id")
df = df[df["not_driver"] != "1"]
clips = pd.read_parquet(path(cfg, "interim") / "clips.parquet")[["id", "racing_number"]]
df = df.merge(clips, left_on="clip_id", right_on="id").drop(columns="id")
df["racing_number"] = df["racing_number"].astype(str)

counts = {"labelled_driver_items": int(len(df)), "by_phase": df["phase"].value_counts().to_dict()}
df = df[df["phase"] == "race"].dropna(subset=["lap"]).copy()
df["lap"] = df["lap"].astype(int)
key = ["race_id", "DriverNumber", "LapNumber"]
at_L = df.merge(rules, left_on=["race_id", "racing_number", "lap"], right_on=key, how="left")
df["lap_before"] = df["lap"] - PLACEBO_SHIFT
at_B = df.merge(rules, left_on=["race_id", "racing_number", "lap_before"], right_on=key, how="left")
base = rules.groupby(["race_id", "DriverNumber"])[rule_cols].mean().reset_index()
at_base = df.merge(base, left_on=["race_id", "racing_number"], right_on=["race_id", "DriverNumber"], how="left")
counts["in_race_graded_items"] = int(len(df))
counts["claim_counts_in_race"] = df["claim"].value_counts().to_dict()


# Functions
def boot_diff(a, b, race_a, race_b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, mb = ~np.isnan(a), ~np.isnan(b)
    a, b, race_a, race_b = a[ma], b[mb], np.asarray(race_a)[ma], np.asarray(race_b)[mb]
    if len(a) < 3 or len(b) < 3:
        return None
    races = np.unique(np.concatenate([race_a, race_b]))
    ia = {r: a[race_a == r] for r in races}
    ib = {r: b[race_b == r] for r in races}
    bs = []
    for _ in range(N_BOOT):
        pick = rng.choice(races, len(races), replace=True)
        xa = np.concatenate([ia[r] for r in pick])
        xb = np.concatenate([ib[r] for r in pick])
        if len(xa) and len(xb):
            bs.append(xa.mean() - xb.mean())
    return {"diff": round(float(a.mean() - b.mean()), 4),
            "ci95": [round(float(np.quantile(bs, q)), 4) for q in (0.025, 0.975)]}


out = {"counts": counts, "placebo_shift_laps": PLACEBO_SHIFT, "claims": {}}
rows = []
for claim, prim in PRIMARY.items():
    is_c = (df["claim"] == claim).to_numpy()
    n_c = int(is_c.sum())
    res = {"n_in_race": n_c, "primary_rule": prim, "thresholds": {}}
    if n_c == 0:
        out["claims"][claim] = res
        continue
    for col in [c for c in rule_cols if c.startswith(claim + "@")]:
        cons = at_L.loc[is_c, col].to_numpy(float)
        r = {
            "n_gradeable": int((~np.isnan(cons)).sum()),
            "consistency": round(float(np.nanmean(cons)), 4) if (~np.isnan(cons)).any() else None,
            "base_rate_matched": round(float(np.nanmean(at_base.loc[is_c, col])), 4),
            "other_claim_rate": round(float(np.nanmean(at_L.loc[~is_c, col])), 4),
            "placebo_rate": round(float(np.nanmean(at_B.loc[is_c, col]))
                                  if (~np.isnan(at_B.loc[is_c, col].to_numpy(float))).any() else np.nan, 4),
            "vs_base": boot_diff(cons, at_base.loc[is_c, col], df.loc[is_c, "race_id"], df.loc[is_c, "race_id"]),
            "vs_other_claims": boot_diff(cons, at_L.loc[~is_c, col], df.loc[is_c, "race_id"], df.loc[~is_c, "race_id"]),
            "vs_placebo": boot_diff(cons, at_B.loc[is_c, col], df.loc[is_c, "race_id"], df.loc[is_c, "race_id"]),
        }
        res["thresholds"][col] = r
    out["claims"][claim] = res
    for i in np.flatnonzero(is_c):
        rows.append({"item_id": df.iloc[i]["item_id"], "claim": claim, "race_id": df.iloc[i]["race_id"],
                     "lap": int(df.iloc[i]["lap"]), "consistent_primary": at_L.iloc[i][prim],
                     "placebo_primary": at_B.iloc[i][prim], "base_rate_primary": at_base.iloc[i][prim]})

pd.DataFrame(rows).to_csv(out_dir / "t2_rules_items.csv", index=False)
(out_dir / "t2_rules.json").write_text(json.dumps(out, indent=1, default=str))
for claim, r in out["claims"].items():
    p = r.get("thresholds", {}).get(r["primary_rule"])
    if p:
        print(f"{claim:9s} n={r['n_in_race']:3d}  consistent {p['consistency']}  base {p['base_rate_matched']}  "
              f"other {p['other_claim_rate']}  placebo {p['placebo_rate']}  vs_base {p['vs_base']}  vs_placebo {p['vs_placebo']}")
