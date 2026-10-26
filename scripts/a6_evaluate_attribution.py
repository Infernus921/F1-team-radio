# Stage A.6: attribution accuracy against hand labels (Checkpoint 1)

# Imports
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Settings
N_BOOT = 2000

# Main
cfg = load_config()
rng = np.random.default_rng(cfg["seed"])
labels_dir, res_dir = path(cfg, "labels"), path(cfg, "results") / "stage_a"
gate = cfg["stage_a"]["accuracy_gate"]

items = pd.read_csv(labels_dir / "attribution_items.csv", dtype={"repeat_of": str}).fillna({"repeat_of": ""})
lab = pd.read_csv(labels_dir / "attribution_labels.csv").drop_duplicates("item_id", keep="last")
annotators = sorted(lab["annotator"].unique())
df = items.merge(lab[["item_id", "label"]], on="item_id", how="inner")
turns = pd.read_parquet(path(cfg, "processed") / "turns_attributed.parquet")
df = df.merge(turns, on=["clip_id", "turn_idx"], how="left", suffixes=("", "_t"))

main = df[df["repeat_of"] == ""]
rep = df[df["repeat_of"] != ""]

rr = rep.merge(main[["item_id", "label"]], left_on="repeat_of", right_on="item_id", suffixes=("", "_first"))
retest = {"n": int(len(rr)), "agreement": float((rr["label"] == rr["label_first"]).mean()) if len(rr) else None}

ev = main[main["label"].isin(["DRIVER", "ENGINEER"])].copy()
season_w = pd.read_parquet(path(cfg, "interim") / "clips.parquet").groupby("year").size()
season_w = season_w / season_w.sum()

systems = {
    "ensemble_silk (primary; both-roles constraint)": "ens_silk",
    "ensemble_silk_raw (original naming rule)": "ens_silk_raw",
    "ensemble_k2 (brief's literal k=2)": "ens_k2",
    "voice_only_silk": "voice_silk",
    "voice_only_k2": "voice_k2",
    "text_only": "text_label",
}


# Functions
def acc_with_ci(d, col):
    correct = (d[col].astype(str) == d["label"]).astype(float)
    races = d["race_id"].to_numpy()
    uniq = np.unique(races)
    by_race = {r: correct[races == r].to_numpy() for r in uniq}
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True)
        c = np.concatenate([by_race[r] for r in pick])
        boots.append(c.mean())
    return float(correct.mean()), [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


out = {
    "annotators": annotators,
    "n_labelled_main": int(len(main)),
    "label_rates": main["label"].value_counts(normalize=True).round(3).to_dict(),
    "label_counts": main["label"].value_counts().to_dict(),
    "n_evaluable": int(len(ev)),
    "gate": gate,
    "test_retest": retest,
    "systems": {},
}
for name, col in systems.items():
    a, ci = acc_with_ci(ev, col)
    per_class = {c: float((ev.loc[ev["label"] == c, col].astype(str) == c).mean()) for c in ["DRIVER", "ENGINEER"]}
    per_season = ev.assign(ok=(ev[col].astype(str) == ev["label"])).groupby("year")["ok"].agg(["mean", "size"])
    reweighted = float((per_season["mean"] * season_w.reindex(per_season.index)).sum() /
                       season_w.reindex(per_season.index).sum())
    out["systems"][name] = {
        "accuracy": round(a, 4), "ci95_race_bootstrap": [round(x, 4) for x in ci],
        "season_reweighted_accuracy": round(reweighted, 4),
        "recall": {k: round(v, 4) for k, v in per_class.items()},
        "per_season": {int(y): {"acc": round(r["mean"], 3), "n": int(r["size"])} for y, r in per_season.iterrows()},
        "unattributed_share": float((ev[col].astype(str) == "UNATTRIBUTED").mean()) if col.startswith("ens") else None,
        "passes_gate_point_estimate": a >= gate,
        "ci_entirely_above_gate": ci[0] >= gate,
    }

(res_dir / "attribution_eval.json").write_text(json.dumps(out, indent=1))
df.to_csv(res_dir / "attribution_eval_items.csv", index=False)
print(json.dumps(out, indent=1))
