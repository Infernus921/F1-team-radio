# Stage A.4: attribute every diarized turn to DRIVER or ENGINEER

# Imports
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.attribution import text_cue_score, words_in_span
from f1radio.config import load_config, path, seed_everything

MIN_CLIPS_PER_CHANNEL = 25
K_RANGE = range(2, 7)
LOW_SIM_QUANTILE = 0.10

cfg = load_config()
seed_everything(cfg["seed"])
interim, processed = path(cfg, "interim"), path(cfg, "processed")
res_dir = path(cfg, "results") / "stage_a"
min_s = cfg["stage_a"]["min_segment_s"]

clips = pd.read_parquet(interim / "clips.parquet").set_index("id")
words = {}
with open(interim / "asr_plain_words.jsonl") as f:
    for line in f:
        r = json.loads(line)
        words[r["id"]] = r["words"]

# Turns and units
turn_rows, unit_rows, unit_emb = [], [], []
with open(interim / "diarization.jsonl") as f:
    for line in f:
        r = json.loads(line)
        cid = r["id"]
        if cid not in clips.index:
            continue
        drv = clips.at[cid, "driver_id"]
        w = words.get(cid, [])
        for k, (s, e, spk) in enumerate(r["turns"]):
            text = words_in_span(w, s, e)
            turn_rows.append(dict(clip_id=cid, turn_idx=k, start=s, end=e, spk=spk, driver_id=drv,
                                  dur=e - s, text=text, text_score=text_cue_score(text, drv)))
        for spk, emb in r["emb"].items():
            unit_rows.append(dict(clip_id=cid, spk=spk, driver_id=drv))
            unit_emb.append(emb)

turns = pd.DataFrame(turn_rows)
units = pd.DataFrame(unit_rows)
E = np.asarray(unit_emb, dtype=np.float32)
nonzero = np.linalg.norm(E, axis=1) > 1e-6
n_zero_emb = int((~nonzero).sum())
units, E = units[nonzero].reset_index(drop=True), E[nonzero]
E /= np.linalg.norm(E, axis=1, keepdims=True)

t_ok = turns[turns["dur"] >= min_s]
agg = t_ok.assign(wscore=t_ok["text_score"] * t_ok["dur"]).groupby(["clip_id", "spk"]).agg(
    dur=("dur", "sum"), wscore=("wscore", "sum"))
units = units.join(agg, on=["clip_id", "spk"])
units[["dur", "wscore"]] = units[["dur", "wscore"]].fillna(0.0)


def cluster_channel(idx, k_fixed=None):
    X = E[idx]
    if k_fixed:
        k_best = k_fixed
    else:
        best = (-1, 2)
        for k in K_RANGE:
            if k >= len(idx):
                break
            lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
            if len(set(lab)) < 2:
                continue
            sc = silhouette_score(X, lab, metric="cosine")
            best = max(best, (sc, k))
        k_best = best[1]
    lab = AgglomerativeClustering(n_clusters=k_best, metric="cosine", linkage="average").fit_predict(X)
    names, raw_names, scores, sims = {}, {}, {}, np.zeros(len(idx), dtype=np.float32)
    for c in set(lab):
        m = lab == c
        cen = X[m].mean(0)
        cen /= np.linalg.norm(cen) + 1e-9
        sims[m] = X[m] @ cen
        u = units.iloc[idx[m]]
        scores[c] = u["wscore"].sum() / max(u["dur"].sum(), 1e-9)
        raw_names[c] = "ENGINEER" if scores[c] > 0 else "DRIVER"
    names = dict(raw_names)
    if len(names) >= 2:
        if "DRIVER" not in names.values():
            names[min(scores, key=scores.get)] = "DRIVER"
        if "ENGINEER" not in names.values():
            names[max(scores, key=scores.get)] = "ENGINEER"
    return lab, k_best, sims, names, raw_names


summary = {"channels": {}, "zero_embeddings_treated_as_missing": n_zero_emb}
VARIANTS = [("silk", None, True), ("silk_raw", None, False), ("k2", 2, True)]
for variant, k_fixed, constrained in VARIANTS:
    units[f"voice_{variant}"] = pd.NA
    units[f"sim_{variant}"] = np.nan
    big = []
    for drv, g in units.groupby("driver_id"):
        n_clips = clips[clips["driver_id"] == drv].shape[0]
        if n_clips < MIN_CLIPS_PER_CHANNEL or len(g) < 3:
            continue
        idx = g.index.to_numpy()
        lab, k, sims, names, raw_names = cluster_channel(idx, k_fixed)
        if not constrained:
            names = raw_names
        units.loc[idx, f"voice_{variant}"] = [names[c] for c in lab]
        units.loc[idx, f"sim_{variant}"] = sims
        big.append(drv)
        summary["channels"].setdefault(drv, {})[variant] = {
            "k": int(k), "n_units": int(len(idx)),
            "clusters": {names[c] + f"_{c}": int((lab == c).sum()) for c in sorted(set(lab))}}
    tr = units[f"voice_{variant}"].notna()
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(E[tr.to_numpy()], units.loc[tr, f"voice_{variant}"].astype(str))
    small = ~tr
    if small.any():
        units.loc[small, f"voice_{variant}"] = clf.predict(E[small.to_numpy()])
        units.loc[small, f"sim_{variant}"] = np.nan
    summary[f"small_channel_units_{variant}"] = int(small.sum())

# Per-turn labels
vcols = [c for v, _, _ in VARIANTS for c in (f"voice_{v}", f"sim_{v}")]
turns = turns.merge(units[["clip_id", "spk"] + vcols], on=["clip_id", "spk"], how="left")
turns["text_label"] = np.select([turns["text_score"] > 0, turns["text_score"] < 0],
                                ["ENGINEER", "DRIVER"], default="NONE")

for variant, _, _ in VARIANTS:
    v, sim = turns[f"voice_{variant}"], turns[f"sim_{variant}"]
    thr = units.groupby("driver_id")[f"sim_{variant}"].quantile(LOW_SIM_QUANTILE)
    low = sim < turns["driver_id"].map(thr)
    use_text = (v.isna() | low.fillna(False)) & (turns["text_label"] != "NONE")
    ens = v.astype(object).where(~use_text, turns["text_label"])
    ens = ens.where(ens.notna(), "UNATTRIBUTED")
    turns[f"ens_{variant}"] = ens
    turns[f"ens_{variant}_source"] = np.where(use_text, "text", np.where(v.notna(), "voice", "none"))

turns["attributable"] = turns["dur"] >= min_s
turns.to_parquet(processed / "turns_attributed.parquet", index=False)

a = turns[turns["attributable"]]
summary.update({
    "n_turns": int(len(turns)), "n_turns_ge_min": int(len(a)),
    "turn_share_by_label_silk": a["ens_silk"].value_counts(normalize=True).round(3).to_dict(),
    "driver_speech_hours_silk": round(a.loc[a["ens_silk"] == "DRIVER", "dur"].sum() / 3600, 2),
    "ens_source_silk": a["ens_silk_source"].value_counts().to_dict(),
    "channels_renamed_by_constraint": sorted(
        d for d, v in summary["channels"].items()
        if v["silk"]["clusters"].keys() != v["silk_raw"]["clusters"].keys()),
    "note": "No accuracy here: accuracy is computed only against hand labels (a6_evaluate).",
})
(res_dir / "attribution_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: v for k, v in summary.items() if k != "channels"}, indent=1, default=str))
