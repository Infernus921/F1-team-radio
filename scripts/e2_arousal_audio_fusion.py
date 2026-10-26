# Stage E.2: T3 arousal from audio, text, and both; plus arousal probabilities for ALL driver

# Imports
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path, seed_everything
from f1radio.metrics import auc_pr_table, paired_delta
from f1radio.splits import assert_no_race_leakage, grouped_kfold
from f1radio.utterances import driver_utterances

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--labels-dir", default=None, help="override (plumbing tests on fake labels only)")
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

AROUSAL = ["1", "2", "3"]
LAYERS = slice(4, 10)
cfg = load_config()
seed_everything(cfg["seed"])
proc = path(cfg, "processed")
out_dir = Path(args.out_dir) if args.out_dir else path(cfg, "results") / "stage_e"
out_dir.mkdir(parents=True, exist_ok=True)
labels_dir = Path(args.labels_dir) if args.labels_dir else path(cfg, "labels")
t4_out = proc / "arousal_all_utterances.parquet" if not args.out_dir else out_dir / "arousal_all_utterances.parquet"

# All driver utterances + audio features
U = driver_utterances(pd.read_parquet(proc / "turns_attributed.parquet"))
z = np.load(proc / "audio_feats.npz", allow_pickle=False)
A = pd.DataFrame(np.concatenate([z["wavlm"][:, LAYERS, :].astype(np.float32).mean(1), z["prosody"]], 1))
A.columns = [f"w{i}" for i in range(768)] + list(z["prosody_names"])
A["utt_id"] = z["utt_id"]
U = U.merge(A, on="utt_id", how="inner")
clips = pd.read_parquet(path(cfg, "interim") / "clips.parquet")[["id", "race_id", "year"]]
U = U.merge(clips, left_on="clip_id", right_on="id").drop(columns="id")
feat_cols = [c for c in A.columns if c != "utt_id"]
X = U[feat_cols].to_numpy(np.float64)
X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
grp = (U["driver_id"] + "|" + U["race_id"]).to_numpy()
for gname in np.unique(grp):
    m = grp == gname
    if m.sum() >= 3:
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    else:
        X[m] = 0.0
season = pd.get_dummies(U["year"].astype(str), prefix="y", dtype=float).to_numpy()
XA = np.concatenate([X, season], 1)
print("utterances with audio:", len(U), flush=True)

# Labels
items = pd.read_csv(labels_dir / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(labels_dir / "claim_labels.csv", encoding="utf-8", dtype=str)
lab = lab[lab["annotator"] == "A1"].drop_duplicates("item_id", keep="last")
L = items[["item_id", "utt_id"]].merge(lab[["item_id", "arousal", "not_driver"]], on="item_id")
L = L[(L["not_driver"] != "1") & L["arousal"].isin(AROUSAL)]
U = U.merge(L[["utt_id", "arousal"]], on="utt_id", how="left")
labelled = U["arousal"].notna().to_numpy()
print("labelled utterances with audio:", int(labelled.sum()), U.loc[labelled, "arousal"].value_counts().to_dict())


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))


def tfidf_lr():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0))


def proba(m, Xte):
    P = np.zeros((Xte.shape[0], 3))
    p = m.predict_proba(Xte)
    for j, c in enumerate(m.classes_):
        P[:, AROUSAL.index(c)] = p[:, j]
    return P


def run_fold(tr, te):
    y = U["arousal"].to_numpy()
    text = U["text"].fillna("").to_numpy()
    out = {}
    out["audio"] = proba(lr().fit(XA[tr], y[tr]), XA[te])
    out["text"] = proba(tfidf_lr().fit(text[tr], y[tr]), text[te])
    inner = grouped_kfold(U.iloc[tr], 5, cfg["seed"] + 1).to_numpy()
    t_oof = np.zeros((len(tr), 3))
    for k in np.unique(inner):
        a, b = tr[inner != k], tr[inner == k]
        t_oof[inner == k] = proba(tfidf_lr().fit(text[a], y[a]), text[b])
    fz = lr().fit(np.concatenate([XA[tr], t_oof], 1), y[tr])
    out["fused"] = proba(fz, np.concatenate([XA[te], out["text"]], 1))
    out["majority"] = np.tile([(y[tr] == a).mean() for a in AROUSAL], (len(te), 1))
    return out


LU = U[labelled].reset_index(drop=False).rename(columns={"index": "u_row"})
res, oof = {}, {}
fold = grouped_kfold(LU, cfg["splits"]["n_group_folds"], cfg["seed"])
assert_no_race_leakage(LU["race_id"], fold)
for proto in ["cv5", "test2025"]:
    P = {m: np.full((len(LU), 3), np.nan) for m in ["audio", "text", "fused", "majority"]}
    if proto == "cv5":
        parts = [(LU["u_row"][fold != k].to_numpy(), LU.index[fold == k].to_numpy(), LU["u_row"][fold == k].to_numpy())
                 for k in sorted(fold.unique())]
    else:
        trm = LU["year"].isin(cfg["splits"]["train_years"]).to_numpy()
        tem = LU["year"].isin(cfg["splits"]["test_years"]).to_numpy()
        parts = [(LU["u_row"][trm].to_numpy(), LU.index[tem].to_numpy(), LU["u_row"][tem].to_numpy())]
    for tr_u, te_l, te_u in parts:
        o = run_fold(tr_u, te_u)
        for m in P:
            P[m][te_l] = o[m]
    done = ~np.isnan(P["audio"][:, 0])
    y, race = LU["arousal"].to_numpy()[done], LU["race_id"].to_numpy()[done]
    res[proto] = {m: auc_pr_table(y, P[m][done], AROUSAL, race, cfg["seed"]) for m in P}
    for a, b in [("audio", "majority"), ("fused", "text"), ("fused", "audio")]:
        res[proto][f"{a}_minus_{b}"] = paired_delta(y, P[a][done], P[b][done], AROUSAL, race, cfg["seed"])
    if proto == "cv5":
        oof = P

best = max(["audio", "text", "fused"], key=lambda m: res["cv5"][m]["macro"]["auc_pr"])
res["t4_model"] = best
print("cv5 macro AUC-PR:", {m: res["cv5"][m]["macro"]["auc_pr"] for m in ["majority", "audio", "text", "fused"]}, "| T4 uses:", best)

# Arousal probabilities for every driver utterance
all_rows = np.arange(len(U))
unl = all_rows[~labelled]
o = run_fold(LU["u_row"].to_numpy(), unl)
Pall = np.zeros((len(U), 3))
Pall[unl] = o[best]
Pall[LU["u_row"].to_numpy()] = oof[best]
outU = U[["utt_id", "clip_id", "driver_id", "race_id", "year", "start", "end"]].copy()
for k, a in enumerate(AROUSAL):
    outU[f"p_ar_{a}"] = Pall[:, k]
outU["p_high"] = Pall[:, 1] + Pall[:, 2]
outU["arousal_label"] = U["arousal"]
outU["prob_source"] = np.where(labelled, "oof_cv5", "fit_all_labels")
outU.to_parquet(t4_out, index=False)
(out_dir / "arousal_metrics.json").write_text(json.dumps(res, indent=1, default=str))
print("wrote arousal probabilities for", len(outU), "utterances")
