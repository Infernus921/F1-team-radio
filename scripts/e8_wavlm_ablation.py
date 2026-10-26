# Stage E.8: paired ablation: WavLM partial fine-tuning (e5) vs frozen WavLM features (e2 'audio'

# Imports
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path
from f1radio.metrics import auc_pr_table, paired_delta
from f1radio.splits import grouped_kfold
from f1radio.utterances import driver_utterances

# Settings
AROUSAL = ["1", "2", "3"]

# Main
cfg = load_config()
proc = path(cfg, "processed")
U = driver_utterances(pd.read_parquet(proc / "turns_attributed.parquet"))
z = np.load(proc / "audio_feats.npz")
A = pd.DataFrame(np.concatenate([z["wavlm"][:, 4:10, :].astype(np.float32).mean(1), z["prosody"]], 1))
A["utt_id"] = z["utt_id"]
U = U.merge(A, on="utt_id").merge(pd.read_parquet(path(cfg, "interim") / "clips.parquet")[["id", "race_id", "year"]],
                                  left_on="clip_id", right_on="id")
feat = [c for c in A.columns if c != "utt_id"]
X = U[feat].to_numpy(np.float64)
X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
g = (U["driver_id"] + "|" + U["race_id"]).to_numpy()
for k in np.unique(g):
    m = g == k
    X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6) if m.sum() >= 3 else 0.0
XA = np.concatenate([X, pd.get_dummies(U["year"].astype(str), dtype=float).to_numpy()], 1)

ft = pd.read_parquet(proc / "arousal_wavlm_ft.parquet")
ft = ft[ft["prob_source"] == "oof_cv5"][["utt_id", "arousal_label", "p_ar_1", "p_ar_2", "p_ar_3"]]
U = U.reset_index(drop=True)
rows = U.index[U["utt_id"].isin(ft["utt_id"])]
LU = U.loc[rows].reset_index(drop=False)
LU = LU.merge(ft, on="utt_id")
fold = grouped_kfold(LU, cfg["splits"]["n_group_folds"], cfg["seed"]).to_numpy()
y = LU["arousal_label"].to_numpy()
P_frozen = np.zeros((len(LU), 3))
for k in np.unique(fold):
    tr, te = fold != k, fold == k
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
    m.fit(XA[LU["index"].to_numpy()[tr]], y[tr])
    p = m.predict_proba(XA[LU["index"].to_numpy()[te]])
    for j, c in enumerate(m.classes_):
        P_frozen[te, AROUSAL.index(c)] = p[:, j]
P_ft = LU[["p_ar_1", "p_ar_2", "p_ar_3"]].to_numpy()
race = LU["race_id"].to_numpy()
out = {"n": int(len(LU)),
       "frozen": auc_pr_table(y, P_frozen, AROUSAL, race, cfg["seed"])["macro"],
       "partial_ft": auc_pr_table(y, P_ft, AROUSAL, race, cfg["seed"])["macro"],
       "ft_minus_frozen": paired_delta(y, P_ft, P_frozen, AROUSAL, race, cfg["seed"])}
(path(cfg, "results") / "stage_e" / "wavlm_ablation.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
