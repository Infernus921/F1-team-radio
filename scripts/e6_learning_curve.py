# Stage E.6: learning curve for T1 claim type (pre-declared contingency: DeBERTa did not beat TF-

# Imports
import json
import runpy
import sys
from pathlib import Path

import numpy as np

# Main
sys.argv = [sys.argv[0], "--min-class", "30", "--protocols", "none"]
ns = runpy.run_path(str(Path(__file__).with_name("e1_text_models.py")), run_name="lc")

df, cfg, kept = ns["df"], ns["cfg"], ns["kept"]
yc, ya, mask_claim, race = ns["yc"], ns["ya"], ns["mask_claim"], ns["race"]
fit_sklearn, fit_deberta, auc_pr_table = ns["fit_sklearn"], ns["fit_deberta"], ns["auc_pr_table"]
grouped_kfold = ns["grouped_kfold"]

# Settings
FRACS = [0.25, 0.5, 0.75, 1.0]
fold = grouped_kfold(df, 5, cfg["seed"]).to_numpy()
rng = np.random.default_rng(cfg["seed"])
out = {"fractions": FRACS, "tfidf": {}, "deberta": {}, "n_train_mean": {}}
for f in FRACS:
    P = {m: np.full((len(df), len(kept)), np.nan) for m in ["tfidf", "deberta"]}
    ntr = []
    for k in np.unique(fold):
        tr_all, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        races = np.unique(race[tr_all])
        keep = rng.choice(races, max(1, int(round(f * len(races)))), replace=False)
        tr = tr_all[np.isin(race[tr_all], keep)]
        ntr.append(int(mask_claim[tr].sum()))
        P["tfidf"][te] = fit_sklearn("tfidf", tr[mask_claim[tr]], te, yc, kept)
        P["deberta"][te] = fit_deberta(tr, te, yc, ya, kept)[0]
    sc = mask_claim
    for m in P:
        t = auc_pr_table(yc[sc], P[m][sc], kept, race[sc], cfg["seed"], n_boot=300)
        out[m][str(f)] = {"macro": t["macro"]["auc_pr"], "ci95": t["macro"]["ci95"]}
    out["n_train_mean"][str(f)] = float(np.mean(ntr))
    print(f, {m: out[m][str(f)] for m in P}, "n_train", out["n_train_mean"][str(f)], flush=True)
(Path(ns["out_dir"]) / "learning_curve.json").write_text(json.dumps(out, indent=1))
