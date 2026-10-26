# Evaluation shared by every Stage E/F model: per-class AUC-PR with race-bootstrap CIs

# Imports
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Settings
N_BOOT = 1000


# Functions
def auc_pr_table(y: np.ndarray, proba: np.ndarray, classes: list[str], race: np.ndarray,
                 seed: int = 1337, n_boot: int = N_BOOT) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    races = np.unique(race)
    idx_by_race = {r: np.flatnonzero(race == r) for r in races}
    out = {}

    def ap(c, ii):
        t = (y[ii] == classes[c]).astype(int)
        if t.sum() == 0 or t.sum() == len(t):
            return np.nan
        return average_precision_score(t, proba[ii, c])

    all_i = np.arange(len(y))
    boots = np.full((n_boot, len(classes)), np.nan)
    for b in range(n_boot):
        ii = np.concatenate([idx_by_race[r] for r in rng.choice(races, len(races), replace=True)])
        boots[b] = [ap(c, ii) for c in range(len(classes))]
    chance = np.array([[average_precision_score((y == name).astype(int), rng.random(len(y)))
                        if 0 < (y == name).sum() < len(y) else np.nan for name in classes] for _ in range(200)])
    for c, name in enumerate(classes):
        bc = boots[:, c][~np.isnan(boots[:, c])]
        out[name] = {
            "auc_pr": _r(ap(c, all_i)),
            "ci95": [_r(np.quantile(bc, 0.025)), _r(np.quantile(bc, 0.975))] if len(bc) else [None, None],
            "prevalence": _r((y == name).mean()),
            "chance_auc_pr": _r(np.nanmean(chance[:, c])),
            "chance_95pct": _r(np.nanquantile(chance[:, c], 0.95)),
            "n_pos": int((y == name).sum()),
        }
    macro = np.nanmean(boots, axis=1)
    chance_macro = np.nanmean(chance, axis=1)
    out["macro"] = {"auc_pr": _r(np.nanmean([out[c]["auc_pr"] for c in classes if out[c]["auc_pr"] is not None])),
                    "ci95": [_r(np.nanquantile(macro, 0.025)), _r(np.nanquantile(macro, 0.975))],
                    "no_skill": _r(np.mean([out[c]["prevalence"] for c in classes])),
                    "chance_auc_pr": _r(np.nanmean(chance_macro)),
                    "chance_95pct": _r(np.nanquantile(chance_macro, 0.95))}
    return out


def paired_delta(y, proba_a, proba_b, classes, race, seed=1337, n_boot=N_BOOT) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    races = np.unique(race)
    idx_by_race = {r: np.flatnonzero(race == r) for r in races}

    def macro(p, ii):
        vals = []
        for c in range(len(classes)):
            t = (y[ii] == classes[c]).astype(int)
            if 0 < t.sum() < len(t):
                vals.append(average_precision_score(t, p[ii, c]))
        return np.mean(vals) if vals else np.nan

    all_i = np.arange(len(y))
    d = macro(proba_a, all_i) - macro(proba_b, all_i)
    bs = []
    for _ in range(n_boot):
        ii = np.concatenate([idx_by_race[r] for r in rng.choice(races, len(races), replace=True)])
        bs.append(macro(proba_a, ii) - macro(proba_b, ii))
    bs = np.array(bs)
    bs = bs[~np.isnan(bs)]
    return {"delta_macro_auc_pr": _r(d), "ci95": [_r(np.quantile(bs, 0.025)), _r(np.quantile(bs, 0.975))],
            "p_a_not_better": _r((bs <= 0).mean())}


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
