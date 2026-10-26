# Stage H.1: error analysis for T1 (cv5 out-of-fold predictions): confusion matrices + examples

# Imports
import sys
from pathlib import Path

import matplotlib

# Main
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

cfg = load_config()
res, figs = path(cfg, "results") / "stage_e", path(cfg, "figures")
oof = pd.read_csv(res / "oof_cv5.csv")
items = pd.read_csv(path(cfg, "labels") / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(path(cfg, "labels") / "claim_labels.csv", encoding="utf-8", dtype=str).drop_duplicates("item_id", keep="last")
gold = items.merge(lab[["item_id", "claim"]], on="item_id")[["item_id", "text", "claim", "driver_id", "race_id"]]
classes = [c.replace("p_claim_", "") for c in oof.columns if c.startswith("p_claim_")]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
ex_rows = []
for ax, model in zip(axes, ["tfidf", "deberta"]):
    d = oof[oof["model"] == model].merge(gold, on="item_id")
    d = d[d["claim"].isin(classes)]
    d["pred"] = np.array(classes)[d[[f"p_claim_{c}" for c in classes]].to_numpy().argmax(1)]
    cm = pd.crosstab(pd.Categorical(d["claim"], classes), pd.Categorical(d["pred"], classes), dropna=False)
    cm.to_csv(res / f"confusion_{model}.csv")
    norm = cm.div(cm.sum(1).replace(0, 1), axis=0)
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm.iat[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=9,
                    color="#ffffff" if norm.iat[i, j] > 0.55 else "#0b0b0b")
    short = [c.replace("_", "\n") for c in classes]
    ax.set_xticks(range(len(classes)), short, fontsize=8)
    ax.set_yticks(range(len(classes)), short, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("hand label")
    acc = (d["pred"] == d["claim"]).mean()
    ax.set_title(f"{model}: accuracy {acc:.2f} (n={len(d)})", loc="left", fontsize=10)
    ax.grid(False)
    if model == "deberta":
        d["p_true"] = [r[f"p_claim_{r['claim']}"] for _, r in d.iterrows()]
        for kind, sub in [("confident_correct", d[d["pred"] == d["claim"]].nlargest(6, "p_true")),
                          ("confident_wrong", d[d["pred"] != d["claim"]].nsmallest(6, "p_true"))]:
            for _, r in sub.iterrows():
                ex_rows.append({"kind": kind, "item_id": r["item_id"], "gold": r["claim"], "pred": r["pred"],
                                "p_gold": round(r["p_true"], 3), "text": r["text"]})
fig.suptitle("T1 claim type: confusion matrices (5-fold race-grouped CV, out-of-fold)", x=0.02, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(figs / "fig8_confusion.png", dpi=200)
pd.DataFrame(ex_rows).to_csv(res / "error_examples.csv", index=False, encoding="utf-8")
print(pd.DataFrame(ex_rows)[["kind", "gold", "pred", "p_gold", "text"]].to_string(max_colwidth=70))
