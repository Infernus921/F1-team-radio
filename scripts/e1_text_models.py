# Stage E.1: T1 claim type + T3 text arousal, with every required baseline

# Imports
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path, seed_everything
from f1radio.metrics import auc_pr_table, paired_delta
from f1radio.splits import assert_no_race_leakage, grouped_kfold, lodo_folds, top_drivers

ap = argparse.ArgumentParser()
ap.add_argument("--min-class", type=int, default=100, help="drop claim classes with fewer labels (brief: 100)")
ap.add_argument("--protocols", default="cv5,val2024,test2025,lodo")
ap.add_argument("--annotator", default="A1")
ap.add_argument("--labels-dir", default=None, help="override (used only for plumbing tests on fake labels)")
ap.add_argument("--out-dir", default=None)
ap.add_argument("--epochs", type=int, default=None, help="override (plumbing tests only)")
args = ap.parse_args()

HP = dict(model="microsoft/deberta-v3-base", max_len=64, lr=2e-5, epochs=8, batch=16, wd=0.01, warmup=0.1)
AROUSAL = ["1", "2", "3"]

cfg = load_config()
seed_everything(cfg["seed"])
if args.epochs:
    HP["epochs"] = args.epochs
out_dir = Path(args.out_dir) if args.out_dir else path(cfg, "results") / "stage_e"
out_dir.mkdir(parents=True, exist_ok=True)
labels_dir = Path(args.labels_dir) if args.labels_dir else path(cfg, "labels")

# Data
items = pd.read_csv(labels_dir / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(labels_dir / "claim_labels.csv", encoding="utf-8", dtype=str)
lab = lab[lab["annotator"] == args.annotator].drop_duplicates("item_id", keep="last")
df = items.merge(lab[["item_id", "claim", "arousal", "not_driver"]], on="item_id")
n_not_driver = int((df["not_driver"] == "1").sum())
df = df[df["not_driver"] != "1"].copy()
df["text"] = df["text"].fillna("")

counts = df["claim"].value_counts()
kept = sorted(counts[counts >= args.min_class].index)
dropped = counts[counts < args.min_class].to_dict()
print("claim counts:", counts.to_dict(), "| kept:", kept, "| dropped (<%d):" % args.min_class, dropped, flush=True)

clips = pd.read_parquet(path(cfg, "interim") / "clips.parquet")[["id", "racing_number"]]
R = pd.read_parquet(path(cfg, "processed") / "lap_residuals.parquet",
                    columns=["race_id", "DriverNumber", "LapNumber", "Position", "TyreLife", "gap_ahead_prev_s"])
R["DriverNumber"] = R["DriverNumber"].astype(str)
df = df.merge(clips, left_on="clip_id", right_on="id", how="left").drop(columns="id")
df["racing_number"] = df["racing_number"].astype(str)
df = df.merge(R, left_on=["race_id", "racing_number", "lap"], right_on=["race_id", "DriverNumber", "LapNumber"], how="left")
maxlap = R.groupby("race_id")["LapNumber"].max()
df["lap_frac"] = df["lap"] / df["race_id"].map(maxlap)
CTX = ["lap_frac", "Position", "TyreLife", "gap_ahead_prev_s"]
X_ctx = df[CTX].astype(float)
X_ctx = pd.concat([X_ctx.fillna(X_ctx.median()), X_ctx.isna().astype(float).add_suffix("_na"),
                   pd.get_dummies(df["phase"], prefix="ph", dtype=float),
                   pd.get_dummies(df["year"].astype(str), prefix="y", dtype=float)], axis=1).to_numpy()
df = df.reset_index(drop=True)


# Models
def fit_sklearn(kind, tr, te, y, classes):
    if kind == "majority":
        prior = np.array([(y[tr] == c).mean() for c in classes])
        return np.tile(prior, (len(te), 1))
    if kind == "context":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        m.fit(X_ctx[tr], y[tr])
    else:
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0))
        m.fit(df["text"].to_numpy()[tr], y[tr])
    P = np.zeros((len(te), len(classes)))
    src = X_ctx[te] if kind == "context" else df["text"].to_numpy()[te]
    p = m.predict_proba(src)
    for j, c in enumerate(m.classes_):
        P[:, classes.index(c)] = p[:, j]
    return P


class MultiHead(torch.nn.Module):
    def __init__(self, n_claim, n_ar):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(HP["model"], dtype=torch.float32)
        h = self.enc.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.claim = torch.nn.Linear(h, n_claim)
        self.ar = torch.nn.Linear(h, n_ar)

    def forward(self, ids, mask):
        z = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        z = self.drop(z)
        return self.claim(z), self.ar(z)


_tok = None


def fit_deberta(tr, te, yc, ya, claims):
    global _tok
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(HP["model"])
    seed_everything(cfg["seed"])
    enc = _tok(df["text"].tolist(), truncation=True, max_length=HP["max_len"], padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    yc_i = torch.tensor([claims.index(c) if c in claims else -100 for c in yc])
    ya_i = torch.tensor([AROUSAL.index(a) if a in AROUSAL else -100 for a in ya])
    model = MultiHead(len(claims), len(AROUSAL)).cuda()
    wc = torch.tensor([1.0 / max(1, (yc[tr] == c).sum()) for c in claims], dtype=torch.float)
    wa = torch.tensor([1.0 / max(1, (ya[tr] == a).sum()) for a in AROUSAL], dtype=torch.float)
    lc = torch.nn.CrossEntropyLoss(weight=(wc / wc.sum() * len(claims)).cuda(), ignore_index=-100)
    la = torch.nn.CrossEntropyLoss(weight=(wa / wa.sum() * len(AROUSAL)).cuda(), ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    steps = HP["epochs"] * int(np.ceil(len(tr) / HP["batch"]))
    sch = get_linear_schedule_with_warmup(opt, int(HP["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(cfg["seed"])
    model.train()
    for _ in range(HP["epochs"]):
        for b in np.array_split(rng.permutation(tr), int(np.ceil(len(tr) / HP["batch"]))):
            with torch.autocast("cuda", dtype=torch.float16):
                oc, oa = model(ids[b].cuda(), mask[b].cuda())
                loss_c = lc(oc.float(), yc_i[b].cuda()) if (yc_i[b] >= 0).any() else 0.0
                loss_a = la(oa.float(), ya_i[b].cuda()) if (ya_i[b] >= 0).any() else 0.0
                loss = loss_c + loss_a
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
    model.eval()
    Pc, Pa = [], []
    with torch.no_grad():
        for b in np.array_split(te, max(1, int(np.ceil(len(te) / 64)))):
            with torch.autocast("cuda", dtype=torch.float16):
                oc, oa = model(ids[b].cuda(), mask[b].cuda())
            Pc.append(torch.softmax(oc.float(), -1).cpu().numpy())
            Pa.append(torch.softmax(oa.float(), -1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(Pc), np.concatenate(Pa)


# Protocols
yc = df["claim"].to_numpy()
ya = df["arousal"].to_numpy()
race = df["race_id"].to_numpy()
mask_claim = np.isin(yc, kept)


def splits(protocol):
    if protocol == "cv5":
        fold = grouped_kfold(df, cfg["splits"]["n_group_folds"], cfg["seed"])
        assert_no_race_leakage(df["race_id"], fold)
        for k in sorted(fold.unique()):
            yield f"cv5_{k}", np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    elif protocol in ("val2024", "test2025"):
        tr = np.flatnonzero(df["year"].isin(cfg["splits"]["train_years"]))
        yrs = cfg["splits"]["val_years"] if protocol == "val2024" else cfg["splits"]["test_years"]
        yield f"chrono_{protocol}", tr, np.flatnonzero(df["year"].isin(yrs))
    elif protocol == "lodo":
        fold = grouped_kfold(df, cfg["splits"]["n_group_folds"], cfg["seed"])
        for d, k, tr_i, te_i in lodo_folds(df, fold, top_drivers(df, cfg["splits"]["lodo_top_n"])):
            yield f"lodo_{d}_{k}", tr_i.to_numpy(), te_i.to_numpy()


results = {"n_items": int(len(df)), "n_not_driver_flagged": n_not_driver, "claim_counts": counts.to_dict(),
           "kept_claims": kept, "dropped_claims": dropped, "min_class": args.min_class, "hyperparams": HP,
           "llm_zero_shot": "not run: no LLM API available", "protocols": {}}
models = ["majority", "context", "tfidf", "deberta"]
for protocol in args.protocols.split(","):
    if protocol == "none":
        continue
    t0 = time.time()
    oof = {m: {"claim": {}, "arousal": {}} for m in models}
    for name, tr, te in splits(protocol):
        trc = tr[mask_claim[tr]]
        tra = tr[np.isin(ya[tr], AROUSAL)]
        for m in models:
            if protocol == "lodo" and m in ("majority", "context"):
                continue
            if m == "deberta":
                Pc, Pa = fit_deberta(tr, te, yc, ya, kept)
            else:
                Pc = fit_sklearn(m, trc, te, yc, kept)
                Pa = fit_sklearn(m, tra, te, ya, AROUSAL)
            for i, row in zip(te, Pc):
                oof[m]["claim"].setdefault(i, []).append(row)
            for i, row in zip(te, Pa):
                oof[m]["arousal"].setdefault(i, []).append(row)
        print(f"{protocol} {name}: done ({time.time()-t0:.0f}s)", flush=True)

    res = {}
    rows = []
    for m in models:
        if not oof[m]["claim"]:
            continue
        idx = np.array(sorted(oof[m]["claim"]))
        Pc = np.stack([np.mean(oof[m]["claim"][i], 0) for i in idx])
        Pa = np.stack([np.mean(oof[m]["arousal"][i], 0) for i in idx])
        sc = mask_claim[idx]
        sa = np.isin(ya[idx], AROUSAL)
        res[m] = {"claim": auc_pr_table(yc[idx][sc], Pc[sc], kept, race[idx][sc], cfg["seed"]),
                  "arousal": auc_pr_table(ya[idx][sa], Pa[sa], AROUSAL, race[idx][sa], cfg["seed"])}
        for j, i in enumerate(idx):
            rows.append({"protocol": protocol, "model": m, "row": int(i), "item_id": df.at[i, "item_id"],
                         **{f"p_claim_{c}": Pc[j, k] for k, c in enumerate(kept)},
                         **{f"p_ar_{a}": Pa[j, k] for k, a in enumerate(AROUSAL)}})
        if m == "deberta":
            store = {"Pc": Pc, "Pa": Pa, "idx": idx}
        if m == "context":
            ctx_store = {"Pc": Pc, "Pa": Pa, "idx": idx}
    if protocol != "lodo":
        same = np.isin(store["idx"], ctx_store["idx"])
        i = store["idx"][same]
        sc = mask_claim[i]
        res["deberta_minus_context"] = {
            "claim": paired_delta(yc[i][sc], store["Pc"][same][sc], ctx_store["Pc"][sc], kept, race[i][sc], cfg["seed"]),
            "arousal": paired_delta(ya[i][np.isin(ya[i], AROUSAL)], store["Pa"][same][np.isin(ya[i], AROUSAL)],
                                    ctx_store["Pa"][np.isin(ya[i], AROUSAL)], AROUSAL,
                                    race[i][np.isin(ya[i], AROUSAL)], cfg["seed"])}
    results["protocols"][protocol] = res
    pd.DataFrame(rows).to_csv(out_dir / f"oof_{protocol}.csv", index=False)
    (out_dir / "text_metrics.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"== {protocol} finished in {time.time()-t0:.0f}s", flush=True)

print(json.dumps({p: {m: v["claim"]["macro"] if isinstance(v, dict) and "claim" in v and "macro" in v["claim"] else v
                      for m, v in r.items()} for p, r in results["protocols"].items()}, indent=1, default=str))
