# Stage E.4 / F: neural T2: does telemetry carry evidence of WHICH claim the driver made?

# Imports
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path, seed_everything
from f1radio.metrics import auc_pr_table, paired_delta
from f1radio.splits import assert_no_race_leakage, grouped_kfold

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--labels-dir", default=None)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--min-class", type=int, default=100)
args = ap.parse_args()

# Settings
SENSOR = ["GRIP_DEG", "TRAFFIC", "REL_PACE", "INCIDENT", "POWER"]
HP = dict(d=32, heads=4, dropout=0.3, lr=1e-3, wd=1e-3, epochs=60, batch=32, seeds=[0, 1, 2])
cfg = load_config()
labels_dir = Path(args.labels_dir) if args.labels_dir else path(cfg, "labels")
out_dir = Path(args.out_dir) if args.out_dir else path(cfg, "results") / "stage_e"
out_dir.mkdir(parents=True, exist_ok=True)
proc = path(cfg, "processed")

W = np.load(proc / "t2_windows.npz", allow_pickle=True)
widx = {u: i for i, u in enumerate(W["utt_id"])}
items = pd.read_csv(labels_dir / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(labels_dir / "claim_labels.csv", encoding="utf-8", dtype=str)
lab = lab[lab["annotator"] == "A1"].drop_duplicates("item_id", keep="last")
df = items.merge(lab[["item_id", "claim", "not_driver"]], on="item_id")
df = df[(df["not_driver"] != "1") & (df["phase"] == "race") & df["utt_id"].isin(widx)].reset_index(drop=True)
counts = df["claim"].value_counts()
kept = [c for c in SENSOR if counts.get(c, 0) >= args.min_class]
CLASSES = kept + ["OTHER"]
df["y"] = df["claim"].where(df["claim"].isin(kept), "OTHER")
wi = df["utt_id"].map(widx).to_numpy()
print("in-race labelled with windows:", len(df), "| classes:", df["y"].value_counts().to_dict(), flush=True)

car, car_pb = torch.tensor(W["car"][wi]), torch.tensor(W["car_pb"][wi])
laps, laps_pb = torch.tensor(W["laps"][wi]), torch.tensor(W["laps_pb"][wi])
y = torch.tensor([CLASSES.index(c) for c in df["y"]])
race = df["race_id"].to_numpy()
fold = grouped_kfold(df, cfg["splits"]["n_group_folds"], cfg["seed"]).to_numpy()
assert_no_race_leakage(df["race_id"], pd.Series(fold))


# Functions
class Telemetry(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(7, d, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU())
        self.lap = nn.Linear(12, d)
        self.pos = nn.Parameter(torch.randn(1, 23 + 7, d) * 0.02)

    def forward(self, c, l):
        tc = self.conv(c.transpose(1, 2)).transpose(1, 2)
        return torch.cat([tc, self.lap(l)], 1) + self.pos


class T2(nn.Module):
    def __init__(self, n_cls, fusion, d=HP["d"]):
        super().__init__()
        self.fusion = fusion
        self.tel = Telemetry(d)
        self.claim = nn.Embedding(n_cls, d)
        self.drop = nn.Dropout(HP["dropout"])
        if fusion == "gated":
            self.att = nn.MultiheadAttention(d, HP["heads"], dropout=HP["dropout"], batch_first=True)
            self.gate = nn.Linear(2 * d, d)
            self.out = nn.Linear(d, 1)
        else:
            self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(HP["dropout"]), nn.Linear(d, 1))

    def forward(self, c, l):
        T = self.drop(self.tel(c, l))
        B, K = T.shape[0], self.claim.num_embeddings
        q = self.claim.weight.unsqueeze(0).expand(B, K, -1)
        if self.fusion == "gated":
            a, _ = self.att(q, T, T)
            g = torch.sigmoid(self.gate(torch.cat([q, a], -1)))
            h = g * a + (1 - g) * q
            return self.out(self.drop(h)).squeeze(-1)
        pooled = T.mean(1, keepdim=True).expand(B, K, -1)
        return self.mlp(torch.cat([q, pooled], -1)).squeeze(-1)


def train_eval(fusion, tr, te, C, L, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = T2(len(CLASSES), fusion).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    w = torch.tensor([1.0 / max(1, (y[tr] == k).sum().item()) for k in range(len(CLASSES))])
    lossf = nn.CrossEntropyLoss(weight=(w / w.sum() * len(CLASSES)).cuda())
    rng = np.random.default_rng(seed)
    for _ in range(HP["epochs"]):
        m.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // HP["batch"])):
            loss = lossf(m(C[b].cuda(), L[b].cuda()), y[b].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(C[te].cuda(), L[te].cuda()), -1).cpu().numpy()


def cv_neural(fusion, C, L):
    P = np.zeros((len(df), len(CLASSES)))
    for k in np.unique(fold):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        P[te] = np.mean([train_eval(fusion, tr, te, C, L, s) for s in HP["seeds"]], 0)
    return P


def cv_sklearn(Xf):
    P = np.zeros((len(df), len(CLASSES)))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
        m.fit(Xf[tr], y.numpy()[tr])
        p = m.predict_proba(Xf[te])
        for j, c in enumerate(m.classes_):
            P[te, c] = p[:, j]
    return P


yn = np.array(CLASSES)[y.numpy()]
P = {}
P["majority"] = np.zeros((len(df), len(CLASSES)))
for k in np.unique(fold):
    P["majority"][fold == k] = [(y.numpy()[fold != k] == c).mean() for c in range(len(CLASSES))]
centre = laps[:, 4, :].numpy()
P["context"] = cv_sklearn(np.c_[df["lap"].to_numpy(float), centre[:, 3], centre[:, 4], centre[:, 1]])
P["telemetry_lr"] = cv_sklearn(laps.reshape(len(df), -1).numpy())
P["neural_concat"] = cv_neural("concat", car, laps)
P["neural_gated"] = cv_neural("gated", car, laps)
P["placebo_neural_gated"] = cv_neural("gated", car_pb, laps_pb)
P["placebo_telemetry_lr"] = cv_sklearn(laps_pb.reshape(len(df), -1).numpy())

res = {"n": int(len(df)), "classes": CLASSES, "class_counts": df["y"].value_counts().to_dict(),
       "dropped_sensor_classes": {c: int(counts.get(c, 0)) for c in SENSOR if c not in kept},
       "hyperparams": HP, "models": {}, "comparisons": {}}
for m, p in P.items():
    res["models"][m] = auc_pr_table(yn, p, CLASSES, race, cfg["seed"])
for a, b in [("neural_gated", "context"), ("neural_gated", "neural_concat"), ("neural_gated", "telemetry_lr"),
             ("neural_gated", "placebo_neural_gated"), ("telemetry_lr", "placebo_telemetry_lr"),
             ("telemetry_lr", "context")]:
    res["comparisons"][f"{a}_minus_{b}"] = paired_delta(yn, P[a], P[b], CLASSES, race, cfg["seed"])
(out_dir / "t2_neural.json").write_text(json.dumps(res, indent=1, default=str))
for m in P:
    print(f"{m:22s} macro AUC-PR {res['models'][m]['macro']['auc_pr']}  CI {res['models'][m]['macro']['ci95']}  (no-skill {res['models'][m]['macro']['no_skill']})")
for k, v in res["comparisons"].items():
    print(k, v)
