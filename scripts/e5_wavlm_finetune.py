# Stage E.5: T3 audio arousal by PARTIAL fine-tuning of WavLM with a learned layer weighting

# Imports
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path, seed_everything
from f1radio.metrics import auc_pr_table
from f1radio.splits import assert_no_race_leakage, grouped_kfold, lodo_folds, top_drivers
from f1radio.utterances import driver_utterances

ap = argparse.ArgumentParser()
ap.add_argument("--labels-dir", default=None)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--protocols", default="cv5,lodo")
ap.add_argument("--epochs", type=int, default=6)
args = ap.parse_args()

SR, MAX_S, BATCH, N_FROZEN = 16000, 8.0, 8, 6
AROUSAL = ["1", "2", "3"]
MODEL = "microsoft/wavlm-base-plus"
cfg = load_config()
seed_everything(cfg["seed"])
proc, interim = path(cfg, "processed"), path(cfg, "interim")
labels_dir = Path(args.labels_dir) if args.labels_dir else path(cfg, "labels")
out_dir = Path(args.out_dir) if args.out_dir else path(cfg, "results") / "stage_e"
out_dir.mkdir(parents=True, exist_ok=True)
probs_out = proc / "arousal_wavlm_ft.parquet" if not args.out_dir else out_dir / "arousal_wavlm_ft.parquet"

# Data
U = driver_utterances(pd.read_parquet(proc / "turns_attributed.parquet"))
clips = pd.read_parquet(interim / "clips.parquet")[["id", "race_id", "year"]]
U = U.merge(clips, left_on="clip_id", right_on="id").drop(columns="id").reset_index(drop=True)
items = pd.read_csv(labels_dir / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(labels_dir / "claim_labels.csv", encoding="utf-8", dtype=str)
lab = lab[lab["annotator"] == "A1"].drop_duplicates("item_id", keep="last")
L = items[["item_id", "utt_id"]].merge(lab[["item_id", "arousal", "not_driver"]], on="item_id")
L = L[(L["not_driver"] != "1") & L["arousal"].isin(AROUSAL)]
U = U.merge(L[["utt_id", "arousal"]], on="utt_id", how="left")
labelled = np.flatnonzero(U["arousal"].notna().to_numpy())
years = sorted(U["year"].unique())
print("utterances:", len(U), "| labelled:", len(labelled), U.loc[labelled, "arousal"].value_counts().to_dict(), flush=True)

fe = AutoFeatureExtractor.from_pretrained(MODEL)
_cache = {}


def load_audio(i):
    u = U.iloc[i]
    if u["clip_id"] not in _cache:
        if len(_cache) > 64:
            _cache.clear()
        x, sr = sf.read(interim / "audio" / f"{u['clip_id']}.mp3", dtype="float32", always_2d=True)
        x = x.mean(1)
        g = np.gcd(sr, SR)
        _cache[u["clip_id"]] = resample_poly(x, SR // g, sr // g).astype(np.float32)
    x = _cache[u["clip_id"]][int(u["start"] * SR): int(u["end"] * SR)]
    if len(x) > MAX_S * SR:
        o = (len(x) - int(MAX_S * SR)) // 2
        x = x[o: o + int(MAX_S * SR)]
    return x if len(x) >= SR // 4 else np.zeros(SR // 4, np.float32)


def batch(idx):
    xs = [load_audio(i) for i in idx]
    enc = fe(xs, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
    season = torch.tensor(np.stack([[float(U.iloc[i]["year"] == y) for y in years] for i in idx]), dtype=torch.float)
    return enc["input_values"], enc["attention_mask"], season


class PartialWavLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(MODEL, dtype=torch.float32)
        self.wavlm.config.layerdrop = 0.0
        self.wavlm.freeze_feature_encoder()
        for p in self.wavlm.feature_projection.parameters():
            p.requires_grad = False
        for p in self.wavlm.encoder.pos_conv_embed.parameters():
            p.requires_grad = False
        for p in self.wavlm.encoder.layer_norm.parameters():
            p.requires_grad = False
        for layer in self.wavlm.encoder.layers[:N_FROZEN]:
            for p in layer.parameters():
                p.requires_grad = False
        n_upper = self.wavlm.config.num_hidden_layers - N_FROZEN
        self.layer_logits = nn.Parameter(torch.zeros(n_upper))
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(self.wavlm.config.hidden_size + len(years), len(AROUSAL))

    def forward(self, x, m, season):
        hs = self.wavlm(x, attention_mask=m, output_hidden_states=True).hidden_states
        upper = torch.stack(hs[N_FROZEN + 1:], 0)
        w = torch.softmax(self.layer_logits, 0)
        h = (w[:, None, None, None] * upper).sum(0)
        fm = self.wavlm._get_feature_vector_attention_mask(h.shape[1], m).unsqueeze(-1).to(h.dtype)
        pooled = (h * fm).sum(1) / fm.sum(1).clamp(min=1)
        return self.head(torch.cat([self.drop(pooled), season.to(pooled)], -1))


def fit_predict(tr, te):
    seed_everything(cfg["seed"])
    model = PartialWavLM().cuda()
    y = torch.tensor([AROUSAL.index(a) for a in U["arousal"].to_numpy()[tr]])
    wc = torch.tensor([1.0 / max(1, (y == k).sum().item()) for k in range(len(AROUSAL))])
    lossf = nn.CrossEntropyLoss(weight=(wc / wc.sum() * len(AROUSAL)).cuda())
    upper = [p for n, p in model.wavlm.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": upper, "lr": 2e-5},
                             {"params": [model.layer_logits, *model.head.parameters()], "lr": 1e-3}], weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(cfg["seed"])
    ymap = dict(zip(tr, y.tolist()))
    for _ in range(args.epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // BATCH)):
            x, m, s = batch(b)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x.cuda(), m.cuda(), s.cuda())
            loss = lossf(out.float(), torch.tensor([ymap[i] for i in b]).cuda())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    model.eval()
    P = []
    with torch.no_grad():
        for b in np.array_split(te, max(1, int(np.ceil(len(te) / 16)))):
            x, m, s = batch(b)
            with torch.autocast("cuda", dtype=torch.float16):
                P.append(torch.softmax(model(x.cuda(), m.cuda(), s.cuda()).float(), -1).cpu().numpy())
    w = torch.softmax(model.layer_logits.detach().float(), 0).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return np.concatenate(P), w


LU = U.iloc[labelled].copy()
fold = grouped_kfold(LU, cfg["splits"]["n_group_folds"], cfg["seed"]).to_numpy()
assert_no_race_leakage(LU["race_id"], pd.Series(fold, index=LU.index))
y_all, race_all = LU["arousal"].to_numpy(), LU["race_id"].to_numpy()
res = {"setup": {"model": MODEL, "frozen_layers": N_FROZEN, "trained_layers": 12 - N_FROZEN, "epochs": args.epochs,
                 "max_s": MAX_S, "batch": BATCH}, "protocols": {}}
oof_cv5 = None
for proto in args.protocols.split(","):
    t0 = time.time()
    P = np.full((len(LU), 3), np.nan)
    weights = []
    if proto == "cv5":
        parts = [(labelled[fold != k], np.flatnonzero(fold == k)) for k in np.unique(fold)]
    else:
        LUr = LU.reset_index(drop=True)
        parts = [(labelled[tr_i.to_numpy()], te_i.to_numpy())
                 for _, _, tr_i, te_i in lodo_folds(LUr, pd.Series(fold), top_drivers(LUr, cfg["splits"]["lodo_top_n"]))]
    for tr_u, te_l in parts:
        p, w = fit_predict(tr_u, labelled[te_l])
        P[te_l] = p
        weights.append(w)
        print(f"{proto}: fold done ({time.time()-t0:.0f}s)", flush=True)
    done = ~np.isnan(P[:, 0])
    r = {"n": int(done.sum()), "metrics": auc_pr_table(y_all[done], P[done], AROUSAL, race_all[done], cfg["seed"]),
         "learned_layer_weights_mean": {f"layer_{N_FROZEN + 1 + i}": round(float(v), 4)
                                        for i, v in enumerate(np.mean(weights, 0))}}
    r["note_ablation"] = "see results/stage_e/arousal_metrics.json cv5.audio for the frozen-feature model"
    res["protocols"][proto] = r
    if proto == "cv5":
        oof_cv5 = P
    print(proto, "macro AUC-PR", r["metrics"]["macro"], "| layer weights", r["learned_layer_weights_mean"], flush=True)

unl = np.setdiff1d(np.arange(len(U)), labelled)
Pu, w_full = fit_predict(labelled, unl)
Pall = np.zeros((len(U), 3))
Pall[unl] = Pu
if oof_cv5 is not None:
    Pall[labelled] = oof_cv5
outU = U[["utt_id", "clip_id", "driver_id", "race_id", "year", "start", "end"]].copy()
for k, a in enumerate(AROUSAL):
    outU[f"p_ar_{a}"] = Pall[:, k]
outU["p_high"] = Pall[:, 1] + Pall[:, 2]
outU["arousal_label"] = U["arousal"]
outU["prob_source"] = np.where(U["arousal"].notna(), "oof_cv5", "fit_all_labels")
outU.to_parquet(probs_out, index=False)
res["full_fit_layer_weights"] = {f"layer_{N_FROZEN + 1 + i}": round(float(v), 4) for i, v in enumerate(w_full)}
(out_dir / "wavlm_ft_metrics.json").write_text(json.dumps(res, indent=1, default=str))
print("wrote", probs_out)
