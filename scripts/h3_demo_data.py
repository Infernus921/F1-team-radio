# Stage H.3: data for the interactive demo (reports/demo). Local use only: contains broadcast aud

# Imports
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

SEED, N_PER_CLASS = 7, {"GRIP_DEG": 3, "INCIDENT": 3, "REL_PACE": 3, "NO_SENSOR": 2, "NO_CLAIM": 2}
cfg = load_config()
interim, proc, res = path(cfg, "interim"), path(cfg, "processed"), path(cfg, "results") / "stage_e"
tel = path(cfg, "raw") / "telemetry"
out = path(cfg, "figures").parent / "demo"
(out / "audio").mkdir(parents=True, exist_ok=True)

items = pd.read_csv(path(cfg, "labels") / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(path(cfg, "labels") / "claim_labels.csv", encoding="utf-8", dtype=str).drop_duplicates("item_id", keep="last")
df = items.merge(lab[["item_id", "claim", "arousal", "not_driver"]], on="item_id")
df = df[(df["not_driver"] != "1") & (df["phase"] == "race") & df["lap"].notna()]
llm = pd.read_csv(res / "llm_zeroshot_items.csv")
classes = [c[2:] for c in llm.columns if c.startswith("p_")]
oof = pd.read_csv(res / "oof_cv5.csv")
aro = pd.read_parquet(proc / "arousal_wavlm_ft.parquet").set_index("utt_id")
df = df[df["item_id"].isin(llm["item_id"]) & df["utt_id"].isin(aro.index)]
df = df[df["race_id"].map(lambda r: (tel / r / "car_data.parquet").exists())]

# Showcase scoring (hand-picked, disclosed)
clips_all = pd.read_parquet(interim / "clips.parquet").set_index("id")
turns_all = pd.read_parquet(proc / "turns_attributed.parquet")
n_spk = turns_all[turns_all["attributable"]].groupby("clip_id")["ens_silk"].nunique()
deb = oof[oof["model"] == "deberta"].set_index("item_id")


def score(r):
    k = r["claim"]
    p_llm = float(llm.loc[llm["item_id"] == r["item_id"], f"p_{k}"].iloc[0])
    d = deb.loc[r["item_id"]] if r["item_id"] in deb.index else None
    deb_ok = d is not None and max(classes, key=lambda c: d[f"p_claim_{c}"]) == k
    llm_ok = max(classes, key=lambda c: float(llm.loc[llm["item_id"] == r["item_id"], f"p_{c}"].iloc[0])) == k
    a = aro.loc[r["utt_id"]]
    aro_ok = ["1", "2", "3"][int(np.argmax([a["p_ar_1"], a["p_ar_2"], a["p_ar_3"]]))] == r["arousal"]
    dur = clips_all.loc[r["clip_id"], "duration_s"]
    both = n_spk.get(r["clip_id"], 1) >= 2
    return pd.Series({"llm_ok": llm_ok, "deb_ok": deb_ok, "aro_ok": aro_ok, "p_llm": p_llm,
                      "dur_ok": 4 <= dur <= 15, "both": both,
                      "score": 2 * llm_ok + 2 * deb_ok + aro_ok + p_llm + 0.5 * both + (4 <= dur <= 15)})


df = pd.concat([df.reset_index(drop=True), df.reset_index(drop=True).apply(score, axis=1)], axis=1)
SHOW_IDS = ["C0039", "C0036", "C0318", "C0216", "C0350", "C0356"]
pick = df.set_index("item_id").loc[SHOW_IDS].reset_index()
pick["showcase"] = "example"
fail = df[df["text"].fillna("").str.startswith("Enjoy one more tea")].assign(showcase="failure")
pick = pd.concat([pick, fail]).reset_index(drop=True)
print(pick[["claim", "arousal", "llm_ok", "deb_ok", "aro_ok", "both", "showcase", "text"]].to_string(max_colwidth=80))

turns = pd.read_parquet(proc / "turns_attributed.parquet")
words = {}
want = set(pick["clip_id"])
with open(interim / "asr_plain_words.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] in want:
            words[r["id"]] = r["words"]
clips = pd.read_parquet(interim / "clips.parquet").set_index("id")
J = pd.read_parquet(proc / "clip_laps.parquet").set_index("id")
R = pd.read_parquet(proc / "lap_residuals.parquet")
R["DriverNumber"] = R["DriverNumber"].astype(str)


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


data = []
for n, it in pick.iterrows():
    cid = it["clip_id"]
    c = clips.loc[cid]
    num = str(c["racing_number"])
    meta = json.loads((tel / it["race_id"] / "meta.json").read_text())
    resu = pd.read_parquet(tel / it["race_id"] / "results.parquet")
    resu["DriverNumber"] = resu["DriverNumber"].astype(str)
    who = resu[resu["DriverNumber"] == num]
    name = who["FullName"].iloc[0] if len(who) else it["driver_id"]
    team = who["TeamName"].iloc[0] if len(who) else ""
    shutil.copy(interim / "audio" / f"{cid}.mp3", out / "audio" / f"c{n}.mp3")
    x, sr = sf.read(interim / "audio" / f"{cid}.mp3", dtype="float32", always_2d=True)
    x = np.abs(x.mean(1))
    nb = 600
    edges = np.linspace(0, len(x), nb + 1).astype(int)
    peaks = [float(x[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    pk = max(peaks) or 1.0
    tt = turns[turns["clip_id"] == cid].sort_values("start")
    trn = [{"s": r3(t.start), "e": r3(t.end), "who": t.ens_silk} for t in tt.itertuples()]
    wd = [[r3(w[0]), r3(w[1]), w[2]] for w in words.get(cid, [])]
    o = oof[oof["item_id"] == it["item_id"]].set_index("model")
    preds = {"LLM zero-shot": [r3(llm.loc[llm["item_id"] == it["item_id"], f"p_{k}"].iloc[0]) for k in classes]}
    for m, nm in [("deberta", "Fine-tuned DeBERTa"), ("tfidf", "TF-IDF")]:
        if m in o.index:
            preds[nm] = [r3(o.loc[m, f"p_claim_{k}"]) for k in classes]
    a = aro.loc[it["utt_id"]]
    L = int(it["lap"])
    rr = R[(R["race_id"] == it["race_id"]) & (R["DriverNumber"] == num)].set_index("LapNumber")
    laps = []
    for l in range(L - 7, L + 5):
        if l in rr.index:
            q = rr.loc[l]
            laps.append({"lap": l, "res": r3(q["residual_s"]) if bool(q["clean"]) else None,
                         "pit": bool(q["pit_lap"]), "sc": bool(q["neutralised"])})
    cd = pd.read_parquet(tel / it["race_id"] / "car_data.parquet", columns=["Date", "Speed", "DriverNumber"])
    cd = cd[cd["DriverNumber"].astype(str) == num].set_index("Date")["Speed"].resample("1s").mean()
    ts = pd.Timestamp(J.loc[cid, "ts"]).floor("s")
    grid = pd.date_range(start=ts - pd.Timedelta(seconds=120), periods=181, freq="s")
    sp = cd.reindex(grid)
    data.append({
        "n": int(n), "audio": f"audio/c{n}.mp3", "dur": r3(c["duration_s"]), "driver": name, "team": team,
        "race": meta["event"], "year": int(it["year"]), "lap": L, "race_laps": int(R[R["race_id"] == it["race_id"]]["LapNumber"].max()),
        "peaks": [round(p / pk, 3) for p in peaks], "turns": trn, "words": wd,
        "utt": [r3(it["start"]), r3(it["end"])], "text": it["text"],
        "human": {"claim": it["claim"], "arousal": it["arousal"]},
        "classes": classes, "preds": preds,
        "arousal": [r3(a["p_ar_1"]), r3(a["p_ar_2"]), r3(a["p_ar_3"])],
        "laps": laps, "speed": [None if not np.isfinite(v) else round(float(v)) for v in sp.to_numpy()],
        "showcase": it["showcase"],
    })
    print(n, it["claim"], name, meta["event"], it["year"], "lap", L)

meta_all = {"rule": "hand-picked for illustration: clearest examples where models agree with the human label, plus one failure",
            "llm_macro": json.loads((res / "llm_zeroshot.json").read_text())["metrics"]["macro"]["auc_pr"]}
(out / "data.js").write_text("window.DEMO = " + json.dumps({"meta": meta_all, "clips": data}) + ";", encoding="utf-8")
print("wrote", len(data), "clips ->", out)
