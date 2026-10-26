# Stage B.2: WER of each ASR system against the hand transcripts (Checkpoint 2)

# Imports
import json
import re
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import ROOT, load_config, path

# Settings
N_BOOT = 2000

# Main
cfg = load_config()
rng = np.random.default_rng(cfg["seed"])
interim, labels = path(cfg, "interim"), path(cfg, "labels")
out_dir = path(cfg, "results") / "stage_b"
out_dir.mkdir(exist_ok=True)

ONES = "zero one two three four five six seven eight nine".split()


# Functions
def spell_digits(s):
    s = re.sub(r"(\d)\.(\d)", r"\1 point \2", s)
    return re.sub(r"\d", lambda m: f" {ONES[int(m.group())]} ", s)


def norm(s: str) -> str:
    s = s.lower() if isinstance(s, str) else ""
    s = re.sub(r"\[\?\]|\[bleep\]|f\*+\w*|s\*+\w*", " ", s)
    s = s.replace("&", " and ").replace("%", " percent ")
    s = spell_digits(s)
    s = re.sub(r"[^a-z' ]", " ", s)
    s = s.replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


ref = pd.read_csv(labels / "hand_transcripts.csv", encoding="utf-8").drop_duplicates("item_id", keep="last")
ref = ref[ref["unintelligible"] == 0]
clips = pd.read_parquet(interim / "clips.parquet").set_index("id")


def load_jsonl(name):
    p = interim / name
    if not p.exists():
        return {}
    return {r["id"]: r["text"] for r in map(json.loads, open(p, encoding="utf-8"))}


systems = {"dataset_cohere": {c: clips.at[c, "transcription"] for c in ref["clip_id"]},
           "whisper_plain": load_jsonl("asr_plain_words.jsonl"),
           "whisper_prompted": load_jsonl("asr_prompted_words.jsonl")}
norm_map_p = ROOT / "configs" / "asr_normalization.json"
if norm_map_p.exists() and systems["whisper_prompted"]:
    nm = json.loads(norm_map_p.read_text())
    def apply_map(t):
        t = t.lower()
        for a, b in nm.items():
            t = re.sub(rf"\b{re.escape(a.lower())}\b", b.lower(), t)
        return t
    systems["whisper_prompted_lexicon"] = {k: apply_map(v) for k, v in systems["whisper_prompted"].items()}

rows = []
for _, r in ref.iterrows():
    R = norm(r["text"])
    if not R:
        continue
    row = {"item_id": r["item_id"], "clip_id": r["clip_id"], "ref_words": len(R.split())}
    for name, hyp in systems.items():
        if r["clip_id"] not in hyp:
            row[f"{name}_err"] = np.nan
            continue
        H = norm(hyp[r["clip_id"]])
        o = jiwer.process_words(R, H if H else "<empty>")
        row[f"{name}_err"] = o.substitutions + o.deletions + o.insertions
    rows.append(row)
d = pd.DataFrame(rows)
d.to_csv(out_dir / "wer_items.csv", index=False)

out = {"n_clips": int(len(d)), "n_ref_words": int(d["ref_words"].sum()), "systems": {}}
idx = np.arange(len(d))
boots = rng.integers(0, len(d), size=(N_BOOT, len(d)))
base = "dataset_cohere"
for name in systems:
    col = f"{name}_err"
    if d[col].isna().any():
        out["systems"][name] = {"note": f"missing for {int(d[col].isna().sum())} clips; not scored"}
        continue
    e, w = d[col].to_numpy(), d["ref_words"].to_numpy()
    wer = e.sum() / w.sum()
    bw = e[boots].sum(1) / w[boots].sum(1)
    res = {"wer": round(float(wer), 4), "ci95": [round(float(np.quantile(bw, q)), 4) for q in (0.025, 0.975)]}
    if name != base:
        eb = d[f"{base}_err"].to_numpy()
        rel = 1 - e.sum() / eb.sum()
        brel = 1 - e[boots].sum(1) / eb[boots].sum(1)
        res["relative_improvement_vs_dataset"] = round(float(rel), 4)
        res["rel_ci95_paired"] = [round(float(np.quantile(brel, q)), 4) for q in (0.025, 0.975)]
        res["meets_15pct_bar"] = bool(rel >= 0.15)
    out["systems"][name] = res
(out_dir / "wer.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
