# Stage E.7: LLM zero-shot baseline for T1 claim type (BUILD_BRIEF Stage E baselines)

# Imports
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path
from f1radio.metrics import auc_pr_table

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--min-class", type=int, default=30)
args = ap.parse_args()

# Settings
DEFS = {
    "GRIP_DEG": "tyre grip or degradation: tyres gone, sliding, no grip, under/oversteer from the tyres",
    "TRAFFIC": "cars around them: stuck behind someone, dirty air, blue flags, being held up",
    "REL_PACE": "their pace relative to other cars, or another car's pace",
    "INCIDENT": "contact, being pushed off, a rival's illegal move, track limits, penalties",
    "POWER": "engine or power unit: power loss, harvesting, gearbox, RPM, DRS",
    "NO_SENSOR": "a factual claim public telemetry cannot check: brakes, steering, setup balance, fuel, drinks, radio, seat",
    "NO_CLAIM": "no assertion: acknowledgements, questions, thanks, requests",
}
cfg = load_config()
items = pd.read_csv(path(cfg, "labels") / "claim_items.csv", encoding="utf-8")
lab = pd.read_csv(path(cfg, "labels") / "claim_labels.csv", encoding="utf-8", dtype=str)
lab = lab[lab["annotator"] == "A1"].drop_duplicates("item_id", keep="last")
df = items.merge(lab[["item_id", "claim", "not_driver"]], on="item_id")
df = df[df["not_driver"] != "1"]
counts = df["claim"].value_counts()
kept = sorted(counts[counts >= args.min_class].index)
df = df[df["claim"].isin(kept)].reset_index(drop=True)
print("items:", len(df), "| classes:", kept, flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).cuda().eval()
defs = "\n".join(f"- {k}: {DEFS[k]}" for k in kept)


# Functions
def prompt(text):
    msgs = [{"role": "system", "content": "You label Formula 1 team-radio messages spoken by the driver."},
            {"role": "user", "content": f"Categories:\n{defs}\n\nDriver message: \"{text}\"\n\n"
                                        f"Answer with exactly one category name from: {', '.join(kept)}."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def label_logprobs(text):
    p = tok(prompt(text), return_tensors="pt").input_ids.cuda()
    out = []
    for lab_ in kept:
        c = tok(lab_, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        ids = torch.cat([p, c], 1)
        logits = model(ids).logits[0, p.shape[1] - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, c[0][:, None]).sum().item()
        out.append(lp)
    return np.array(out)


L = np.stack([label_logprobs(t if isinstance(t, str) else "") for t in df["text"]])
P = np.exp(L - L.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)
res = {"model": args.model, "n": int(len(df)), "classes": kept,
       "metrics": auc_pr_table(df["claim"].to_numpy(), P, kept, df["race_id"].to_numpy(), cfg["seed"]),
       "accuracy_argmax": round(float((np.array(kept)[P.argmax(1)] == df["claim"].to_numpy()).mean()), 4)}
out_dir = path(cfg, "results") / "stage_e"
(out_dir / "llm_zeroshot.json").write_text(json.dumps(res, indent=1))
pd.concat([df[["item_id", "claim"]], pd.DataFrame(P, columns=[f"p_{k}" for k in kept])], axis=1).to_csv(
    out_dir / "llm_zeroshot_items.csv", index=False)
print("macro AUC-PR", res["metrics"]["macro"], "| argmax accuracy", res["accuracy_argmax"])
