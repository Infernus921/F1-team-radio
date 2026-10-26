# Stage B.1: Whisper large-v3 with a season-specific initial_prompt (prompt-only biasing)

# Imports
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import ROOT, load_config, path, seed_everything
from f1radio.jsonl import resume_ids

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--subset", choices=["wer", "all"], required=True)
args = ap.parse_args()

cfg = load_config()
seed_everything(cfg["seed"])
interim = path(cfg, "interim")
out_path = interim / "asr_prompted_words.jsonl"
lex = json.loads((ROOT / "configs" / "f1_lexicon.json").read_text(encoding="utf-8"))
tel = path(cfg, "raw") / "telemetry"

clips = pd.read_parquet(interim / "clips.parquet").set_index("id")
if args.subset == "wer":
    ids = pd.read_csv(path(cfg, "labels") / "wer_items.csv")["clip_id"].tolist()
else:
    ids = sorted(clips.index)


# Functions
def season_names(year: int) -> list[str]:
    names = set()
    for d in tel.glob(f"{year}_*"):
        f = d / "results.parquet"
        if f.exists():
            r = pd.read_parquet(f, columns=["FirstName", "LastName"])
            names |= {f"{a} {b}" for a, b in zip(r["FirstName"], r["LastName"]) if a and b}
    return sorted(names)


# Settings
CORE_TERMS = ["supersoft", "ultrasoft", "hypersoft", "inters", "virtual safety car", "VSC", "DRS", "box box",
              "stay out", "undercut", "delta", "strat mode", "harvesting", "deg", "graining", "understeer",
              "oversteer", "blue flag", "track limits", "Plan B", "copy"]
prompts = {}
for y in sorted(clips["year"].unique()):
    names = season_names(int(y))
    surnames = sorted({n.split()[-1] for n in names})
    firsts = sorted({n.split()[0] for n in names})
    p = ("F1 team radio. Drivers: " + ", ".join(surnames) + ". " + ", ".join(firsts) + ". "
         + "Terms: " + ", ".join(CORE_TERMS) + ".")
    prompts[int(y)] = p[:900]
(path(cfg, "results") / "stage_b").mkdir(exist_ok=True)
(path(cfg, "results") / "stage_b" / "prompts.json").write_text(json.dumps(prompts, indent=1))

done = resume_ids(out_path)
todo = [i for i in ids if i not in done]
print(f"{len(done)} done, {len(todo)} to go", flush=True)

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
t0, audio_s = time.time(), 0.0
with open(out_path, "a") as f:
    for n, cid in enumerate(todo, 1):
        segs, info = model.transcribe(
            str(interim / "audio" / f"{cid}.mp3"),
            language="en", beam_size=5, word_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, temperature=0.0,
            initial_prompt=prompts[int(clips.at[cid, "year"])],
        )
        words = [[round(w.start, 2), round(w.end, 2), w.word, round(w.probability, 3)]
                 for s in segs for w in (s.words or [])]
        f.write(json.dumps({"id": cid, "text": "".join(w[2] for w in words).strip(), "words": words}) + "\n")
        audio_s += info.duration
        if n % 50 == 0 or n == len(todo):
            f.flush()
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {audio_s/el:.1f}x realtime", flush=True)
