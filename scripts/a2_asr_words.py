# Stage A.2: plain Whisper large-v3 (no initial_prompt) with word timestamps

# Imports
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path, seed_everything
from f1radio.jsonl import resume_ids

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="asr_plain_words.jsonl")
args = ap.parse_args()

cfg = load_config()
seed_everything(cfg["seed"])
interim = path(cfg, "interim")
out_path = interim / args.out

clips = pd.read_parquet(interim / "clips.parquet").sort_values("id")
done = resume_ids(out_path)
todo = [i for i in clips["id"] if i not in done][: args.limit]
print(f"{len(done)} done, {len(todo)} to go", flush=True)

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
t0, audio_s = time.time(), 0.0
with open(out_path, "a") as f:
    for n, cid in enumerate(todo, 1):
        segs, info = model.transcribe(
            str(interim / "audio" / f"{cid}.mp3"),
            language="en", beam_size=5, word_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, temperature=0.0,
        )
        words = [[round(w.start, 2), round(w.end, 2), w.word, round(w.probability, 3)]
                 for s in segs for w in (s.words or [])]
        f.write(json.dumps({"id": cid, "text": "".join(w[2] for w in words).strip(),
                            "words": words}) + "\n")
        audio_s += info.duration
        if n % 200 == 0 or n == len(todo):
            f.flush()
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {audio_s/el:.1f}x realtime  eta {(len(todo)-n)*el/n/60:.0f} min", flush=True)
