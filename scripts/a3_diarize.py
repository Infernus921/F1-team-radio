# Stage A.3: pyannote diarization of every clip

# Imports
import os

# Main
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path, seed_everything
from f1radio.jsonl import resume_ids

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

cfg = load_config()
seed_everything(cfg["seed"])
interim = path(cfg, "interim")
out_path = interim / "diarization.jsonl"

# Settings
MERGE_GAP_S = 0.3
SR = 16000

from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained(cfg["stage_a"]["diarization_model"], token=True)
pipe.to(torch.device("cuda"))
pipe.segmentation_batch_size = cfg["stage_a"]["diarization_batch_size"]
pipe.embedding_batch_size = cfg["stage_a"]["diarization_batch_size"]

clips = pd.read_parquet(interim / "clips.parquet").sort_values("id")
done = resume_ids(out_path)
todo = [i for i in clips["id"] if i not in done][: args.limit]
print(f"{len(done)} done, {len(todo)} to go", flush=True)


# Functions
def load16k(cid):
    x, sr = sf.read(interim / "audio" / f"{cid}.mp3", dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def merged_turns(annotation):
    turns = []
    for seg, _, spk in annotation.itertracks(yield_label=True):
        if turns and turns[-1][2] == spk and seg.start - turns[-1][1] <= MERGE_GAP_S:
            turns[-1][1] = round(seg.end, 3)
        else:
            turns.append([round(seg.start, 3), round(seg.end, 3), spk])
    return turns


t0, audio_s = time.time(), 0.0
with open(out_path, "a") as f:
    for n, cid in enumerate(todo, 1):
        x = load16k(cid)
        audio_s += len(x) / SR
        rec = {"id": cid, "turns": [], "overlap_s": 0.0, "speech_s": 0.0, "emb": {}}
        if len(x) >= SR // 2:
            out = pipe({"waveform": torch.from_numpy(x)[None], "sample_rate": SR}, max_speakers=3)
            full = out.speaker_diarization
            rec["turns"] = merged_turns(out.exclusive_speaker_diarization)
            rec["speech_s"] = round(full.get_timeline().support().duration(), 3)
            rec["overlap_s"] = round(full.get_overlap().duration(), 3)
            if out.speaker_embeddings is not None:
                for k, spk in enumerate(full.labels()):
                    e = out.speaker_embeddings[k]
                    if np.all(np.isfinite(e)):
                        rec["emb"][spk] = np.round(e, 5).tolist()
        f.write(json.dumps(rec) + "\n")
        if n % 200 == 0 or n == len(todo):
            f.flush()
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {audio_s/el:.1f}x realtime  eta {(len(todo)-n)*el/n/60:.0f} min", flush=True)
