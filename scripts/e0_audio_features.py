# Stage E.0: audio features for every driver utterance (T3 arousal)

# Imports
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path, seed_everything
from f1radio.utterances import driver_utterances

# Settings
SR = 16000
MAX_S = 20.0

# Main
cfg = load_config()
seed_everything(cfg["seed"])
interim, processed = path(cfg, "interim"), path(cfg, "processed")

U = driver_utterances(pd.read_parquet(processed / "turns_attributed.parquet"))
print("driver utterances:", len(U), flush=True)

fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval().half()


# Functions
def prosody(x):
    frame, hop = 400, 160
    n = 1 + max(0, len(x) - frame) // hop
    fr = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop][:n]
    rms_db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-8)
    zcr = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(1)
    spec = np.abs(np.fft.rfft(fr * np.hanning(frame), axis=1))
    freqs = np.fft.rfftfreq(frame, 1 / SR)
    cent = (spec * freqs).sum(1) / (spec.sum(1) + 1e-8)
    f0 = AF.detect_pitch_frequency(torch.from_numpy(x)[None], SR).numpy().ravel()
    f0 = f0[(f0 > 70) & (f0 < 500)]
    return [rms_db.mean(), rms_db.std(), cent.mean(), zcr.mean(),
            np.median(f0) if len(f0) else np.nan,
            np.subtract(*np.percentile(f0, [75, 25])) if len(f0) > 3 else np.nan]


PROSODY_NAMES = ["rms_db_mean", "rms_db_std", "spec_centroid_hz", "zcr", "f0_median_hz", "f0_iqr_hz"]
cache = {}
ids, W, P = [], [], []
t0 = time.time()
for n, u in enumerate(U.itertuples(), 1):
    if u.clip_id not in cache:
        cache.clear()
        x, sr = sf.read(interim / "audio" / f"{u.clip_id}.mp3", dtype="float32", always_2d=True)
        x = x.mean(1)
        g = np.gcd(sr, SR)
        cache[u.clip_id] = resample_poly(x, SR // g, sr // g).astype(np.float32)
    x = cache[u.clip_id][int(u.start * SR): int(min(u.end, u.start + MAX_S) * SR)]
    if len(x) < SR // 2:
        continue
    inp = fe(x, sampling_rate=SR, return_tensors="pt")["input_values"].cuda().half()
    with torch.no_grad():
        hs = model(inp, output_hidden_states=True).hidden_states
    W.append(torch.stack([h[0].mean(0) for h in hs]).float().cpu().numpy().astype(np.float16))
    P.append(prosody(x))
    ids.append(u.utt_id)
    if n % 1000 == 0:
        print(f"{n}/{len(U)}  {n/(time.time()-t0):.1f} utt/s", flush=True)

np.savez_compressed(processed / "audio_feats.npz", utt_id=np.array(ids), wavlm=np.stack(W),
                    prosody=np.array(P, dtype=np.float32), prosody_names=np.array(PROSODY_NAMES))
print("saved", len(ids), "utterances")
