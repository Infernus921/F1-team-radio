# Benchmark pyannote batch sizes (memory) at default step; outputs should be identical

# Imports
import os

# Main
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
import time, numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import resample_poly
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
p = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=True).to(torch.device("cuda"))
ids = pd.read_parquet("data/interim/clips.parquet").sample(40, random_state=7).id.tolist()

# Functions
def load(c):
    x, sr = sf.read(f"data/interim/audio/{c}.mp3", dtype="float32", always_2d=True); x = x.mean(1)
    g = np.gcd(sr, 16000); return resample_poly(x, 16000 // g, sr // g).astype(np.float32)

# Settings
X = [load(c) for c in ids]; aud = sum(len(x) for x in X) / 16000
def run():
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    out = [p({"waveform": torch.from_numpy(x)[None], "sample_rate": 16000}, max_speakers=3).exclusive_speaker_diarization for x in X]
    torch.cuda.synchronize(); return out, aud / (time.time() - t), torch.cuda.max_memory_allocated() / 1e9
ref = None
for bs in [32, 8, 4]:
    p.segmentation_batch_size = bs; p.embedding_batch_size = bs
    run() if ref is None else None
    hyp, sp, mem = run()
    if ref is None: ref = hyp
    der = DiarizationErrorRate()
    for r, h in zip(ref, hyp): der(r, h)
    print(f"batch={bs}: {sp:.1f}x  peak {mem:.2f} GB  DER vs batch32 = {abs(der):.4f}", flush=True)
