# Stage A.0: unpack the HF parquet shards into per-clip MP3s + a metadata table

# Imports
import io
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import load_config, path

# Main
cfg = load_config()
raw, interim = path(cfg, "raw"), path(cfg, "interim")
audio_dir = interim / "audio"
audio_dir.mkdir(exist_ok=True)

rows = []
for shard in sorted(raw.glob("train-*.parquet")):
    t = pq.read_table(shard).to_pandas()
    for r in t.itertuples(index=False):
        b = r.audio["bytes"]
        out = audio_dir / f"{r.id}.mp3"
        if not out.exists():
            out.write_bytes(b)
        try:
            info = sf.info(io.BytesIO(b))
            dur, sr, ch = info.duration, info.samplerate, info.channels
        except Exception as e:
            dur, sr, ch = float("nan"), -1, -1
            print(f"decode fail {r.id}: {e}", file=sys.stderr)
        rows.append(dict(
            id=r.id, driver_id=r.driver_id, racing_number=str(r.racing_number),
            grand_prix=r.grand_prix, race_id=r.race_id, session_date=r.session_date,
            message_timestamp=r.message_timestamp, transcription=r.transcription,
            duration_s=dur, sr=sr, channels=ch, n_bytes=len(b),
        ))
    print(f"{shard.name}: {len(t)} clips")

meta = pd.DataFrame(rows)
meta["year"] = meta["race_id"].str.slice(0, 4).astype(int)
assert meta["id"].is_unique, "duplicate clip ids"
meta.to_parquet(interim / "metadata.parquet", index=False)
print(meta.shape, "->", interim / "metadata.parquet")
