# Stage B.0b: fill configs/f1_lexicon.json 'drivers' from FastF1 race results (all seasons)

# Imports
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1radio.config import ROOT, load_config, path

# Main
cfg = load_config()
names = set()
for f in (path(cfg, "raw") / "telemetry").glob("*/results.parquet"):
    r = pd.read_parquet(f, columns=["FirstName", "LastName"])
    names |= {f"{a} {b}" for a, b in zip(r["FirstName"], r["LastName"]) if isinstance(a, str) and isinstance(b, str)}
p = ROOT / "configs" / "f1_lexicon.json"
lex = json.loads(p.read_text(encoding="utf-8", errors="replace"))
lex["drivers"] = sorted(names, key=lambda n: n.split()[-1])
lex["_note"] = lex["_note"].replace("scripts/b1_build_lexicon.py", "scripts/b0_build_lexicon_drivers.py")
p.write_text(json.dumps(lex, indent=2, ensure_ascii=False), encoding="utf-8")
print(len(lex["drivers"]), "drivers:", ", ".join(lex["drivers"][:8]), "...")
