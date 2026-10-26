# Local Stage C annotation page: claim type + arousal for driver-side utterances (127.0.0.1)

# Imports
import argparse
import csv
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

# Settings
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labels" / "claim_labels.csv"
AUDIO = ROOT / "data" / "interim" / "audio"
UI = ROOT / "scripts" / "claim_ui.html"
PAD_S = 0.25
FIELDS = ["item_id", "clip_id", "claim", "arousal", "multi_claim", "not_driver", "annotator", "ts"]

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--annotator", required=True)
ap.add_argument("--items", default=str(ROOT / "labels" / "claim_items.csv"))
ap.add_argument("--overlap-only", action="store_true", help="second annotator: only the overlap subset")
ap.add_argument("--port", type=int, default=8767)
args = ap.parse_args()

with open(args.items, newline="", encoding="utf-8") as f:
    items = list(csv.DictReader(f))
if args.overlap_only:
    items = [it for it in items if it.get("overlap_set") == "1"]
by_id = {it["item_id"]: it for it in items}


# Functions
def load_done():
    out = {}
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["annotator"] == args.annotator:
                    out[r["item_id"]] = r
    return out


def segment_wav(it):
    x, sr = sf.read(AUDIO / f"{it['clip_id']}.mp3", dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    a = max(0, int((float(it["start"]) - PAD_S) * sr))
    b = min(len(x), int((float(it["end"]) + PAD_S) * sr))
    buf = io.BytesIO()
    sf.write(buf, x[a:b], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(UI.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/state":
            state = {"annotator": args.annotator,
                     "items": [{"item_id": it["item_id"], "text": it.get("text", ""),
                                "dur": round(float(it["end"]) - float(it["start"]), 1)} for it in items],
                     "done": load_done()}
            return self._send(json.dumps(state).encode(), "application/json")
        if u.path == "/seg":
            return self._send(segment_wav(by_id[q["item"][0]]), "audio/wav")
        if u.path == "/clip":
            return self._send((AUDIO / f"{by_id[q['item'][0]]['clip_id']}.mp3").read_bytes(), "audio/mpeg")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/save":
            return self._send(b"not found", "text/plain", 404)
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        it = by_id[b["item_id"]]
        new = not OUT.exists()
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({"item_id": it["item_id"], "clip_id": it["clip_id"], "claim": b.get("claim", ""),
                        "arousal": b.get("arousal", ""), "multi_claim": int(bool(b.get("multi_claim"))),
                        "not_driver": int(bool(b.get("not_driver"))), "annotator": args.annotator,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._send(b"{}", "application/json")


print(f"{len(items)} items, annotator={args.annotator}. Open http://127.0.0.1:{args.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
