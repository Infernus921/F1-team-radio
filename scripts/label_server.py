# Local labelling server for Stage A speaker attribution (stdlib only, 127.0.0.1)

# Imports
import argparse
import csv
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

# Settings
ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "labels" / "attribution_items.csv"
LABELS = ROOT / "labels" / "attribution_labels.csv"
AUDIO = ROOT / "data" / "interim" / "audio"
UI = ROOT / "scripts" / "label_ui.html"
PAD_S = 0.25
FIELDS = ["item_id", "clip_id", "label", "annotator", "ts"]

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--annotator", required=True)
ap.add_argument("--port", type=int, default=8765)
args = ap.parse_args()

with open(ITEMS, newline="") as f:
    items = list(csv.DictReader(f))
by_id = {it["item_id"]: it for it in items}


# Functions
def load_labels():
    out = {}
    if LABELS.exists():
        with open(LABELS, newline="") as f:
            for r in csv.DictReader(f):
                if r["annotator"] == args.annotator:
                    out[r["item_id"]] = r["label"]
    return out


def segment_wav(it):
    x, sr = sf.read(AUDIO / f"{it['clip_id']}.mp3", dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    a = max(0, int((float(it["start"]) - PAD_S) * sr))
    b = min(len(x), int((float(it["end"]) + PAD_S) * sr))
    seg = x[a:b]
    peak = np.abs(seg).max() if len(seg) else 0
    if peak > 0:
        seg = seg / peak * 0.9
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
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
                     "items": [{"item_id": it["item_id"], "dur": round(float(it["end"]) - float(it["start"]), 1)}
                               for it in items],
                     "labels": load_labels()}
            return self._send(json.dumps(state).encode(), "application/json")
        if u.path == "/seg":
            return self._send(segment_wav(by_id[q["item"][0]]), "audio/wav")
        if u.path == "/clip":
            it = by_id[q["item"][0]]
            return self._send((AUDIO / f"{it['clip_id']}.mp3").read_bytes(), "audio/mpeg")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/label":
            return self._send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        it = by_id[body["item_id"]]
        new = not LABELS.exists()
        with open(LABELS, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({"item_id": it["item_id"], "clip_id": it["clip_id"], "label": body["label"],
                        "annotator": args.annotator, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._send(b"{}", "application/json")


print(f"{len(items)} items, annotator={args.annotator}. Open http://127.0.0.1:{args.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
