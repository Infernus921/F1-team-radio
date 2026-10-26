# Local hand-transcription page for Stage B WER (stdlib only, 127.0.0.1)

# Imports
import argparse
import csv
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Settings
ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "labels" / "wer_items.csv"
OUT = ROOT / "labels" / "hand_transcripts.csv"
AUDIO = ROOT / "data" / "interim" / "audio"
UI = ROOT / "scripts" / "transcribe_ui.html"
GLOSSARY = ROOT / "configs" / "f1_lexicon.json"
FIELDS = ["item_id", "clip_id", "text", "unintelligible", "annotator", "ts"]

# Main
ap = argparse.ArgumentParser()
ap.add_argument("--annotator", required=True)
ap.add_argument("--port", type=int, default=8766)
args = ap.parse_args()

with open(ITEMS, newline="") as f:
    items = list(csv.DictReader(f))
by_id = {it["item_id"]: it for it in items}


# Functions
def load_done():
    out = {}
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["annotator"] == args.annotator:
                    out[r["item_id"]] = {"text": r["text"], "unintelligible": r["unintelligible"]}
    return out


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
            lex = json.loads(GLOSSARY.read_text(encoding="utf-8")) if GLOSSARY.exists() else {}
            state = {"annotator": args.annotator,
                     "items": [{"item_id": it["item_id"], "year": it["year"]} for it in items],
                     "done": load_done(), "glossary": lex}
            return self._send(json.dumps(state).encode(), "application/json")
        if u.path == "/clip":
            it = by_id[q["item"][0]]
            return self._send((AUDIO / f"{it['clip_id']}.mp3").read_bytes(), "audio/mpeg")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/save":
            return self._send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        it = by_id[body["item_id"]]
        new = not OUT.exists()
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({"item_id": it["item_id"], "clip_id": it["clip_id"], "text": body["text"].strip(),
                        "unintelligible": int(bool(body.get("unintelligible"))),
                        "annotator": args.annotator, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._send(b"{}", "application/json")


print(f"{len(items)} clips, annotator={args.annotator}. Open http://127.0.0.1:{args.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
