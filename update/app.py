"""Demo: an offline Arabic tool router for AI assistants.

Every AI assistant first decides WHICH tool handles a request (search, image, code, translate...).
Usually that decision is a full LLM call. Here a small Arabic model makes it on the laptop CPU, offline.

Run on your laptop (after copying models/student_final from Colab):
    pip install -r requirements-laptop.txt
    python app.py                 # works with Wi-Fi OFF, opens http://127.0.0.1:7860
"Ask GPT when unsure" needs internet and:  set OPENAI_API_KEY=...  (Windows) | export OPENAI_API_KEY=... (Mac/Linux)

No web framework: Python's built-in HTTP server + one HTML page (web/index.html), no CDN, no fonts to download.
"""
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
from sklearn.metrics import f1_score

from src.config import CASCADE_THRESHOLD, TEACHER_MODEL
from src.deploy import load_student, predict
from src.labels import LABEL_LIST, LABELS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/student_final")
PORT = int(os.environ.get("PORT", 7860))
TEACHER_MS = 707  # measured p50 latency of the GPT-4.1-mini routing call (experiments/compute.csv)
ROOT = os.path.dirname(os.path.abspath(__file__))

print("loading router…")
tok, model = load_student(MODEL_PATH, quantize=False)  # fp32: int8 lost 0.10 macro-F1 (see docs/DECISIONS.md)
LOCK = threading.Lock()
with LOCK:
    predict(tok, model, ["تجربة"])  # warm-up

_big = None


def big_model():
    global _big
    if _big is None:
        from src.llm import client_for
        from src.teacher import Teacher

        def key(name):
            k = os.environ.get(name)
            if not k:
                raise RuntimeError(f"set {name} and connect to the internet")
            return k
        client = client_for(TEACHER_MODEL, key)
        client.min_interval, client.max_retries = 0, 2
        _big = Teacher(client, model=TEACHER_MODEL, cache_path="data/demo_cache.json")
    return _big


def route(text, ask_gpt=False, threshold=CASCADE_THRESHOLD):
    t0 = time.perf_counter()
    with LOCK:
        p = predict(tok, model, [text])[0]
    ms = (time.perf_counter() - t0) * 1000
    order = p.argsort()[::-1]
    out = {"tool": LABEL_LIST[order[0]], "confidence": float(p[order[0]]), "ms": round(ms, 1),
           "alts": [{"tool": LABEL_LIST[i], "confidence": float(p[i])} for i in order[1:3]],
           "source": "router", "teacher_ms": TEACHER_MS}
    if ask_gpt and out["confidence"] < threshold:
        try:
            t1 = time.perf_counter()
            out["router_tool"] = out["tool"]
            out["tool"] = big_model().label(text)
            out["source"], out["gpt_ms"] = "gpt", round((time.perf_counter() - t1) * 1000)
        except Exception as e:
            out["source"], out["error"] = "router", f"GPT unavailable: {str(e)[:80]}"
    return out


def test_set():
    from src.generate import apply_review
    df = pd.read_csv(os.path.join(ROOT, "data/messages.csv"))
    df = apply_review(df[df["set"] == "test"], os.path.join(ROOT, "data/test_review.csv"))
    return df.reset_index(drop=True)


def bench_events():
    df = test_set()
    preds, gold, t0 = [], [], time.perf_counter()
    for i in range(0, len(df), 16):
        chunk = df.iloc[i:i + 16]
        with LOCK:
            probs = predict(tok, model, list(chunk["text"]))
        preds += [LABEL_LIST[p.argmax()] for p in probs]
        gold += list(chunk["gold"])
        n, el = len(preds), time.perf_counter() - t0
        yield {"n": n, "total": len(df), "seconds": round(el, 2), "llm_seconds": round(n * TEACHER_MS / 1000, 1),
               "macro_f1": round(f1_score(gold, preds, average="macro", labels=LABEL_LIST, zero_division=0), 3),
               "counts": {k: preds.count(k) for k in LABEL_LIST}, "last": str(chunk["text"].iloc[-1])[:90]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "web/index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html")
        if self.path == "/api/meta":
            return self._send(200, {"labels": {k: v["ar"] for k, v in LABELS.items()}, "teacher": TEACHER_MODEL,
                                    "teacher_ms": TEACHER_MS, "threshold": CASCADE_THRESHOLD,
                                    "gpt_key": bool(os.environ.get("OPENAI_API_KEY"))})
        if self.path == "/api/bench":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for ev in bench_events():
                    self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"event: done\ndata: {}\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/route":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = str(body.get("text", "")).strip()[:500]
        if not text:
            return self._send(400, {"error": "empty request"})
        self._send(200, route(text, bool(body.get("ask_gpt")), float(body.get("threshold", CASCADE_THRESHOLD))))


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"Arabic Tool Router running at {url}  (Ctrl+C to stop)")
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
