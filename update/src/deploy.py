"""Make the student laptop-ready and MEASURE the compute savings.

- int8 dynamic quantization (optional, quantize=True): ~3x faster on CPU but lost ~0.10 macro-F1 on this
  model, so the shipped default is fp32. Applied at load time; the saved model stays standard HF format.
- Benchmark on CPU with 1 thread per request (like a modest laptop): latency p50/p95, throughput,
  model size, RAM.
"""
import io
import os
import time

import numpy as np

from src.labels import LABEL_LIST


def load_student(path, quantize=False, threads=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).float().eval()
    if quantize:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tok, model


def predict(tok, model, texts, max_len=64):
    import torch
    with torch.no_grad():
        enc = tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits, -1).numpy()
    return probs


def model_size_mb(model):
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def benchmark(tok, model, texts, n=200, warmup=10):
    """Latency of single-message requests (the realistic case for a routing service)."""
    texts = list(texts)[:n]
    for t in texts[:warmup]:
        predict(tok, model, [t])
    lat = []
    for t in texts:
        t0 = time.perf_counter()
        predict(tok, model, [t])
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return {"p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "msgs_per_sec": round(1000 / float(lat.mean()), 1), "size_mb": round(model_size_mb(model), 1),
            "ram_mb": round(rss_mb(), 0)}


def route(tok, model, text):
    """Top-3 departments with confidence - used by the demo app."""
    p = predict(tok, model, [text])[0]
    top = np.argsort(-p)[:3]
    return [(LABEL_LIST[i], float(p[i])) for i in top]
