"""Demo: an offline Arabic tool router for AI assistants.

Every AI assistant first decides WHICH tool should handle a request (search, image, code, translate...).
Usually that decision is a full LLM call. Here a small Arabic model makes it on the laptop CPU, offline.

Run on your laptop (after copying models/student_final from Colab):
    pip install -r requirements-laptop.txt
    python app.py                 # works with Wi-Fi OFF
"Ask GPT when unsure" needs internet and:  set OPENAI_API_KEY=...  (Windows) | export OPENAI_API_KEY=... (Mac/Linux)
"""
import html
import os
import time

import gradio as gr
import pandas as pd

from src.config import CASCADE_THRESHOLD, TEACHER_MODEL
from src.deploy import load_student, predict
from src.labels import LABEL_LIST, LABELS

GR_MAJOR = int(gr.__version__.split(".")[0])
MODEL_PATH = "models/student_final"
tok, model = load_student(MODEL_PATH, quantize=True)
TEACHER_MS = 678  # measured p50 latency of the GPT-4.1-mini router call (experiments/compute.csv)

ICONS = {"web_search": "🔎", "image_generation": "🎨", "code_assistant": "💻", "translation": "🌐",
         "math_calculation": "🧮", "document_qa": "📄", "scheduling": "📅", "general_chat": "💬"}

CSS = """
.gradio-container {background:#0F1A30 !important; color:#F6F3EE; font-family:'IBM Plex Sans','Segoe UI',sans-serif}
.card {background:#17243F; border:1px solid #26365A; border-radius:16px; padding:22px; color:#F6F3EE}
.big {font-size:44px; font-weight:700}
.muted {color:#9FB0C8; font-size:15px}
.accent {color:#F2A65A}
.row {display:flex; gap:14px; flex-wrap:wrap}
.pill {display:inline-block; padding:4px 12px; border-radius:999px; background:#26365A; font-size:14px; margin:2px}
.bar {height:10px; border-radius:6px; background:#26365A; overflow:hidden}
.bar > div {height:10px; background:#F2A65A}
textarea {direction:rtl; font-size:20px !important}
"""

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


def route_card(text, ask_gpt, threshold):
    if not text.strip():
        return ""
    t0 = time.perf_counter()
    p = predict(tok, model, [text])[0]
    ms = (time.perf_counter() - t0) * 1000
    order = p.argsort()[::-1]
    tool, conf = LABEL_LIST[order[0]], float(p[order[0]])
    source = f"⚡ Router on this CPU · offline · {ms:.0f} ms · $0"
    if ask_gpt and conf < threshold:
        try:
            t1 = time.perf_counter()
            tool = big_model().label(text)
            source = (f"🧠 Router unsure ({conf:.0%} &lt; {threshold:.0%}) → asked {TEACHER_MODEL} · "
                      f"{(time.perf_counter() - t1) * 1000:.0f} ms · API call")
        except Exception as e:
            source += f" · GPT unavailable ({html.escape(str(e)[:60])})"
    alts = "".join(f'<span class="pill">{ICONS[LABEL_LIST[i]]} {LABELS[LABEL_LIST[i]]["ar"]} {p[i]:.0%}</span>'
                   for i in order[1:3])
    speedup = TEACHER_MS / max(ms, 1)
    return f"""
<div class="card">
  <div class="muted">Tool to call</div>
  <div class="big">{ICONS[tool]} {LABELS[tool]['ar']} <span class="muted">({tool})</span></div>
  <div style="margin:12px 0 6px" class="muted">Router confidence {conf:.0%}</div>
  <div class="bar"><div style="width:{conf * 100:.0f}%"></div></div>
  <div style="margin-top:12px" class="muted">Runner-up: {alts}</div>
  <div style="margin-top:16px">{source}</div>
</div>
<div class="row" style="margin-top:14px">
  <div class="card" style="flex:1"><div class="muted">This router</div><div class="big accent">{ms:.0f} ms</div><div class="muted">laptop CPU · offline · $0</div></div>
  <div class="card" style="flex:1"><div class="muted">LLM router ({TEACHER_MODEL})</div><div class="big">{TEACHER_MS} ms</div><div class="muted">GPU + internet + paid call</div></div>
  <div class="card" style="flex:1"><div class="muted">Faster</div><div class="big accent">{speedup:.0f}×</div><div class="muted">same accuracy on our test set</div></div>
</div>"""


def stream_batch():
    """Route the whole held-out test set live, offline."""
    df = pd.read_csv("data/messages.csv")
    df = df[df["set"] == "test"].reset_index(drop=True)
    counts = {k: 0 for k in LABEL_LIST}
    correct, t0 = 0, time.perf_counter()
    for i in range(0, len(df), 16):
        chunk = df.iloc[i:i + 16]
        probs = predict(tok, model, list(chunk["text"]))
        for (_, row), p in zip(chunk.iterrows(), probs):
            tool = LABEL_LIST[p.argmax()]
            counts[tool] += 1
            correct += tool == row["gen_label"]
        n = min(i + 16, len(df))
        el = time.perf_counter() - t0
        last = html.escape(str(chunk["text"].iloc[-1])[:80])
        bars = "".join(
            f'<div style="display:flex;gap:10px;align-items:center;margin:6px 0"><div style="width:190px">{ICONS[k]} {LABELS[k]["ar"]}</div>'
            f'<div class="bar" style="flex:1"><div style="width:{100 * c / max(1, max(counts.values())):.0f}%"></div></div><div style="width:40px">{c}</div></div>'
            for k, c in counts.items())
        yield f"""
<div class="row">
  <div class="card" style="flex:1"><div class="muted">Requests routed</div><div class="big accent">{n}/{len(df)}</div></div>
  <div class="card" style="flex:1"><div class="muted">Time (offline, CPU)</div><div class="big">{el:.1f}s</div><div class="muted">{n / el:.0f} req/s</div></div>
  <div class="card" style="flex:1"><div class="muted">Same job with LLM routing</div><div class="big">{n * TEACHER_MS / 1000:.0f}s</div><div class="muted">{n} API calls</div></div>
  <div class="card" style="flex:1"><div class="muted">Agreement with intended tool</div><div class="big">{correct / n:.0%}</div></div>
</div>
<div class="card" style="margin-top:14px">{bars}<div class="muted" style="margin-top:10px;direction:rtl">آخر طلب: {last}</div></div>"""


with gr.Blocks(title="Offline Arabic Tool Router", **({} if GR_MAJOR >= 6 else {"css": CSS})) as demo:
    gr.HTML("""<div style="padding:8px 0 4px"><div style="font-size:30px;font-weight:700;color:#F6F3EE">
    Offline Arabic Tool Router <span style="color:#F2A65A">for AI assistants</span></div>
    <div class="muted">Decides which tool handles a request (search, image, code, translate…) in milliseconds on a CPU,
    instead of a full LLM call. Najdi · Hijazi · Eastern · MSA · Arabizi · mixed English.</div></div>""")
    with gr.Tab("Route a request"):
        inp = gr.Textbox(label="User request", lines=2, placeholder="اكتب طلبك هنا…")
        with gr.Row():
            ask = gr.Checkbox(label="Ask GPT when the router is unsure (needs internet)", value=False)
            thr = gr.Slider(0.3, 0.99, value=CASCADE_THRESHOLD, step=0.01, label="Unsure below")
        btn = gr.Button("Route", variant="primary")
        out = gr.HTML()
        gr.Examples([
            "وش اخبار مباراة الهلال اليوم؟",
            "سوي لي شعار لمقهى اسمه نجد بألوان ترابية",
            "اكتب لي سكربت بايثون يقرا ملف CSV ويطلع المتوسط",
            "ترجم لي هالايميل للانجليزي بشكل رسمي",
            "كم يطلع القسط لو اخذت 80 الف على 5 سنين بفايدة 4%",
            "لخص لي العقد اللي رفعته وطلع لي الشروط المهمة",
            "ذكرني بكرة الساعة 9 الصبح عندي اجتماع",
            "abi tarjim hal msg lel english",
            "سوي لي reminder للـ meeting حق الأحد",
        ], inp)
        btn.click(route_card, [inp, ask, thr], out)
        inp.submit(route_card, [inp, ask, thr], out)
    with gr.Tab("Live stream: route the whole test set"):
        go = gr.Button("▶ Route 319 requests offline", variant="primary")
        live = gr.HTML()
        go.click(stream_batch, None, live)

if __name__ == "__main__":
    demo.launch(**({"css": CSS} if GR_MAJOR >= 6 else {}))
