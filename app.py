"""Demo: Arabic customer-message routing with three modes.

  ⚡ Mini model      - offline on the laptop CPU, instant, free
  🔀 Hybrid          - mini model answers; only UNSURE messages go to the big model
  🧠 Big model only  - every message goes to the LLM API (slow, paid, needs internet)

Run on your laptop (after copying models/student_final from Colab):
    pip install -r requirements-laptop.txt
    python app.py                       # mini mode works with Wi-Fi OFF
Hybrid / big modes need internet and:   set GEMINI_API_KEY=...   (Windows)  |  export GEMINI_API_KEY=... (Mac/Linux)
"""
import os
import time

import gradio as gr

from src.config import CASCADE_THRESHOLD, TEACHER_MODEL
from src.deploy import load_student, route
from src.labels import LABELS

MODEL_PATH = "models/student_final"
tok, model = load_student(MODEL_PATH, quantize=True)

_big = None


def big_model():
    """Lazy-load the big model client (only when a mode needs it)."""
    global _big
    if _big is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY and connect to the internet to use the big model.")
        from src.llm import GeminiClient
        from src.teacher import Teacher
        _big = Teacher(GeminiClient(key, min_interval=0, max_retries=2), model=TEACHER_MODEL,
                       cache_path="data/demo_cache.json")
    return _big


def name(k):
    return f"{LABELS[k]['ar']} ({k})"


def classify(text, mode, threshold):
    if not text.strip():
        return {}, ""
    t0 = time.perf_counter()
    top = route(tok, model, text)
    mini_ms = (time.perf_counter() - t0) * 1000
    conf = top[0][1]
    use_big = mode.startswith("🧠") or (mode.startswith("🔀") and conf < threshold)
    if not use_big:
        why = "" if mode.startswith("⚡") else f" · confident ({conf:.0%} ≥ {threshold:.0%}) → no API call"
        return {name(k): p for k, p in top}, f"⚡ mini model · {mini_ms:.0f} ms · offline · free{why}"
    try:
        t1 = time.perf_counter()
        label = big_model().label(text)
        big_ms = (time.perf_counter() - t1) * 1000
    except Exception as e:
        return {name(k): p for k, p in top}, f"⚠️ big model unavailable ({str(e)[:80]}) → mini model answer shown"
    why = f"mini model unsure ({conf:.0%} < {threshold:.0%}) → " if mode.startswith("🔀") else ""
    return {name(label): 1.0}, f"🧠 {why}big model · {big_ms:.0f} ms · API call"


with gr.Blocks(title="Arabic Customer Routing") as demo:
    gr.Markdown("## 📨 Arabic customer-message routing — mini model on CPU, big model only when needed")
    inp = gr.Textbox(label="Customer message (any dialect, Arabizi, mixed English)", lines=3, rtl=True)
    with gr.Row():
        mode = gr.Radio(["⚡ Mini model (offline, instant)", "🔀 Hybrid (mini, big model when unsure)",
                         "🧠 Big model only"], value="⚡ Mini model (offline, instant)", label="Mode")
        thr = gr.Slider(0.3, 0.99, value=CASCADE_THRESHOLD, step=0.01, label="Hybrid: confidence threshold")
    btn = gr.Button("Route", variant="primary")
    out = gr.Label(num_top_classes=3, label="Department")
    info = gr.Markdown()
    gr.Examples([
        "طلبي متأخر من أسبوع ووين المندوب؟",
        "انخصم المبلغ مرتين من بطاقتي ابي حقي",
        "ابغى ارجع الجزمة المقاس صغير",
        "ما اقدر ادخل حسابي الكود ما يوصل",
        "التطبيق يطفي كل ما افتح السلة",
        "abi a3rf mta y9l 6albi",
        "الـ order حقي delayed وش السالفة",
    ], inp)
    btn.click(classify, [inp, mode, thr], [out, info])
    inp.submit(classify, [inp, mode, thr], [out, info])

if __name__ == "__main__":
    demo.launch()
