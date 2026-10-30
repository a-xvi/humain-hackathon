"""Offline demo: Arabic customer-message routing on a laptop CPU. No GPU, no internet.

Run on your laptop (after copying models/student_final from Colab):
    pip install -r requirements-laptop.txt
    python app.py
Then turn Wi-Fi OFF and open http://127.0.0.1:7860
"""
import time

import gradio as gr

from src.deploy import load_student, route
from src.labels import LABELS

MODEL_PATH = "models/student_final"
tok, model = load_student(MODEL_PATH, quantize=True)

EXAMPLES = [
    "طلبي متأخر من أسبوع ووين المندوب؟",
    "انخصم المبلغ مرتين من بطاقتي ابي حقي",
    "ابغى ارجع الجزمة المقاس صغير",
    "ما اقدر ادخل حسابي الكود ما يوصل",
    "التطبيق يطفي كل ما افتح السلة",
    "abi a3rf mta y9l 6albi",
    "الـ order حقي delayed وش السالفة",
]


def classify(text):
    if not text.strip():
        return {}, ""
    t0 = time.perf_counter()
    top = route(tok, model, text)
    ms = (time.perf_counter() - t0) * 1000
    result = {f"{LABELS[k]['ar']} ({k})": p for k, p in top}
    return result, f"⚡ {ms:.0f} ms on CPU · offline · int8"


with gr.Blocks(title="Offline Arabic Customer Routing") as demo:
    gr.Markdown("## 📨 Arabic customer-message routing — runs offline on a laptop CPU")
    inp = gr.Textbox(label="Customer message (any dialect, Arabizi, mixed English)", lines=3, rtl=True)
    btn = gr.Button("Route", variant="primary")
    out = gr.Label(num_top_classes=3, label="Department")
    speed = gr.Markdown()
    gr.Examples(EXAMPLES, inp)
    btn.click(classify, inp, [out, speed])
    inp.submit(classify, inp, [out, speed])

if __name__ == "__main__":
    demo.launch()
