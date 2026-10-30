"""Run ON YOUR LAPTOP (no GPU, Wi-Fi off) to measure the real offline numbers for the slides.

    python bench_laptop.py
"""
import platform

import numpy as np
import pandas as pd

from src.deploy import benchmark, load_student, predict
from src.evaluation import scores
from src.labels import LABEL_LIST

MODEL = "models/student_final"
df = pd.read_csv("data/messages.csv")
test = df[df["set"] == "test"]
texts, gold = list(test["text"]), list(test["gen_label"])  # gen_label; final gold scores come from the notebook

print(f"CPU: {platform.processor() or platform.machine()} | {platform.system()} {platform.release()}")
for q in [False, True]:
    tok, m = load_student(MODEL, quantize=q, threads=1)
    probs = np.concatenate([predict(tok, m, texts[i:i + 64]) for i in range(0, len(texts), 64)])
    f1 = scores(gold, [LABEL_LIST[i] for i in probs.argmax(1)])["macro_f1"]
    print("int8" if q else "fp32", {"macro_f1_vs_gen_label": round(f1, 3), **benchmark(tok, m, texts)})
