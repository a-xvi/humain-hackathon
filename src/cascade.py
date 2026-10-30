"""HYBRID MODE (cascade): the mini model answers instantly; only messages it is UNSURE about
go to the big model. One knob - the confidence threshold - trades cost for quality.

    confidence >= threshold  -> mini model's answer (offline, ~ms, free)
    confidence <  threshold  -> big model's answer  (API, ~seconds, paid)

Evaluated with zero extra API calls: the big model's answers on the test set are already cached.
"""
import numpy as np
import pandas as pd

from src.evaluation import scores
from src.labels import LABEL_LIST


def cascade_predict(student_probs, teacher_preds, threshold):
    conf = student_probs.max(1)
    mini = [LABEL_LIST[i] for i in student_probs.argmax(1)]
    escalate = conf < threshold
    preds = [t if e else m for m, t, e in zip(mini, teacher_preds, escalate)]
    return preds, escalate


def cascade_curve(student_probs, teacher_preds, gold, thresholds=None):
    """macro-F1 vs share of messages sent to the big model, for many thresholds."""
    thresholds = thresholds if thresholds is not None else [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]
    rows = []
    for th in thresholds:
        preds, esc = cascade_predict(student_probs, teacher_preds, th)
        rows.append({"threshold": th, "sent_to_big_model": round(float(esc.mean()), 3),
                     "macro_f1": round(scores(gold, preds)["macro_f1"], 4)})
    return pd.DataFrame(rows)


def cheapest_threshold(curve, target_f1):
    """Lowest share of big-model calls that still reaches `target_f1`."""
    ok = curve[curve["macro_f1"] >= target_f1]
    return None if ok.empty else ok.sort_values("sent_to_big_model").iloc[0].to_dict()


def plot_cascade(curve, mini_f1, teacher_f1, path="experiments/cascade.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = curve.sort_values("sent_to_big_model")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(100 * c["sent_to_big_model"], c["macro_f1"], marker="o", label="hybrid (mini + big when unsure)")
    ax.axhline(teacher_f1, ls="--", c="k", label=f"big model only {teacher_f1:.3f}")
    ax.axhline(mini_f1, ls=":", c="gray", label=f"mini model only {mini_f1:.3f}")
    ax.set_xlabel("% of messages sent to the big model (cost)")
    ax.set_ylabel("macro-F1 (human-verified test)")
    ax.set_title("Hybrid mode: quality vs big-model usage")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path
