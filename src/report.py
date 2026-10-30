"""Charts and the RESULTS.md summary."""
import os

import pandas as pd

from src.active import labels_to_reach, summarize_curves


def plot_curves(curves, teacher_f1=None, full_pool_f1=None, path="experiments/learning_curves.png", title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summarize_curves(curves)
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat, g in s.groupby("strategy"):
        ax.plot(g["labels_used"], g["mean"], marker="o", label=strat)
        ax.fill_between(g["labels_used"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), alpha=0.15)
    if teacher_f1 is not None:
        ax.axhline(teacher_f1, ls="--", c="k", label=f"teacher (LLM) {teacher_f1:.3f}")
    if full_pool_f1 is not None:
        ax.axhline(full_pool_f1, ls=":", c="gray", label=f"student, ALL pool labels {full_pool_f1:.3f}")
    ax.set_xlabel("teacher labels used (training cost)")
    ax.set_ylabel("macro-F1 on human-verified test set")
    ax.set_title(title or "Active distillation: quality vs teacher labels (mean ± std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    return path


def savings_table(curves, reference_f1, fractions=(0.90, 0.95, 0.98)):
    """Teacher labels each strategy needs to reach X% of a reference score, and savings vs random."""
    rows = []
    for f in fractions:
        need = labels_to_reach(curves, f * reference_f1)
        row = {"target": f"{int(f * 100)}% of reference ({f * reference_f1:.3f})", **need.to_dict()}
        if "random" in need and pd.notna(need["random"]):
            for k, v in need.items():
                if k != "random" and pd.notna(v):
                    row[f"{k}_saving_vs_random"] = f"{100 * (1 - v / need['random']):.0f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def write_results(sections, path="experiments/RESULTS.md"):
    """sections: list of (title, DataFrame or str)."""
    out = ["# Results\n"]
    for title, body in sections:
        out.append(f"## {title}\n")
        out.append(body.to_markdown(index=False) if isinstance(body, pd.DataFrame) else str(body))
        out.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path
