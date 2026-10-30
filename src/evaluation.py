"""Evaluation: fixed splits, metrics with confidence intervals, and the experiment log."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.labels import LABEL_LIST


def make_split(df, target, val_size=0.2, seed=42, path="val_idx.json"):
    """Fixed train/val split saved to disk, so every experiment uses the same rows."""
    if os.path.exists(path):
        with open(path) as f:
            val_idx = json.load(f)
        return df.drop(val_idx), df.loc[val_idx]
    train_idx, val_idx = train_test_split(df.index, test_size=val_size, random_state=seed, stratify=df[target])
    with open(path, "w") as f:
        json.dump(list(map(int, val_idx)), f)
    return df.loc[train_idx], df.loc[val_idx]


def scores(y_true, y_pred):
    """Macro-F1 (primary: every department counts equally) and accuracy."""
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABEL_LIST, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_pred, n=1000, seed=0):
    """95% confidence interval of macro-F1: is a difference real, or noise from a small test set?"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[i], y_pred[i], labels=LABEL_LIST, average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_class(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=LABEL_LIST, average=None, zero_division=0)
    return pd.Series(f1, index=LABEL_LIST, name="f1")


def confusion(y_true, y_pred):
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABEL_LIST), index=LABEL_LIST, columns=LABEL_LIST)


def evaluate(name, y_true, y_pred, extra=None, log=True):
    """Score + CI, print one line, and append it to experiments/runs.csv."""
    s = scores(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "n": len(y_true),
           "macro_f1": round(s["macro_f1"], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
           "accuracy": round(s["accuracy"], 4), **(extra or {})}
    print(f"{name:<40} macro-F1 {s['macro_f1']:.3f} [{lo:.3f}-{hi:.3f}]  acc {s['accuracy']:.3f}  n={len(y_true)}")
    if log:
        log_run(row)
    return row


def log_run(row, path="experiments/runs.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
