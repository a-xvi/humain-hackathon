"""ACTIVE DISTILLATION: the student decides which messages are worth paying the teacher for.

Loop (pool-based active learning):
  1. teacher labels a small random SEED set
  2. train the student on everything labeled so far
  3. student scores every still-unlabeled pool message
  4. pick the STEP_SIZE messages it is least sure about -> teacher labels only those
  5. repeat until the BUDGET of teacher labels is spent

We can't know which messages the student gets WRONG without labeling everything,
so uncertainty is the proxy:
- random          : baseline - pick at random (what you'd do without active learning)
- margin          : smallest gap between the top-2 class probabilities (student torn between two departments)
- entropy         : most spread-out probability distribution
- margin_diverse  : take the 3x most uncertain, cluster them, keep the most uncertain per cluster
                    -> avoids spending budget on 20 near-identical confusing messages

Simulation note (be transparent about this): to compare strategies fairly and cheaply, the teacher
labels the whole pool ONCE (cached). The loop only "reveals" a label when a strategy requests it, and
cost = number of labels requested. This is the standard way active learning is evaluated, and it's
exactly what a live deployment would pay.
"""
import numpy as np
import pandas as pd

from src.evaluation import scores


def margin(probs):
    p = np.sort(probs, axis=1)
    return p[:, -1] - p[:, -2]


def entropy(probs):
    return -(probs * np.log(probs + 1e-12)).sum(1)


def select(strategy, probs, k, rng, features=None):
    """Return positions (into `probs`) of the k messages to send to the teacher."""
    n = len(probs)
    k = min(k, n)
    if strategy == "random":
        return rng.choice(n, size=k, replace=False)
    if strategy == "margin":
        return np.argsort(margin(probs))[:k]
    if strategy == "entropy":
        return np.argsort(-entropy(probs))[:k]
    if strategy == "margin_diverse":
        from sklearn.cluster import KMeans
        cand = np.argsort(margin(probs))[:min(n, 3 * k)]
        if len(cand) <= k or features is None:
            return cand[:k]
        km = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(1_000_000))).fit(features[cand])
        m = margin(probs[cand])
        picked = [cand[np.where(km.labels_ == c)[0][np.argmin(m[km.labels_ == c])]] for c in range(k)]
        return np.array(picked)
    raise ValueError(f"unknown strategy {strategy}")


def run_active(pool_texts, oracle_labels, test_texts, test_labels, make_student, strategy,
               seed_size, step, budget, seed, features=None, verbose=True):
    """One active-distillation run. Returns one row per round (labels used -> test scores).

    oracle_labels: teacher labels for every pool text (revealed only when requested).
    make_student : function(seed) -> fresh student with fit / predict_proba / predict.
    """
    rng = np.random.default_rng(seed)
    pool_texts = list(pool_texts)
    oracle_labels = list(oracle_labels)
    labeled = list(rng.choice(len(pool_texts), size=min(seed_size, len(pool_texts)), replace=False))
    rows = []
    while True:
        student = make_student(seed).fit([pool_texts[i] for i in labeled], [oracle_labels[i] for i in labeled])
        s = scores(test_labels, student.predict(test_texts))
        rows.append({"strategy": strategy, "seed": seed, "round": len(rows), "labels_used": len(labeled), **s})
        if verbose:
            print(f"  {strategy:<15} seed {seed} | labels {len(labeled):>5} | macro-F1 {s['macro_f1']:.3f}")
        if len(labeled) >= budget:
            break
        unl = np.setdiff1d(np.arange(len(pool_texts)), labeled)
        if len(unl) == 0:
            break
        probs = student.predict_proba([pool_texts[i] for i in unl])
        feats = features[unl] if features is not None else None
        picked = select(strategy, probs, min(step, budget - len(labeled)), rng, feats)
        labeled += list(unl[picked])
    return rows, labeled, student


def run_curves(pool_texts, oracle_labels, test_texts, test_labels, make_student, strategies, seeds,
               seed_size, step, budget, features=None, out=None):
    """All strategies x all seeds -> one DataFrame of learning curves."""
    import os
    all_rows = []
    if out and os.path.exists(out):  # resume: keep finished (strategy, seed) runs
        prev = pd.read_csv(out)
        done = prev.groupby(["strategy", "seed"])["labels_used"].max()
        keep = [k for k, v in done.items() if v >= min(budget, len(pool_texts))]
        all_rows = prev[prev.set_index(["strategy", "seed"]).index.isin(keep)].to_dict("records")
        if keep:
            print(f"resuming: {len(keep)} finished runs loaded from {out}")
    finished = {(r["strategy"], r["seed"]) for r in all_rows}
    for strategy in strategies:
        for seed in seeds:
            if (strategy, seed) in finished:
                continue
            rows, _, _ = run_active(pool_texts, oracle_labels, test_texts, test_labels, make_student,
                                    strategy, seed_size, step, budget, seed, features)
            all_rows += rows
            if out:
                pd.DataFrame(all_rows).to_csv(out, index=False)
    return pd.DataFrame(all_rows)


def summarize_curves(curves):
    """Mean and std over seeds, per strategy and budget."""
    return (curves.groupby(["strategy", "labels_used"])["macro_f1"]
            .agg(["mean", "std", "count"]).reset_index())


def labels_to_reach(curves, target):
    """For each strategy: fewest teacher labels at which the MEAN macro-F1 reaches `target` (NaN if never)."""
    s = summarize_curves(curves)
    out = {}
    for strat, g in s.groupby("strategy"):
        hit = g[g["mean"] >= target]
        out[strat] = int(hit["labels_used"].min()) if len(hit) else np.nan
    return pd.Series(out, name=f"labels_to_reach_{target:.3f}")
