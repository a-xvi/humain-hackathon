# Technical decisions & trade-offs

Fill in the numbers from `experiments/RESULTS.md`. Each entry: what, why, evidence, trade-off.

## D1 · Classification with an encoder student, not a generative LLM
- **Why:** routing needs one of 8 labels, not generated text. An encoder (MARBERTv2, ~160M params) reads the message once and outputs 8 probabilities.
- **Evidence:** student = __% of teacher macro-F1 at __ ms/message on 1 CPU thread (`experiments/compute.csv`).
- **Trade-off:** cannot explain its answer or handle brand-new departments without retraining.

## D2 · MARBERTv2 as the student
- **Why:** pretrained on ~1B Arabic tweets, dialect-heavy (incl. Saudi). Customer messages look like tweets: short, dialectal, typos.
- **Alternative:** CAMeLBERT-mix (smaller vocab). Not tested for time.

## D3 · Active distillation (margin / margin + diversity) instead of random labeling
- **Why:** every teacher label costs an API call. Uncertainty sampling spends the budget on messages near decision boundaries.
- **Proxy honesty:** we can't know which messages the student gets wrong without labeling all of them, so uncertainty is the proxy.
- **Diversity:** pure margin can pick 20 near-identical confusing messages; clustering the top-3k candidates spreads the budget.
- **Evidence:** labels to reach 95% of the all-labels student: random __ vs margin __ vs margin_diverse __ (`savings` table), mean of 3 seeds.

## D4 · Simulated loop with cached teacher labels
- **Why:** comparing strategies fairly needs many runs (3 strategies × 3 seeds × 8 rounds). The teacher labels the pool once (cached); a strategy "pays" only for labels it requests. Standard pool-based active-learning evaluation.
- **Trade-off:** in production the loop would call the teacher live; cost is identical (labels requested).

## D5 · Synthetic data + human-verified test set
- **Why:** no public Saudi customer-service dataset with these departments; real customer data must not be used.
- **Bias controls:** generator model ≠ teacher model; test set generated separately; near-duplicates removed; test labels human-verified (all generator/teacher disagreements + 30 spot-checks).
- **Trade-off:** synthetic messages are cleaner than real ones → stress sets and perturbations exist to probe that.

## D6 · Teacher model choice
- Tested: gemini-3.8-flash (503 overloaded), gemini-2.5-flash (not available to new users), __ (chosen: fastest correct).
- Generator gemini-3.5-flash-lite labeled a delivery complaint as "complaints" in a probe → too weak to TEACH, fine to WRITE messages.

## D7 · Metrics
- Primary: **macro-F1** (each department counts equally; accuracy hides weak small classes). 95% bootstrap CI on every score.
- Compute: p50/p95 latency on 1 CPU thread, throughput, model size, RAM, teacher labels used.

## D8 · int8 dynamic quantization
- **Why:** Linear layers in 8-bit → smaller, usually faster on CPU, no retraining.
- **Evidence:** fp32 __ ms / int8 __ ms, macro-F1 __ vs __.

## D9 · Augmentation for robustness (zero extra teacher cost)
- Perturbed copies (rule-based Arabizi, typos, letter variants) of already-labeled messages.
- **Caveat:** the rule-based Arabizi test uses the same rule as the augmentation → optimistic. The LLM-written natural Arabizi stress set is the honest check.

## Did not work / not done
| Idea | Result | Why |
|---|---|---|
| | | |
