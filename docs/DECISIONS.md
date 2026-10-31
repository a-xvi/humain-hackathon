# Technical decisions and trade-offs

Each entry gives the decision, why it was made, the evidence and the trade-off. Numbers come from [`experiments/RESULTS.md`](../experiments/RESULTS.md) and are measured on the 320-request human-checked test set.

## D1 · A classifier, not a generative model
- **Why:** routing outputs one of 8 tools, not text. An encoder reads the request once and returns 8 probabilities. There's no decoding loop and no prompt tokens.
- **Evidence:** macro-F1 0.944 vs GPT-4.1-mini 0.918 (the confidence intervals overlap, so they match). 101 ms vs 707 ms p50 on 1 CPU thread.
- **Trade-off:** adding a new tool means retraining, and the router can't explain its choice.

## D2 · MARBERTv2 as the router
- **Why:** it was pretrained on about 1B Arabic tweets and is heavy on dialect, including Saudi. Assistant requests look like tweets: short, dialectal and full of typos.
- **Evidence it was worth it over TF-IDF:** they're close on clean text (0.944 vs 0.931), but MARBERTv2 is much stronger on unseen dialects (0.949 vs 0.880) and Arabizi (0.621 vs 0.087).
- **Not tested (time):** CAMeLBERT-mix, smaller multilingual encoders.

## D3 · Active distillation instead of labeling everything
- **Why:** every teacher label is an API call. Margin sampling spends the budget on requests where the router's top two tools are closest. Adding k-means diversity stops it from picking 80 near-identical confusing requests.
- **Proxy honesty:** we can't know what the router gets wrong without labeling everything, so uncertainty stands in for it.
- **Evidence (mean of 3 seeds):** to reach 95% of the reference score (0.888), random needed 480 labels, margin 400 and margin + diversity 320. Margin at 480 labels (0.940) beats random at 640 (0.931). At 640 labels the spread across seeds is ≤ 0.012.
- **Trade-off:** the gains are largest at small budgets and shrink as the budget grows.

## D4 · Simulated loop with cached teacher labels
- **Why:** a fair comparison needs 3 strategies × 3 seeds × 8 rounds. The teacher labels the pool once and caches the labels. Each strategy "pays" only for the labels it requests. This is standard pool-based active-learning evaluation.
- **Trade-off:** in production the loop would call the teacher live. The cost is the same, because it's counted in labels requested.

## D5 · Synthetic data and a human-checked test set
- **Why:** there's no public dataset of Arabic assistant requests labeled by tool.
- **Bias controls:**
  - The generator (Gemini 3.5 Flash-Lite) is a different model from the teacher (GPT-4.1-mini).
  - The test set is generated separately from the pool.
  - Near-duplicates and leaks are removed.
- **Human check:** the generator's intended tool and the teacher's label agreed 87.6% of the time. All 57 disagreements were reviewed by hand, plus 30 random spot checks.
- **Trade-off:** synthetic requests are cleaner than real ones, which is why the stress sets and perturbations exist.

## D6 · Teacher model
- Gemini models failed on availability (503 overload, a 404 for new users, free-tier 429 limits). GPT-4.1-mini was used with strict JSON-schema output, batches of 100, temperature 0 and a cache.
- The generator stays a separate model, so the teacher isn't grading its own writing.

## D7 · Metrics
- **Primary:** macro-F1, so every tool counts equally. Accuracy hides weak small classes. Every main score has a 95% bootstrap confidence interval.
- **Compute:** p50 and p95 latency on 1 CPU thread, throughput, model size, RAM, and teacher labels used.
- **Cost per successful task:** latency ÷ accuracy gives 107 ms per correct decision for the router vs 770 ms for GPT.

## D8 · int8 quantization: tested and rejected
- **Tried:** dynamic int8 on all Linear layers, plus two selective variants.
- **Evidence:**

  | Variant | p50 latency | Macro-F1 |
  |---|---|---|
  | fp32 | 100.7 ms | 0.944 |
  | int8 (all Linear layers) | 34.8 ms | 0.842 |
  | Selective variant 1 | not recorded | 0.842 |
  | Selective variant 2 | not recorded | 0.884 |

- **Decision:** ship fp32. Cheap only counts if the answer is right.
- **Next:** distill to a smaller encoder, or use quantization-aware training.

## D9 · Augmentation for robustness, with no extra GPT calls
- **Method:** perturbed copies of requests that already had labels: rule-based Arabizi, typos and letter variants.
- **Evidence:**
  - Natural Arabizi went from 0.156 to 0.621.
  - Clean text was unchanged (0.942 → 0.944).
  - Mixed English dropped from 0.836 to 0.775. That's a real trade-off, and it's reported here.
- **Caveat:** the rule-based Arabizi perturbation test uses the same rule as the augmentation, so its score is optimistic. The LLM-written natural Arabizi stress set is the honest check.

## D10 · Hybrid mode (model choice per task)
- **How:** if the router's confidence is below a threshold, the request goes to GPT.
- **Evidence:**

  | Threshold | Requests sent to GPT | Macro-F1 |
  |---|---|---|
  | 0.5 | 2.5% | 0.953 |
  | 0.8 | 9.7% | 0.975 |
  | (GPT only) | 100% | 0.918 |

- **Why it beats both:** the two models make different mistakes. The router keeps what it's sure of, and GPT catches the rest.

## D11 · Demo without a web framework
- **Why:** it must run with Wi-Fi off on any laptop. It's Python's built-in HTTP server plus one HTML file, with no CDN, no web fonts and no Gradio.

## What didn't work
| Idea | Result | Why |
|---|---|---|
| int8 dynamic quantization | −0.10 macro-F1 | The fine-tuned classifier is sensitive to weight precision |
| Gemini as the teacher | 503 / 404 / 429 errors | Availability and free-tier quota limits |
| v1: customer-service routing | Replaced | Weaker story for compute savings; the pipeline was re-targeted to tool routing |
| Augmentation for every weakness | Hurt mixed English | Likely: Arabizi copies shifted how Latin-script words are read |
