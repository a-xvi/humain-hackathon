# Sikkah · سكة
**An offline Arabic tool router for AI assistants**

**Challenge 2 · Deliver AI with Less Compute** · Future AI Builders Hackathon (Misk × HUMAIN)

Before an AI assistant does anything, it has to decide **which tool** handles the request: web search, image generation, code, translation, math, a document, the calendar, or plain chat. That decision is usually a full LLM call, which means a GPU, internet access and a paid API call on every request.

This project moves that decision to **a small Arabic model that runs offline on a laptop CPU**. It covers Saudi dialects, Arabizi and mixed English. GPT-4.1-mini teaches the small model through **active distillation**: the small model picks which requests are worth paying GPT to label. That makes it cheaper to **train** as well as to **run**.

## Results
All results are on 320 held-out requests. The test labels were checked by a person (all 57 cases where the two LLMs disagreed, plus 30 random spot checks).

| Router | Macro-F1 [95% CI] | p50 latency (1 CPU thread) | GPT labels to train | Offline |
|---|---|---|---|---|
| GPT-4.1-mini (API) | 0.918 [0.887–0.949] | 707 ms | — | No |
| TF-IDF + logistic regression | 0.931 | < 1 ms | 1,920 | Yes |
| MARBERTv2, all labels | 0.935 | 101 ms | 1,920 | Yes |
| **MARBERTv2, active distillation (shipped)** | **0.944 [0.918–0.968]** | **101 ms** | **640** | **Yes** |
| **Hybrid: router + GPT only when unsure (< 0.80)** | **0.975** | 9.7% of requests call GPT | 640 | Mostly |

- **Matches GPT and is 7× faster** on clean requests, with no GPU, no internet and no API cost per request.
- **Cheaper to train:** margin sampling at 480 labels beats random sampling at 640. Reaching 95% of the reference score took 33% fewer GPT labels with margin + diversity.
- **Model choice per task (hybrid mode):** sending the 9.7% least-confident requests to GPT gives **0.975**, which is higher than either model alone.
- **Robustness (macro-F1, shipped router / GPT):**
  - Unseen dialects (Egyptian, Levantine): **0.949** / 0.938
  - Typos: 0.926
  - Arabizi: 0.621 / 0.802. This was 0.156 before zero-cost augmentation.
  - Mixed Arabic–English: 0.775 / 0.938
- **int8 rejected:** it was 3× faster but lost about 0.10 macro-F1, so the router ships in fp32 (652 MB).

![learning curves](experiments/curves_bert.png)

Full tables: [`experiments/RESULTS.md`](experiments/RESULTS.md) · Decisions and trade-offs: [`docs/DECISIONS.md`](docs/DECISIONS.md)

## How it works
1. **Data.** A separate generator LLM writes assistant requests in Najdi, Hijazi, Eastern and MSA: a 1,920-request pool and a separate 320-request test set. It also writes three 80-request stress sets: unseen dialects, natural Arabizi, and Arabic–English code-switching.
2. **Teacher.** GPT-4.1-mini labels requests with a forced choice of 8 tools (temperature 0, batched, cached). It is a different model from the generator, to reduce bias.
3. **Human check.** A person reviews every test request where the generator and the teacher disagree, plus random spot checks.
4. **Active distillation.**
   - The teacher labels 80 seed requests, and the router is trained on them.
   - The router flags the 80 requests where its top two tools are closest (margin), spread out with k-means for diversity.
   - The teacher labels only those, and the loop repeats up to a 640-label budget.
   - This is compared against random selection, with 3 seeds per strategy.
5. **Router.** MARBERTv2, an Arabic BERT pretrained on dialect-heavy tweets, fine-tuned as an 8-class classifier.
6. **Robustness.** Stress sets and perturbations (typos, letter variants, Arabizi, truncation, emoji). Weak spots are then fixed by augmentation, with no extra GPT calls.
7. **Hybrid mode.** If the router's confidence is below a threshold, the request goes to GPT. One number trades cost against quality.

## Run the demo (offline)
**1. Get the trained model** (≈600 MB, not stored in Git): download [sikkah_model.zip](https://drive.google.com/file/d/1W0MZWwMhsqXusa47nuv6NXFG3U0sxXQy/view?usp=sharing) and unzip it into `models/`, so you have `models/student_final/config.json`.

**2. Run:**
```bash
pip install -r requirements-laptop.txt
python app.py            # opens http://127.0.0.1:7860; turn Wi-Fi off first
python bench_laptop.py   # latency on your own CPU
```
The demo is a single page served by Python's built-in HTTP server, with no web framework and nothing downloaded at runtime. It has two views:
- **Route requests:** type any dialect and see the tool, the confidence, the runner-ups and the latency.
- **Benchmark:** routes all 320 test requests live and shows macro-F1 against the human labels.

Hybrid mode needs internet and `OPENAI_API_KEY`.

## Reproduce training
Open `notebooks/hackathon.ipynb` in Colab (T4 GPU) and add `OPENAI_API_KEY` and `GEMINI_API_KEY` to Colab Secrets. Set `MOCK = True` to run the whole pipeline on fake data in about 2 minutes, with no keys.

## Repository
```
notebooks/hackathon.ipynb   the full pipeline, step by step
src/labels.py               the 8 tools (single source of truth)
src/llm.py                  Gemini and OpenAI clients: pacing, retries, JSON output
src/generate.py             synthetic data, dedup and leak removal, human review
src/teacher.py              batched, cached teacher labels
src/students.py             TF-IDF baseline + MARBERTv2 router
src/active.py               selection strategies + active distillation loop
src/evaluation.py           macro-F1, bootstrap CI, run log
src/robustness.py           perturbations + augmentation
src/cascade.py              hybrid mode (escalate when unsure)
src/deploy.py               CPU loading + benchmark
src/report.py               curves, savings table, RESULTS.md
app.py · web/index.html     offline demo
bench_laptop.py             laptop benchmark
docs/DECISIONS.md           technical decisions and trade-offs
```

## Limitations
- The data is synthetic. This is mitigated by a separate generator model, a human-checked test set and the stress sets.
- The router inherits GPT's mistakes.
- It handles one tool per request.
- Arabizi and mixed English still trail GPT. Hybrid mode covers them.
- The router ships in fp32 at 652 MB, because int8 lost accuracy.

**Next:** learn from real use (every request hybrid mode sends to GPT becomes a free label on a hard case; retrain weekly so the router needs GPT less); multi-tool requests; distilling to a smaller encoder that survives int8; ALLaM as the online teacher.

The same pipeline was first built for Arabic customer-service routing (v1) and then re-targeted to tool routing.
