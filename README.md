# Offline Arabic Tool Router for AI Assistants, with Active Distillation

**Challenge 2 · Deliver AI with Less Compute** — Future AI Builders Hackathon (HUMAIN)

> *"AI assistants waste a full LLM call deciding which tool to use. I moved that decision to a laptop CPU."*

Before an AI assistant answers, it must decide **which tool** handles the request: web search, image generation, code, translation, math, a document, the calendar, or plain chat. That routing step is usually a full LLM call — GPU, internet and money on every request. We distill the LLM's routing decision into a small Arabic model (**student**) that makes it **offline on a CPU in milliseconds**, across Saudi dialects, Arabizi and mixed English. The twist: the student chooses which requests are worth paying the teacher for (**active distillation**), so it's cheaper to **run** *and* cheaper to **train**.

## Results
_Filled from `experiments/RESULTS.md`._

| | Teacher (LLM API) | Student (ours, CPU, offline) |
|---|---|---|
| macro-F1 (human-verified test) | | |
| Latency per message | | |
| Needs GPU / internet | yes | **no** |
| Teacher labels used for training | — | __ / __ pool requests |

![learning curves](experiments/curves_bert.png)

## How it works
1. **Data** — a generator LLM writes realistic assistant requests in Najdi, Hijazi, Eastern and MSA (pool + held-out test), plus stress sets: unseen dialects (Egyptian, Levantine), natural Arabizi, Arabic–English code-switching.
2. **Teacher** — GPT-4.1-mini labels requests (forced choice of 8 tools, temperature 0, batched, cached).
3. **Human verification** — every test message where the generator and the teacher disagree is checked by hand (+ random spot-checks).
4. **Active distillation** — seed 80 labels → train student → pick the 80 messages it's least sure about (margin, + diversity clustering) → teacher labels only those → repeat to a 640-label budget. Compared against random selection, 3 seeds each.
5. **Student** — MARBERTv2 fine-tuned for classification, int8-quantized for CPU.
6. **Robustness** — stress sets + perturbations (typos, letter variants, Arabizi, truncation, emojis), then zero-cost augmentation to fix weak spots.

## Reproduce
**Colab (T4 GPU):** open `notebooks/hackathon.ipynb`, add `GEMINI_API_KEY` (and `GH_TOKEN`) to Colab Secrets, run top to bottom. `MOCK = True` runs everything on fake data in ~2 minutes.

**Laptop (offline demo):**
```bash
pip install -r requirements-laptop.txt
# put the trained model in models/student_final (from Google Drive)
python bench_laptop.py   # real CPU numbers
python app.py            # turn Wi-Fi off, open http://127.0.0.1:7860
```

## Repository
```
notebooks/hackathon.ipynb   the full pipeline, step by step
src/labels.py               8 tools (single source of truth)
src/llm.py                  Gemini wrapper: pacing, retries, JSON output
src/generate.py             synthetic data, dedup/leak removal, human review
src/teacher.py              batched, cached teacher labels
src/students.py             TF-IDF baseline + MARBERTv2 student
src/active.py               selection strategies + active distillation loop
src/evaluation.py           macro-F1, bootstrap CI, run log
src/robustness.py           perturbations + augmentation
src/deploy.py               int8 quantization + CPU benchmark
src/report.py               curves, savings table, RESULTS.md
app.py · bench_laptop.py    offline demo (single request + live stream of the test set) + laptop benchmark
docs/DECISIONS.md           technical decisions & trade-offs
```

## Limitations
Synthetic data (mitigated: separate generator, human-verified test, stress sets) · the student inherits teacher mistakes · one tool per request · limited dialect coverage.

v1 of this project routed customer-service messages (archived in `archive/v1_customer_routing`); the same pipeline was re-targeted to tool routing for AI assistants.
**Next:** optional online cascade (escalate low-confidence messages to ALLaM when connected), confidence calibration, urgency label, real anonymized data.
