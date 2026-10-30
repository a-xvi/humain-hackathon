"""Builds notebooks/hackathon.ipynb (the Colab notebook you run top to bottom)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Offline Arabic Tool Router for AI assistants — Active Distillation
**Challenge 2 · Deliver AI with Less Compute**

AI assistants spend a full LLM call just to decide which tool handles a request (search, image, code, translate…).
A large LLM (teacher) labels Arabic requests with the right tool; a small Arabic model (student) learns that decision and makes it **offline on a CPU in milliseconds**.
Twist: the student picks which requests are worth paying the teacher for (**active distillation**) → cheaper to *run* and cheaper to *train*.

Runtime → Change runtime type → **T4 GPU**. Run cells top to bottom. Every step is resumable: if something fails, fix it and re-run that cell."""))

C.append(md("## 0 · Setup"))
C.append(code("""import os
from google.colab import userdata

REPO = "/content/humain-hackathon"
if not os.path.exists(REPO):
    token = userdata.get("GH_TOKEN")
    !git clone -q https://{token}@github.com/a-xvi/humain-hackathon.git {REPO}
%cd {REPO}
!git pull -q
!pip -q install google-genai openai"""))

C.append(code("""# Archive the first version (customer-service routing) so this run starts clean
import shutil, glob
os.makedirs("archive/v1_customer_routing", exist_ok=True)
for f in ["data/messages.csv", "data/test_review.csv", "experiments"]:
    if os.path.exists(f) and not os.path.exists(f"archive/v1_customer_routing/{os.path.basename(f)}"):
        shutil.move(f, "archive/v1_customer_routing/")
os.makedirs("experiments", exist_ok=True)
print(os.listdir("archive/v1_customer_routing"))"""))

C.append(code("""MOCK = False   # True = whole pipeline on fake data in ~2 min, no API calls (to check everything runs)

from src import config
from src.labels import LABEL_LIST
STUDENT_MODEL = config.STUDENT_MODEL
llm = None
if not MOCK:
    from src.llm import GeminiClient
    llm = GeminiClient(userdata.get("GEMINI_API_KEY"))"""))

C.append(md("""## 1 · Teacher model (OpenAI)
Probe on 3 requests. Expected: `web_search, code_assistant, scheduling`."""))
C.append(code("""import time
from src.teacher import Teacher
from src.llm import OpenAIClient, client_for
if not MOCK:
    oa = OpenAIClient(userdata.get("OPENAI_API_KEY"))
    probe = ["وش اخبار مباراة الهلال اليوم؟", "اكتب لي كود بايثون يرتب قائمة", "ذكرني بكرة الساعة 9"]
    for m in ["gpt-4.1-mini", "gpt-4o-mini"]:
        t0 = time.time()
        try:
            print(m, f"{time.time() - t0:.1f}s", Teacher(oa, model=m, cache_path="/tmp/probe.json").label_many(probe, verbose=False))
        except Exception as e:
            print(m, "ERROR", str(e)[:90])"""))
C.append(code("""TEACHER_MODEL = "gpt-4.1-mini"             # teacher (OpenAI)
GENERATOR_MODEL = "gemini-3.5-flash-lite"  # writes the requests (different company = less bias)
teacher_llm = llm if MOCK else oa"""))

C.append(md("""## 2 · Generate the requests (≈160 API calls, ~15–20 min on the free tier)
Pool (training), test (held-out), and stress sets (unseen dialects, Arabizi, code-switching). Re-run after any error: finished calls are cached.\nIf Gemini says 429 (quota), change the call to `generate_all(oa, model=\"gpt-4o-mini\")`."""))
C.append(code("""from src.generate import generate_all, make_mock_messages
if MOCK:
    df = make_mock_messages()
    os.makedirs("data", exist_ok=True); df.to_csv("data/messages.csv", index=False)
else:
    df = generate_all(llm, model=GENERATOR_MODEL)
df.groupby(["set"]).size()"""))
C.append(code("""df.sample(10, random_state=1)[["set", "dialect", "gen_label", "text"]]"""))

C.append(md("## 3 · Teacher labels every message (≈25 calls, 100 per call, cached)"))
C.append(code("""from src.teacher import MockTeacher
teacher = MockTeacher(dict(zip(df.text, df.gen_label))) if MOCK else Teacher(teacher_llm, model=TEACHER_MODEL)
teacher_labels = dict(zip(df.text, teacher.label_many(list(df.text), batch_size=100)))
agree = (df.text.map(teacher_labels) == df.gen_label).mean()
print(f"teacher agrees with the generator's intended label on {agree:.1%} of messages")"""))

C.append(md("""## 4 · Human verification of the evaluation sets (you, ~15 min)
Only messages where the **generator and teacher disagree** + 30 random spot-checks.
For each one: **Enter** = generator is right · **t** = teacher is right · **d** = drop (nonsense/ambiguous) · or the number 0–7 · **q** = stop (re-run to continue)."""))
C.append(code("""from src.generate import make_review_file, review_interactively
if not os.path.exists("data/test_review.csv"):
    make_review_file(df, teacher_labels)
if not MOCK:
    review_interactively()"""))
C.append(code("""from src.generate import apply_review
df = apply_review(df)
pool = df[df.set == "pool"].reset_index(drop=True)
test = df[df.set == "test"].reset_index(drop=True)
stress = {s: df[df.set == s].reset_index(drop=True) for s in sorted(df.set.unique()) if s.startswith("stress")}
pool_texts, pool_teacher = list(pool.text), [teacher_labels[t] for t in pool.text]
test_texts, test_gold = list(test.text), list(test.gold)
print("pool", len(pool), "| test", len(test), "|", {k: len(v) for k, v in stress.items()})"""))

C.append(md("## 5 · Baselines: majority class · teacher (ceiling) · TF-IDF + logistic regression"))
C.append(code("""from collections import Counter
from src.evaluation import evaluate, per_class, confusion
from src.students import TfidfStudent
majority = Counter(pool_teacher).most_common(1)[0][0]
r_majority = evaluate("baseline_majority", test_gold, [majority] * len(test_gold))
r_teacher = evaluate(f"teacher_{TEACHER_MODEL if not MOCK else 'mock'}", test_gold, [teacher_labels[t] for t in test_texts])
tfidf_full = TfidfStudent().fit(pool_texts, pool_teacher)
r_tfidf = evaluate("tfidf_lr_all_pool_labels", test_gold, tfidf_full.predict(test_texts), {"labels_used": len(pool_texts)})
TEACHER_F1 = r_teacher["macro_f1"]"""))

C.append(md("""## 6 · Active distillation with the cheap TF-IDF student (fast, CPU)
3 strategies × 3 seeds. Quick evidence before the GPU runs."""))
C.append(code("""from sklearn.feature_extraction.text import TfidfVectorizer
from src.text import normalize_arabic
from src.active import run_curves, summarize_curves
from src.report import plot_curves, savings_table
from IPython.display import Image, display

features = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), preprocessor=normalize_arabic,
                           min_df=2).fit_transform(pool_texts)  # used by margin_diverse clustering
curves_tfidf = run_curves(pool_texts, pool_teacher, test_texts, test_gold, lambda s: TfidfStudent(seed=s),
                          config.STRATEGIES, config.SEEDS, config.SEED_SIZE, config.STEP_SIZE, config.BUDGET,
                          features=features, out="experiments/curves_tfidf.csv")
display(Image(plot_curves(curves_tfidf, TEACHER_F1, r_tfidf["macro_f1"], "experiments/curves_tfidf.png", "TF-IDF student")))
savings_table(curves_tfidf, r_tfidf["macro_f1"])"""))

C.append(md("""## 7 · The real student: MARBERTv2 on GPU
First the reference: student trained on **all** pool labels. Then active distillation curves (≈30–45 min; resumable)."""))
C.append(code("""import torch
from src.students import BertStudent
print("GPU:", torch.cuda.is_available())
make_bert = lambda s: BertStudent(STUDENT_MODEL, seed=s, max_len=config.MAX_LEN, epochs=config.EPOCHS,
                                  lr=config.LR, batch_size=config.BATCH_SIZE)
bert_full = make_bert(0).fit(pool_texts, pool_teacher)
r_bert_full = evaluate("bert_all_pool_labels", test_gold, bert_full.predict(test_texts), {"labels_used": len(pool_texts)})"""))
C.append(code("""curves_bert = run_curves(pool_texts, pool_teacher, test_texts, test_gold, make_bert,
                         config.STRATEGIES, config.SEEDS, config.SEED_SIZE, config.STEP_SIZE, config.BUDGET,
                         features=features, out="experiments/curves_bert.csv")
display(Image(plot_curves(curves_bert, TEACHER_F1, r_bert_full["macro_f1"], "experiments/curves_bert.png", "MARBERTv2 student")))
savings = savings_table(curves_bert, r_bert_full["macro_f1"])
savings"""))

C.append(md("## 8 · Final student: best strategy at the full budget → evaluate everywhere"))
C.append(code("""from src.active import run_active
s = summarize_curves(curves_bert)
BEST = s[s.labels_used == s.labels_used.max()].sort_values("mean").iloc[-1]["strategy"]
print("best strategy:", BEST)
_, picked, student = run_active(pool_texts, pool_teacher, test_texts, test_gold, make_bert, BEST,
                                config.SEED_SIZE, config.STEP_SIZE, config.BUDGET, seed=0, features=features)
r_student = evaluate(f"bert_{BEST}_{len(picked)}_labels", test_gold, student.predict(test_texts), {"labels_used": len(picked)})
print(f"student = {r_student['macro_f1'] / TEACHER_F1:.1%} of the teacher, using {len(picked)}/{len(pool_texts)} teacher labels")
per_class(test_gold, student.predict(test_texts)).to_frame().T"""))
C.append(code("""confusion(test_gold, student.predict(test_texts))"""))

C.append(md("## 9 · Robustness: stress sets + perturbations"))
C.append(code("""import pandas as pd
from src.evaluation import scores
from src.robustness import PERTURBATIONS, perturb
models = {"student (active)": student.predict, "student (all labels)": bert_full.predict, "tfidf (all labels)": tfidf_full.predict}
rows = []
sets = {"test (clean)": (test_texts, test_gold)}
sets.update({k: (list(v.text), list(v.gold)) for k, v in stress.items()})
sets.update({f"perturbed: {p}": (perturb(test_texts, p), test_gold) for p in PERTURBATIONS})
for set_name, (texts, gold) in sets.items():
    row = {"set": set_name, "n": len(texts)}
    for m, fn in models.items():
        row[m] = round(scores(gold, fn(texts))["macro_f1"], 3)
    if set_name in ("test (clean)",) or set_name.startswith("stress"):
        row["teacher"] = round(scores(gold, [teacher_labels[t] for t in texts])["macro_f1"], 3)
    rows.append(row)
robust = pd.DataFrame(rows)
robust.to_csv("experiments/robustness.csv", index=False)
robust"""))

C.append(md("""## 9b · Iteration: fix the weak spots with augmentation (zero extra teacher calls)
If the table above shows big drops (e.g. Arabizi, typos), retrain on the SAME teacher labels plus perturbed copies,
then compare. If it helps, the augmented model becomes the final student."""))
C.append(code("""from src.robustness import augment
picked_texts = [pool_texts[i] for i in picked]
picked_labels = [pool_teacher[i] for i in picked]
aug_t, aug_l = augment(picked_texts, picked_labels)
student_aug = make_bert(0).fit(aug_t, aug_l)
r_aug = evaluate(f"bert_{BEST}_{len(picked)}_labels_augmented", test_gold, student_aug.predict(test_texts), {"labels_used": len(picked)})
robust["student (active + augmentation)"] = [round(scores(g, student_aug.predict(t))["macro_f1"], 3) for t, g in sets.values()]
robust.to_csv("experiments/robustness.csv", index=False)
robust"""))
C.append(code("""# Keep the augmented student only if it is not worse on the clean test set
if r_aug["macro_f1"] >= r_student["macro_f1"] - 0.01:
    student, r_student = student_aug, r_aug
    print("final student = augmented")
else:
    print("final student = not augmented (augmentation hurt clean accuracy)")"""))

C.append(md("## 10 · Compute: CPU latency, size, quality after int8 (1 CPU thread)"))
C.append(code("""import numpy as np
from src.deploy import load_student, benchmark, predict as fast_predict
student.save(config.MODEL_DIR)
bench = []
for q in [False, True]:
    tok, m = load_student(config.MODEL_DIR, quantize=q, threads=1)
    probs = np.concatenate([fast_predict(tok, m, test_texts[i:i + 64]) for i in range(0, len(test_texts), 64)])
    preds = [LABEL_LIST[i] for i in probs.argmax(1)]
    bench.append({"model": "student int8" if q else "student fp32", "macro_f1": round(scores(test_gold, preds)["macro_f1"], 3),
                  **benchmark(tok, m, test_texts)})
torch.set_num_threads(os.cpu_count())
if not MOCK:  # teacher latency on 20 NEW single-message calls (not cached)
    t_teacher = Teacher(teacher_llm, model=TEACHER_MODEL, cache_path="/tmp/latency_probe.json")
    lat = []
    for t in test_texts[:20]:
        t0 = time.time(); t_teacher.label_many([t + " "], verbose=False); lat.append((time.time() - t0) * 1000)
    bench.append({"model": f"teacher {TEACHER_MODEL} (API)", "macro_f1": round(TEACHER_F1, 3),
                  "p50_ms": round(float(np.percentile(lat, 50))), "p95_ms": round(float(np.percentile(lat, 95))),
                  "msgs_per_sec": None, "size_mb": None, "ram_mb": None})
bench = pd.DataFrame(bench)
bench.to_csv("experiments/compute.csv", index=False)
bench"""))
C.append(md("Note: teacher latency includes the free-tier pacing wait (`MIN_SECONDS_BETWEEN_CALLS`) — set it to 0 for that cell if you want pure API latency."))

C.append(md("""## 10b · Hybrid mode: mini model first, big model only when unsure
No extra API calls: the big model's test answers are cached. Find the smallest share of messages that must go to the big model to match its quality."""))
C.append(code("""from src.cascade import cascade_curve, cheapest_threshold, plot_cascade
student_probs = student.predict_proba(test_texts)
teacher_test = [teacher_labels[t] for t in test_texts]
cascade = cascade_curve(student_probs, teacher_test, test_gold)
cascade.to_csv("experiments/cascade.csv", index=False)
display(Image(plot_cascade(cascade, r_student["macro_f1"], TEACHER_F1)))
for frac in (0.98, 0.99, 1.0):
    best = cheapest_threshold(cascade, frac * TEACHER_F1)
    if best:
        print(f"reach {frac:.0%} of big-model quality by sending only {best['sent_to_big_model']:.0%} of messages "
              f"(threshold {best['threshold']})")
cascade"""))

C.append(md("## 11 · Results file + push to GitHub"))
C.append(code("""from src.report import write_results
runs = pd.read_csv("experiments/runs.csv")
write_results([
    ("Main results (human-verified test set)", runs.drop_duplicates("name", keep="last")[["name", "macro_f1", "ci_low", "ci_high", "accuracy", "n"] + (["labels_used"] if "labels_used" in runs else [])]),
    ("Teacher labels needed (MARBERTv2 student)", savings),
    ("Robustness (macro-F1)", robust),
    ("Compute (1 CPU thread)", bench),
    ("Hybrid mode: big-model usage vs quality", cascade),
    ("Data", df.groupby(["set", "dialect"]).size().reset_index(name="n")),
])
print(open("experiments/RESULTS.md").read()[:3000])"""))
C.append(code("""!git add -A src scripts notebooks experiments docs data/messages.csv data/test_review.csv data/teacher_cache.json data/gen_cache.jsonl README.md
!git commit -qm "Experiments: data, curves, robustness, compute" && git push -q && echo pushed"""))

C.append(md("## 12 · Get the model onto your laptop (for the offline demo)"))
C.append(code("""!cd models && zip -qr student_final.zip student_final && ls -lh student_final.zip
from google.colab import drive
drive.mount("/content/drive")
!cp models/student_final.zip /content/drive/MyDrive/ && echo "saved to Google Drive -> download it on your laptop" """))

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "notebooks/hackathon.ipynb")
print("notebooks/hackathon.ipynb written,", len(C), "cells")
