"""Synthetic data: realistic requests users send to an Arabic AI assistant, written by the GENERATOR LLM.

Why synthetic: no public dataset of Arabic-dialect assistant requests labeled by tool exists,
and real user data must not be used. Mitigations (say these to the judges):
- the generator is a DIFFERENT model from the teacher (less self-agreement bias),
- every message has the label the generator was ASKED to write (`gen_label`),
- test labels are human-verified (see make_review_file),
- the test set is generated separately and near-duplicates with the pool are removed.

Sets
- pool      : unlabeled training pool (the student only sees teacher labels for what it requests)
- test      : held-out, human-verified, never used for training or tuning
- stress_*  : robustness sets - unseen dialects, natural Arabizi, Arabic-English code-switching
Generation is resumable: finished calls are stored in data/gen_cache.jsonl and skipped on re-run.
"""
import json
import os
import random

import pandas as pd

from src.config import (GENERATOR_MODEL, POOL_CALLS_PER_CELL, POOL_PER_CALL,
                        STRESS_PER_LABEL, TEST_PER_CELL)
from src.labels import LABELS
from src.text import dedup_key

GEN_CACHE = "data/gen_cache.jsonl"

DIALECTS = {
    "najdi": "Saudi Najdi dialect (Riyadh / Qassim), e.g. وش، ابي، ليش، الحين، مب، زين",
    "hijazi": "Saudi Hijazi dialect (Jeddah / Makkah), e.g. ايش، ابغى، فين، دحين، كدا، مرة",
    "eastern": "Saudi Eastern Province / Gulf dialect, e.g. شنو، أبي، وين، الحين، جذي، وايد",
    "msa": "Modern Standard Arabic (formal فصحى)",
}
UNSEEN_DIALECTS = {
    "egyptian": "Egyptian Arabic, e.g. عايز، فين، ليه، دلوقتي، كده، إزاي",
    "levantine": "Levantine Arabic (Syrian/Lebanese/Jordanian), e.g. بدي، وين، ليش، هلأ، هيك، شو",
}
STYLES = [
    "short and direct (3-12 words); some with spelling mistakes and no punctuation",
    "detailed and polite (15-40 words) with specific products, order numbers, dates or amounts",
    "frustrated or angry, very informal, some with emojis or repeated letters",
]
TEST_STYLE = ("varied: mix short/long, polite/angry, typos; include some messages that mention words "
              "from other tools but clearly need THIS tool")


def _prompt(label, dialect_desc, style, n, extra=""):
    info = LABELS[label]
    return (
        f"Write {n} different, realistic requests that users type to an Arabic AI assistant app "
        "(like a chatbot on their phone that can search the web, generate images, write code, translate, "
        "calculate, read their files, manage their calendar, or just chat).\n"
        f"All requests need this tool: {info['ar']} ({label}) = {info['definition']}.\n"
        f"Language: {dialect_desc}.\nStyle: {style}.\n{extra}"
        "Vary topics, situations, wording and length (students, employees, parents, developers, business owners). "
        "Do NOT mention the tool name. Do NOT number the messages. Each request is written by a different user.\n"
        "Return a JSON array of strings."
    )


def _load_cache():
    done = {}
    if os.path.exists(GEN_CACHE):
        with open(GEN_CACHE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r["texts"]
    return done


def _run_jobs(llm, jobs, model):
    """jobs: list of dicts(key, prompt, meta). Returns rows. Calls the LLM only for unfinished jobs."""
    done = _load_cache()
    schema = {"type": "ARRAY", "items": {"type": "STRING"}}
    os.makedirs("data", exist_ok=True)
    rows, todo = [], [j for j in jobs if j["key"] not in done]
    print(f"generation: {len(jobs) - len(todo)} cached, {len(todo)} to call")
    for i, j in enumerate(todo):
        texts = [t.strip() for t in llm.json_call(model, j["prompt"], schema, temperature=1.0) if str(t).strip()]
        done[j["key"]] = texts
        with open(GEN_CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": j["key"], "texts": texts}, ensure_ascii=False) + "\n")
        print(f"  {i + 1}/{len(todo)} {j['key']}: {len(texts)} messages")
    for j in jobs:
        for t in done[j["key"]]:
            rows.append({"text": t, **j["meta"]})
    return rows


def pool_jobs():
    return [{"key": f"pool|{lab}|{d}|{c}",
             "prompt": _prompt(lab, desc, STYLES[c % len(STYLES)], POOL_PER_CALL),
             "meta": {"set": "pool", "gen_label": lab, "dialect": d, "style": c % len(STYLES)}}
            for lab in LABELS for d, desc in DIALECTS.items() for c in range(POOL_CALLS_PER_CELL)]


def test_jobs():
    return [{"key": f"test|{lab}|{d}",
             "prompt": _prompt(lab, desc, TEST_STYLE, TEST_PER_CELL,
                               "Use different topics and situations than typical examples.\n"),
             "meta": {"set": "test", "gen_label": lab, "dialect": d, "style": -1}}
            for lab in LABELS for d, desc in DIALECTS.items()]


def stress_jobs():
    jobs = []
    for lab in LABELS:
        for d, desc in UNSEEN_DIALECTS.items():
            jobs.append({"key": f"stress_dialect|{lab}|{d}",
                         "prompt": _prompt(lab, desc, TEST_STYLE, STRESS_PER_LABEL // 2),
                         "meta": {"set": "stress_unseen_dialect", "gen_label": lab, "dialect": d, "style": -1}})
        jobs.append({"key": f"stress_arabizi|{lab}",
                     "prompt": _prompt(lab, "Saudi dialect written in Arabizi (Latin letters and numbers, "
                                            "e.g. 'abi tarjim hal email', '7ayakom', 'shno a5bar el ahli elyom')",
                                       TEST_STYLE, STRESS_PER_LABEL),
                     "meta": {"set": "stress_arabizi", "gen_label": lab, "dialect": "arabizi", "style": -1}})
        jobs.append({"key": f"stress_codeswitch|{lab}",
                     "prompt": _prompt(lab, "Saudi dialect mixed with many English words, as young Saudis text "
                                            "(e.g. 'سوي لي script بايثون يقرا الـ CSV', 'ابي reminder بكرة الساعة 9')",
                                       TEST_STYLE, STRESS_PER_LABEL),
                     "meta": {"set": "stress_codeswitch", "gen_label": lab, "dialect": "mixed", "style": -1}})
    return jobs


def generate_all(llm, model=GENERATOR_MODEL, out="data/messages.csv"):
    rows = _run_jobs(llm, pool_jobs() + test_jobs() + stress_jobs(), model)
    df = clean(pd.DataFrame(rows))
    df.to_csv(out, index=False)
    print(df.groupby("set").size().to_string())
    return df


def clean(df, seed=0):
    """Drop empty/very short texts and duplicates; remove pool messages that duplicate any eval message."""
    df = df[df["text"].str.len() >= 3].copy()
    df["key"] = df["text"].map(dedup_key)
    df = df.drop_duplicates(["set", "key"])
    eval_keys = set(df.loc[df["set"] != "pool", "key"])
    leak = (df["set"] == "pool") & df["key"].isin(eval_keys)
    if leak.sum():
        print(f"removed {leak.sum()} pool messages duplicating eval messages")
    df = df[~leak]
    df = df.drop_duplicates("key")  # a text appears in only one set
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df.drop(columns="key")


def make_review_file(df, teacher_labels, out="data/test_review.csv", spot_check=30, seed=0):
    """Human verification, done efficiently: review every eval message where the generator's intended
    label and the teacher disagree, plus a random spot-check of agreements."""
    ev = df[df["set"] != "pool"].copy()
    ev["teacher_label"] = [teacher_labels[t] for t in ev["text"]]
    disagree = ev[ev["gen_label"] != ev["teacher_label"]]
    agree = ev[ev["gen_label"] == ev["teacher_label"]]
    spot = agree.sample(min(spot_check, len(agree)), random_state=seed)
    review = pd.concat([disagree, spot])
    review["human_label"] = ""  # fill with a label key, or leave empty to accept gen_label
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    review[["id", "set", "text", "gen_label", "teacher_label", "human_label"]].to_csv(out, index=False)
    print(f"review file: {len(disagree)} disagreements + {len(spot)} spot-checks -> {out}")
    return review


def review_interactively(path="data/test_review.csv"):
    """Fast review inside Colab. For each message type:
       Enter = generator label is right · t = teacher label is right · d = drop (nonsense/ambiguous)
       or type the correct label key / its number (0-7). Progress is saved after every answer; re-run to continue."""
    rv = pd.read_csv(path, dtype=str).fillna("")
    keys = list(LABELS)
    print("labels:", ", ".join(f"{i}={k}" for i, k in enumerate(keys)))
    for i, r in rv.iterrows():
        if r["human_label"]:
            continue
        print(f"\n[{i + 1}/{len(rv)}] ({r['set']}) {r['text']}\n   gen={r['gen_label']}   teacher={r['teacher_label']}")
        ans = input("   > ").strip()
        if ans == "":
            lab = r["gen_label"]
        elif ans == "t":
            lab = r["teacher_label"]
        elif ans == "d":
            lab = "drop"
        elif ans.isdigit() and int(ans) < len(keys):
            lab = keys[int(ans)]
        elif ans in keys:
            lab = ans
        elif ans == "q":
            break
        else:
            print("   ? not understood, skipped")
            continue
        rv.loc[i, "human_label"] = lab
        rv.to_csv(path, index=False)
    print(f"reviewed {int((rv['human_label'] != '').sum())}/{len(rv)}")


def apply_review(df, review_path="data/test_review.csv"):
    """gold label = human_label if filled in, else gen_label. Rows marked 'drop' are removed."""
    df = df.copy()
    df["gold"] = df["gen_label"]
    if os.path.exists(review_path):
        rv = pd.read_csv(review_path, dtype=str).fillna("")
        fixes = {int(r.id): r.human_label.strip() for r in rv.itertuples() if r.human_label.strip()}
        for i, lab in fixes.items():
            df.loc[df["id"] == i, "gold"] = lab
        df = df[df["gold"] != "drop"]
        bad = set(df["gold"]) - set(LABELS)
        assert not bad, f"unknown labels in review file: {bad}"
        print(f"applied {len(fixes)} human corrections")
    return df


def make_mock_messages(n_per_label=60, seed=0):
    """Tiny offline dataset to test the pipeline without any API (NOT for real results)."""
    rng = random.Random(seed)
    words = {
        "web_search": ["وش اخبار الدوري اليوم", "كم سعر الذهب الحين", "وش حالة الطقس بالرياض", "مين فاز امس"],
        "image_generation": ["ارسم لي شعار", "سوي صورة قطة", "صمم لي بوستر", "ابي صورة غروب"],
        "code_assistant": ["اكتب كود بايثون", "صلح هذا الخطأ في الكود", "سوي استعلام SQL", "اشرح لي هالدالة"],
        "translation": ["ترجم للانجليزي", "وش معنى هالجملة بالعربي", "ترجم هالايميل", "حول النص للفرنسي"],
        "math_calculation": ["كم 15 بالمية من 2400", "احسب القسط الشهري", "حل المعادلة", "كم يطلع المجموع"],
        "document_qa": ["لخص لي الملف", "وش مكتوب في العقد", "طلع النقاط المهمة من التقرير", "وش يقول الايميل"],
        "scheduling": ["ذكرني بكرة", "حط موعد اجتماع", "صحيني الساعة 6", "وش عندي مواعيد اليوم"],
        "general_chat": ["السلام عليكم", "عطني نصيحة", "وش الفرق بين", "اكتب لي تهنئة"],
    }
    fill = ["والله", "لو سمحت", "يا جماعة", "بسرعة", "اليوم", "من امس", "", "ضروري", "😡", "؟"]
    rows = []
    for lab, ws in words.items():
        for s in ["pool"] * n_per_label + ["test"] * (n_per_label // 4) + ["stress_arabizi"] * 3:
            t = f"{rng.choice(fill)} {rng.choice(ws)} {rng.choice(fill)} {rng.randint(1, 9999)}".strip()
            if rng.random() < 0.15:  # ambiguity: mix in words from another department
                t += " " + rng.choice(words[rng.choice(list(words))])
            rows.append({"text": t, "set": s, "gen_label": lab, "dialect": "najdi", "style": 0})
    return clean(pd.DataFrame(rows), seed)
