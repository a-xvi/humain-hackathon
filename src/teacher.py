"""The TEACHER: a large LLM that labels messages. Its labels are what the student learns from.

Design decisions (defend these):
- Forced choice: the JSON schema only allows our 8 label keys -> no invented labels.
- Temperature 0 -> the same message always gets the same label (reproducible).
- Batching: 20 messages per call -> ~20x fewer API calls (free-tier friendly).
- Cache on disk -> re-running costs nothing; a message is never paid for twice.
- Cost is reported as LABELS REQUESTED (messages sent to the teacher), the quantity
  active distillation tries to minimise. API calls are reported too.
"""
import hashlib
import json
import os

from src.config import TEACHER_BATCH_SIZE, TEACHER_MODEL
from src.labels import LABEL_LIST, LABELS

CACHE_PATH = "data/teacher_cache.json"


def build_prompt(texts):
    deps = "\n".join(f"- {k}: {v['definition']}" for k, v in LABELS.items())
    msgs = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (
        "You are the routing system of a Saudi e-commerce company's customer service.\n"
        "Assign EACH customer message (Saudi dialect, other Arabic dialects, MSA, Arabizi or mixed English) "
        "to exactly ONE department.\n\n"
        f"Departments:\n{deps}\n\n"
        f"Messages:\n{msgs}\n\n"
        f"Return a JSON array with exactly {len(texts)} department keys, in the same order as the messages."
    )


class Teacher:
    def __init__(self, llm, model=TEACHER_MODEL, cache_path=CACHE_PATH):
        self.llm, self.model, self.cache_path = llm, model, cache_path
        self.labels_requested = 0
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)

    def _key(self, text):
        return hashlib.sha1(f"{self.model}|{text}".encode()).hexdigest()

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)

    def _label_batch(self, texts):
        schema = {"type": "ARRAY", "items": {"type": "STRING", "enum": LABEL_LIST}}
        out = self.llm.json_call(self.model, build_prompt(texts), schema, temperature=0.0)
        if isinstance(out, list) and len(out) == len(texts) and all(o in LABEL_LIST for o in out):
            return out
        if len(texts) == 1:
            return ["other"]  # should not happen with the enum schema; safe fallback
        mid = len(texts) // 2  # length mismatch -> split the batch and retry each half
        return self._label_batch(texts[:mid]) + self._label_batch(texts[mid:])

    def label_many(self, texts, batch_size=TEACHER_BATCH_SIZE, verbose=True):
        """Label a list of texts. Only uncached texts are sent to the API."""
        todo = list(dict.fromkeys(t for t in texts if self._key(t) not in self.cache))
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            for t, lab in zip(batch, self._label_batch(batch)):
                self.cache[self._key(t)] = lab
            self.labels_requested += len(batch)
            self._save()
            if verbose:
                print(f"  teacher: {min(i + batch_size, len(todo))}/{len(todo)} new labels")
        return [self.cache[self._key(t)] for t in texts]

    def label(self, text):
        return self.label_many([text], verbose=False)[0]


class MockTeacher:
    """Offline stand-in for tests: returns the generator's intended label, with some noise."""

    def __init__(self, truth, noise=0.05, seed=0):
        import random
        self.truth, self.noise, self.rng = truth, noise, random.Random(seed)
        self.labels_requested = 0

    def label_many(self, texts, **_):
        self.labels_requested += len(texts)
        return [self.truth[t] if self.rng.random() > self.noise else self.rng.choice(LABEL_LIST) for t in texts]
