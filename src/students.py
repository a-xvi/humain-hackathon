"""STUDENT models. Both share one interface:  fit(texts, labels) / predict_proba(texts) / predict(texts).

- TfidfStudent : char n-gram TF-IDF + logistic regression. Classic cheap baseline, trains in seconds on CPU.
- BertStudent  : small Arabic transformer (MARBERTv2) fine-tuned for classification. The real student.

Why an ENCODER (BERT-style) and not a small generative LLM:
routing is classification - we need one label, not generated text. An encoder of ~160M parameters
reads the message once and outputs 8 probabilities: smaller, faster on CPU, and gives calibrated-ish
confidence scores that the active-learning loop needs.
"""
import numpy as np

from src.labels import LABEL2ID, LABEL_LIST, NUM_LABELS
from src.text import normalize_arabic


def _full_proba(proba, classes):
    """Map a model's probabilities (only for classes it saw) onto all 8 labels."""
    out = np.zeros((proba.shape[0], NUM_LABELS))
    for j, c in enumerate(classes):
        out[:, LABEL2ID[c]] = proba[:, j]
    return out


class TfidfStudent:
    name = "tfidf_lr"

    def __init__(self, seed=0, C=4.0):
        self.seed, self.C = seed, C

    def fit(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, preprocessor=normalize_arabic),
            LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed),
        )
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts):
        return _full_proba(self.pipe.predict_proba(list(texts)), self.pipe.classes_)

    def predict(self, texts):
        return [LABEL_LIST[i] for i in self.predict_proba(texts).argmax(1)]


class BertStudent:
    name = "bert"

    def __init__(self, model_name, seed=0, max_len=64, epochs=4, lr=3e-5, batch_size=16, device=None):
        import torch
        self.model_name, self.seed, self.max_len = model_name, seed, max_len
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _tok(self):
        from transformers import AutoTokenizer
        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def fit(self, texts, labels):
        """Fine-tune from the pretrained checkpoint (fresh start every call = fair comparison between rounds)."""
        import torch
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        tok = self._tok()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=NUM_LABELS,
            id2label=dict(enumerate(LABEL_LIST)), label2id=LABEL2ID,
            ignore_mismatched_sizes=True).to(self.device)
        texts, y = list(texts), np.array([LABEL2ID[l] for l in labels])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * int(np.ceil(len(texts) / self.batch_size))
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(texts))
            for s in range(0, len(texts), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = tok([texts[i] for i in idx], truncation=True, max_length=self.max_len,
                          padding=True, return_tensors="pt").to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = self.model(**enc, labels=torch.tensor(y[idx], device=self.device)).loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
        return self

    def predict_proba(self, texts, batch_size=64):
        import torch
        tok = self._tok()
        self.model.eval()
        texts, out = list(texts), []
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                enc = tok(texts[s:s + batch_size], truncation=True, max_length=self.max_len,
                          padding=True, return_tensors="pt").to(self.model.device)
                out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, NUM_LABELS))

    def predict(self, texts):
        return [LABEL_LIST[i] for i in self.predict_proba(texts).argmax(1)]

    def save(self, path):
        self.model.save_pretrained(path)
        self._tok().save_pretrained(path)
