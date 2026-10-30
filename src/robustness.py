"""Robustness: perturb the clean test set in ways real customers write, and re-score.

Programmatic perturbations (no API needed) complement the LLM-written stress sets
(unseen dialects, natural Arabizi, code-switching) from generate.py.
"""
import random

ARABIZI = {
    "ا": "a", "أ": "2", "إ": "e", "آ": "a", "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "7", "خ": "5",
    "د": "d", "ذ": "th", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "9", "ض": "9'", "ط": "6", "ظ": "6'",
    "ع": "3", "غ": "3'", "ف": "f", "ق": "g", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "h",
    "و": "w", "ي": "y", "ى": "a", "ء": "2", "ؤ": "2", "ئ": "2", "؟": "?", "،": ",",
}


def typos(text, rng, rate=0.1):
    """Delete or swap letters in ~10% of words."""
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 3 and rng.random() < rate:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:] if rng.random() < 0.5 else w[:j] + w[j + 1:]
    return " ".join(words)


def letter_variants(text, rng):
    """Spelling variants people type: hamza forms, ya/alef-maqsura, ta-marbuta/ha, tatweel."""
    swaps = {"ا": "أ", "أ": "ا", "إ": "ا", "ي": "ى", "ى": "ي", "ة": "ه", "ه": "ة"}
    out = []
    for ch in text:
        out.append(swaps[ch] if ch in swaps and rng.random() < 0.3 else ch)
        if "ء" <= ch <= "ي" and rng.random() < 0.03:
            out.append("ـ")
    return "".join(out)


def to_arabizi(text, rng):
    """Rule-based Arabic -> Arabizi transliteration."""
    return "".join(ARABIZI.get(ch, ch) for ch in text)


def truncate(text, rng, keep=0.5):
    """Customer sends only half the message."""
    w = text.split()
    return " ".join(w[:max(2, int(len(w) * keep))])


def noise(text, rng):
    """Repeated letters, emojis, extra punctuation."""
    if len(text) > 4 and rng.random() < 0.5:
        i = rng.randrange(len(text))
        text = text[:i] + text[i] * 3 + text[i:]
    return text + rng.choice([" 😡😡", "!!!", "؟؟", " 🙏", "..."])


PERTURBATIONS = {"typos": typos, "letter_variants": letter_variants, "arabizi_rule_based": to_arabizi,
                 "truncated_half": truncate, "noise_emoji": noise}


def perturb(texts, name, seed=0):
    rng = random.Random(seed)
    return [PERTURBATIONS[name](t, rng) for t in texts]


def augment(texts, labels, kinds=("arabizi_rule_based", "typos", "letter_variants"), seed=0):
    """Data augmentation with ZERO extra teacher cost: add perturbed copies of already-labeled messages.
    The label of a perturbed copy = label of the original (the teacher is not asked again)."""
    texts, labels = list(texts), list(labels)
    out_t, out_l = texts[:], labels[:]
    for k in kinds:
        out_t += perturb(texts, k, seed)
        out_l += labels
    return out_t, out_l
