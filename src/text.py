"""Arabic text normalization and near-duplicate detection."""
import re

_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")
_SPACES = re.compile(r"\s+")


def normalize_arabic(text):
    """Remove diacritics/tatweel, unify alef/ya/ta-marbuta forms, lowercase Latin, squeeze spaces."""
    s = _DIACRITICS.sub("", str(text))
    s = s.replace("ـ", "")
    s = re.sub("[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    return _SPACES.sub(" ", s).strip().lower()


def dedup_key(text):
    """Key used to detect duplicates: normalized text without punctuation."""
    return re.sub(r"[^\w\s]", "", normalize_arabic(text))
