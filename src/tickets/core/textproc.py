"""Text normalisation and language detection.

Pure functions only -- these are called from Spark UDFs, from the offline
runner and from tests, and must behave identically in all three.
"""

from __future__ import annotations

import re
import unicodedata

# The CSV stores newlines as the two characters backslash-n rather than as real
# newlines, so bodies look like "Dear Team,\\n\\nI would like...". Left alone,
# the token "n" pollutes every TF-IDF vector.
_ESCAPED_NEWLINE = re.compile(r"\\+[nrt]")
_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
# Placeholder tokens the dataset publisher already substituted for real PII,
# e.g. <name>, <tel_num>, <acc_num>. We keep them as a single token so they do
# not fragment into "<", "name", ">".
_PLACEHOLDER = re.compile(r"<([a-z_]+)>", re.IGNORECASE)

_TOKEN = re.compile(r"[a-zA-ZäöüÄÖÜßàâçéèêëîïôûùüÿñæœ]+")

# Frequency-ranked function words. English and German share almost none of
# these, which makes a simple stopword vote a reliable discriminator for the two
# languages actually present in this corpus.
_EN_MARKERS = {
    "the", "and", "is", "are", "for", "with", "you", "that", "this", "have", "would",
    "please", "could", "our", "your", "any", "from", "was", "were", "been", "will",
    "regarding", "issue", "team", "thank", "thanks", "dear", "kindly", "support",
    "about", "further", "assistance", "information", "we", "it", "to", "of", "in",
}

_DE_MARKERS = {
    "der", "die", "das", "und", "ich", "ist", "nicht", "mit", "sich", "für", "auf",
    "wir", "sie", "ein", "eine", "einen", "einem", "einer", "des", "dem", "den",
    "zu", "im", "am", "bei", "aus", "über", "von", "sehr", "geehrtes", "geehrte",
    "geehrter", "bitte", "danke", "vielen", "dank", "unser", "unsere", "problem",
    "können", "möchte", "haben", "wurde", "werden", "durch", "auch", "noch", "es",
}


def unescape(text: str) -> str:
    """Convert literal escape sequences into spaces."""
    if not text:
        return ""
    return _ESCAPED_NEWLINE.sub(" ", text)


def normalize(text: str) -> str:
    """Canonical whitespace + Unicode form. Keeps casing and accents.

    Casing is preserved because the embedding models are case-aware and because
    the redactor needs the original form to report what it found.
    """
    if not text:
        return ""
    text = unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def build_text(subject: str, body: str) -> str:
    """Join subject and body into the single field we embed and search.

    ``subject`` is null in ~13% of rows, so this must not produce a leading
    separator when it is missing.
    """
    parts = [normalize(subject), normalize(body)]
    return " ".join(p for p in parts if p).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens. URLs and <placeholders> are dropped first."""
    if not text:
        return []
    text = _URL.sub(" ", unescape(text))
    text = _PLACEHOLDER.sub(" ", text)
    return [t.lower() for t in _TOKEN.findall(text)]


def detect_language(text: str) -> str:
    """Return ``'en'``, ``'de'`` or ``'unknown'`` by stopword vote.

    Deliberately dependency-free: langdetect/fastText would be more accurate on
    arbitrary input, but this corpus contains exactly two languages and a
    stopword vote is transparent, instant and easy to defend in a viva. It is
    also stable, unlike ``langdetect``, which is seeded randomly by default.
    """
    tokens = tokenize(text)
    if not tokens:
        return "unknown"

    en_hits = sum(1 for t in tokens if t in _EN_MARKERS)
    de_hits = sum(1 for t in tokens if t in _DE_MARKERS)

    # German orthography: these characters essentially never appear in English.
    umlaut_hits = sum(1 for ch in text if ch in "äöüßÄÖÜ")
    de_hits += min(umlaut_hits, 5)

    if en_hits == 0 and de_hits == 0:
        return "unknown"
    # Require a clear margin; near-ties are genuinely ambiguous short texts.
    if en_hits >= de_hits * 1.25:
        return "en"
    if de_hits >= en_hits * 1.25:
        return "de"
    return "unknown"


def truncate(text: str, max_chars: int = 2000) -> str:
    """Clip text for prompts/embeddings without cutting mid-word."""
    if not text or len(text) <= max_chars:
        return text or ""
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars * 0.6 else cut).rstrip() + "..."
