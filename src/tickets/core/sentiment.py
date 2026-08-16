"""Bilingual lexicon sentiment scoring.

A deliberately simple, explainable baseline that runs offline in microseconds.
It exists so the pipeline always produces a sentiment signal even with no model
downloaded and no API key; when ``LLM_PROVIDER`` is configured the LLM verdict
overrides it (see :mod:`tickets.ai.llm`).

Scope note for the viva: this is a lexicon, not a classifier. It has no notion
of negation scope beyond a one-token window and no sarcasm handling. We use it
as a *baseline*, and the evaluation script quantifies how far it is from the
LLM labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from .textproc import tokenize

_NEGATIVE = {
    # English
    "angry", "annoyed", "broken", "cannot", "complaint", "crash", "crashed", "critical",
    "delay", "delayed", "disappointed", "down", "error", "fail", "failed", "failure",
    "frustrated", "frustrating", "immediately", "impact", "impacted", "inconvenience",
    "issue", "lost", "outage", "poor", "problem", "refund", "reject", "rejected",
    "severe", "slow", "stuck", "unable", "unacceptable", "unavailable", "urgent",
    "wrong", "breach", "vulnerability", "escalate", "escalation", "blocked",
    # German
    "abgelehnt", "ärgerlich", "ausfall", "beschwerde", "defekt", "dringend", "fehler",
    "fehlgeschlagen", "gestört", "kaputt", "kritisch", "langsam", "leider", "mangel",
    "problem", "störung", "unbrauchbar", "unzufrieden", "verzögerung", "verloren",
    "nicht", "keine", "schwerwiegend", "beeinträchtigt", "verzögert",
}

_POSITIVE = {
    # English
    "appreciate", "appreciated", "brilliant", "excellent", "fantastic", "glad", "good",
    "grateful", "great", "happy", "helpful", "impressed", "love", "perfect", "pleased",
    "positive", "quick", "resolved", "satisfied", "smooth", "success", "successful",
    "thank", "thanks", "wonderful", "works", "working", "efficient", "clear",
    # German
    "ausgezeichnet", "begeistert", "danke", "dankbar", "erfolgreich", "freundlich",
    "gelöst", "gut", "hervorragend", "hilfreich", "perfekt", "schnell", "super",
    "toll", "wunderbar", "zufrieden", "reibungslos", "klar",
}

# A negator immediately before a sentiment word flips its polarity.
_NEGATORS = {"not", "no", "never", "without", "kein", "keine", "nicht", "ohne", "nie"}


@dataclass(frozen=True)
class SentimentResult:
    label: str  # "negative" | "neutral" | "positive"
    score: float  # -1.0 .. 1.0


def score_sentiment(text: str) -> SentimentResult:
    """Score text on a -1..1 axis and bucket it into a label."""
    tokens = tokenize(text)
    if not tokens:
        return SentimentResult("neutral", 0.0)

    positive = 0
    negative = 0
    for index, token in enumerate(tokens):
        negated = index > 0 and tokens[index - 1] in _NEGATORS
        if token in _POSITIVE:
            negative += 1 if negated else 0
            positive += 0 if negated else 1
        elif token in _NEGATIVE:
            positive += 1 if negated else 0
            negative += 0 if negated else 1

    total = positive + negative
    if total == 0:
        return SentimentResult("neutral", 0.0)

    # Normalise by hits rather than by length: a long ticket is not automatically
    # more neutral than a short one.
    raw = (positive - negative) / total
    # Dampen very thin evidence so a single word does not yield a +/-1.0 score.
    confidence = min(total, 5) / 5
    score = round(raw * confidence, 4)

    if score <= -0.15:
        label = "negative"
    elif score >= 0.15:
        label = "positive"
    else:
        label = "neutral"
    return SentimentResult(label, score)
