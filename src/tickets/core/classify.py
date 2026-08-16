"""Rule-based zero-shot routing and priority scoring.

This is the *baseline* against which the AI components are measured. It is a
transparent keyword scorer: every prediction can be traced to the exact terms
that fired, which makes it trivial to explain and gives the evaluation script a
meaningful floor. If the embedding kNN classifier or the LLM cannot beat this,
they are not earning their keep.

Ground truth for the comparison is the dataset's own ``queue`` and ``priority``
columns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .schema import PRIORITIES, QUEUES
from .textproc import tokenize

# Indicative terms per queue, English and German. Weighted: a term that is
# near-diagnostic of a queue scores higher than a merely suggestive one.
_QUEUE_TERMS: dict[str, dict[str, float]] = {
    "Billing and Payments": {
        "invoice": 3.0, "billing": 3.0, "payment": 3.0, "charge": 2.0, "charged": 2.0,
        "refund": 2.5, "subscription": 1.5, "price": 1.5, "receipt": 2.0, "tax": 1.5,
        "rechnung": 3.0, "zahlung": 3.0, "abrechnung": 3.0, "erstattung": 2.5,
        "gebühr": 2.0, "preis": 1.5, "abonnement": 1.5,
    },
    "Returns and Exchanges": {
        "return": 3.0, "exchange": 2.5, "replacement": 2.5, "rma": 3.0, "defective": 2.0,
        "damaged": 2.0, "shipping": 1.5, "delivery": 1.5, "warranty": 2.0,
        "rückgabe": 3.0, "umtausch": 3.0, "ersatz": 2.0, "beschädigt": 2.0,
        "garantie": 2.0, "lieferung": 1.5, "versand": 1.5,
    },
    "Service Outages and Maintenance": {
        "outage": 3.5, "downtime": 3.0, "maintenance": 3.0, "unavailable": 2.0,
        "restore": 2.0, "incident": 1.5, "disruption": 2.5, "degraded": 2.5,
        "ausfall": 3.5, "wartung": 3.0, "störung": 3.0, "ausgefallen": 2.5,
        "wiederherstellung": 2.0, "unterbrechung": 2.5,
    },
    "Sales and Pre-Sales": {
        "quote": 3.0, "pricing": 2.5, "demo": 2.5, "trial": 2.0, "purchase": 2.0,
        "license": 1.5, "contract": 1.5, "proposal": 2.0, "upgrade": 1.0,
        "angebot": 3.0, "kauf": 2.0, "vertrag": 1.5, "lizenz": 1.5, "testversion": 2.0,
    },
    "Human Resources": {
        "onboarding": 2.5, "employee": 2.5, "payroll": 3.0, "leave": 2.0, "hr": 3.0,
        "recruitment": 3.0, "training": 1.5, "candidate": 2.0,
        "mitarbeiter": 2.5, "personal": 2.0, "gehalt": 3.0, "urlaub": 2.0,
        "einarbeitung": 2.5, "schulung": 1.5,
    },
    "IT Support": {
        "password": 2.5, "login": 2.0, "account": 1.5, "access": 1.5, "vpn": 3.0,
        "laptop": 2.0, "printer": 2.5, "email": 1.5, "network": 2.0, "firewall": 2.5,
        "passwort": 2.5, "anmeldung": 2.0, "zugang": 1.5, "drucker": 2.5,
        "netzwerk": 2.0, "konto": 1.5,
    },
    "Technical Support": {
        "error": 2.0, "bug": 2.5, "crash": 2.5, "exception": 2.5, "server": 2.0,
        "database": 2.0, "api": 2.0, "performance": 1.5, "latency": 2.0, "security": 1.5,
        "fehler": 2.0, "absturz": 2.5, "datenbank": 2.0, "leistung": 1.5,
        "sicherheit": 1.5, "schnittstelle": 2.0,
    },
    "Product Support": {
        "feature": 2.0, "documentation": 2.0, "integration": 2.0, "configuration": 1.5,
        "setup": 1.5, "how": 0.5, "guide": 2.0, "tutorial": 2.0, "compatibility": 2.0,
        "funktion": 2.0, "dokumentation": 2.0, "einrichtung": 1.5, "anleitung": 2.0,
        "konfiguration": 1.5,
    },
    "Customer Service": {
        "complaint": 2.5, "feedback": 2.0, "satisfaction": 2.0, "apolog": 1.5,
        "service": 1.0, "experience": 1.5, "cancel": 2.0,
        "beschwerde": 2.5, "kundendienst": 2.5, "zufriedenheit": 2.0, "kündigen": 2.0,
    },
    "General Inquiry": {
        "inquiry": 2.0, "question": 1.5, "information": 1.0, "general": 1.5,
        "anfrage": 2.0, "frage": 1.5, "auskunft": 1.5,
    },
}

# Terms that argue for a given priority.
_PRIORITY_TERMS: dict[str, dict[str, float]] = {
    "high": {
        "urgent": 3.0, "critical": 3.0, "immediately": 2.5, "asap": 3.0, "outage": 2.5,
        "down": 2.0, "breach": 3.0, "security": 1.5, "blocked": 2.0, "severe": 2.5,
        "production": 2.0, "escalate": 2.5, "emergency": 3.0, "data loss": 3.0,
        "dringend": 3.0, "kritisch": 3.0, "sofort": 2.5, "notfall": 3.0,
        "schwerwiegend": 2.5, "ausfall": 2.5, "eskalation": 2.5,
    },
    "low": {
        "whenever": 2.0, "minor": 2.5, "cosmetic": 3.0, "suggestion": 2.0,
        "nice to have": 3.0, "question": 1.0, "curious": 2.0, "documentation": 1.0,
        "gelegentlich": 2.0, "geringfügig": 2.5, "vorschlag": 2.0, "frage": 1.0,
    },
}


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    evidence: list[str]


def _score(tokens: list[str], text_lower: str, terms: dict[str, float]) -> tuple[float, list[str]]:
    """Score one label. Single tokens match the token list; multi-word terms
    are checked against the raw lowercased text."""
    total = 0.0
    evidence: list[str] = []
    token_set = set(tokens)
    for term, weight in terms.items():
        if " " in term:
            if term in text_lower:
                total += weight
                evidence.append(term)
        elif term in token_set:
            total += weight
            evidence.append(term)
    return total, evidence


def classify_queue(text: str) -> Prediction:
    """Predict the routing queue for a ticket.

    Confidence is the winning score's share of all scores, so a runaway winner
    reports high confidence and a near-tie reports low. Falls back to the
    corpus-majority queue when nothing matches.
    """
    tokens = tokenize(text)
    text_lower = (text or "").lower()

    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = {}
    for queue, terms in _QUEUE_TERMS.items():
        value, hits = _score(tokens, text_lower, terms)
        scores[queue] = value
        evidence[queue] = hits

    total = sum(scores.values())
    if total <= 0:
        # "Technical Support" is the majority class (29% of the corpus), so it is
        # the least-wrong guess in the absence of any signal.
        return Prediction("Technical Support", 0.0, [])

    best = max(scores, key=lambda q: scores[q])
    return Prediction(best, round(scores[best] / total, 4), evidence[best][:8])


def classify_priority(text: str) -> Prediction:
    """Predict priority. ``medium`` is the default when neither pole fires."""
    tokens = tokenize(text)
    text_lower = (text or "").lower()

    high, high_hits = _score(tokens, text_lower, _PRIORITY_TERMS["high"])
    low, low_hits = _score(tokens, text_lower, _PRIORITY_TERMS["low"])

    if high == 0 and low == 0:
        return Prediction("medium", 0.0, [])
    if high >= low * 1.5:
        return Prediction("high", round(high / (high + low + 1e-9), 4), high_hits[:6])
    if low >= high * 1.5:
        return Prediction("low", round(low / (high + low + 1e-9), 4), low_hits[:6])
    return Prediction("medium", 0.25, (high_hits + low_hits)[:6])


def valid_queue(label: str) -> bool:
    return label in QUEUES


def valid_priority(label: str) -> bool:
    return label in PRIORITIES
