"""PII detection and redaction.

The brief is explicit: *never send private or sensitive data to a third-party
service*. This module is the enforcement point. Anything that leaves the cluster
for a hosted LLM API goes through :func:`redact` first, and
:func:`assert_safe_for_external` is the hard gate that refuses to send text
that still looks like it contains personal data.

The Kaggle publisher already replaced most direct identifiers with placeholder
tokens such as ``<name>`` and ``<tel_num>``. We treat that as a starting point,
not as a guarantee, and re-scan for anything that slipped through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: more specific patterns run first so that, for example, an IBAN
# is not partially eaten by the generic long-number rule.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    (
        "phone",
        re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]?\d{3,4}(?:[ .-]?\d{2,4})?(?!\w)"),
    ),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("url", re.compile(r"https?://\S+|www\.\S+")),
    # Publisher placeholders. Detecting these lets us report honestly that the
    # source was already anonymised, and keeps them out of embeddings.
    ("placeholder", re.compile(r"<[a-z_]{2,20}>", re.IGNORECASE)),
]

_REPLACEMENT = {
    "email": "[EMAIL]",
    "iban": "[IBAN]",
    "credit_card": "[CARD]",
    "phone": "[PHONE]",
    "ip_address": "[IP]",
    "url": "[URL]",
    "placeholder": "[REDACTED]",
}

# Patterns that indicate genuinely sensitive personal data, as opposed to
# incidental identifiers like a URL. Only these block an external API call.
_BLOCKING = {"email", "iban", "credit_card", "phone"}


@dataclass(frozen=True)
class RedactionResult:
    text: str
    kinds: list[str]

    @property
    def has_pii(self) -> bool:
        return bool(self.kinds)

    @property
    def has_blocking_pii(self) -> bool:
        return any(k in _BLOCKING for k in self.kinds)


def _looks_like_false_positive(kind: str, match: str) -> bool:
    """Suppress obvious non-PII matches of the loose numeric patterns.

    The phone and credit-card regexes are intentionally greedy, which means they
    also match things like years, order counts and version numbers. Redacting
    those would silently destroy useful text, so we filter the clearest cases.
    """
    digits = re.sub(r"\D", "", match)
    if kind == "phone":
        # Fewer than 7 digits is a quantity, not a phone number.
        if len(digits) < 7:
            return True
        # A bare 4-digit year embedded in a longer numeric run.
        if len(digits) == 8 and digits.startswith(("19", "20")):
            return False
    if kind == "credit_card":
        if len(digits) < 13:
            return True
    return False


def redact(text: str) -> RedactionResult:
    """Replace anything that looks like PII with a typed placeholder.

    Returns both the redacted text and the sorted list of PII kinds found, which
    we index in Elasticsearch so the team can show *where* PII appears.
    """
    if not text:
        return RedactionResult("", [])

    found: set[str] = set()
    out = text

    for kind, pattern in _PATTERNS:

        def _sub(match: re.Match[str], _kind: str = kind) -> str:
            if _looks_like_false_positive(_kind, match.group(0)):
                return match.group(0)
            found.add(_kind)
            return _REPLACEMENT[_kind]

        out = pattern.sub(_sub, out)

    return RedactionResult(out, sorted(found))


def assert_safe_for_external(text: str) -> None:
    """Raise if ``text`` still contains blocking PII.

    Called immediately before any hosted-LLM request. This is a fail-closed
    check: we would rather drop an enrichment than leak a phone number.
    """
    result = redact(text)
    if result.has_blocking_pii:
        kinds = ", ".join(k for k in result.kinds if k in _BLOCKING)
        raise ValueError(
            f"refusing to send text to an external service: unredacted PII present ({kinds})"
        )
