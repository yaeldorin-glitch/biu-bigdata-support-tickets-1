"""Canonical ticket schema.

The Kaggle CSV is *semi-structured*: two free-text fields (``subject``, ``body``),
one free-text agent reply (``answer``), and a ragged tag list spread across eight
sparse columns (``tag_1`` .. ``tag_8``, 1255 distinct values, mostly null past
``tag_4``). This module is the single place that knows how to turn that shape
into a clean record, and it is deliberately free of Spark/ES imports so it can be
unit tested and reused inside UDFs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Columns exactly as they appear in the Kaggle CSV.
RAW_COLUMNS = [
    "subject",
    "body",
    "answer",
    "type",
    "queue",
    "priority",
    "language",
    "version",
    *[f"tag_{i}" for i in range(1, 9)],
]

TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]

# Label vocabularies observed in the data. Used to validate incoming records and
# as the candidate set for zero-shot / LLM classification.
QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]

TYPES = ["Incident", "Request", "Problem", "Change"]

PRIORITIES = ["low", "medium", "high"]

LANGUAGES = ["en", "de"]


@dataclass
class Ticket:
    """A single support ticket after parsing, before enrichment."""

    ticket_id: str
    subject: str
    body: str
    answer: str
    type: str
    queue: str
    priority: str
    language: str
    version: int
    tags: list[str] = field(default_factory=list)

    # Populated by the enrichment stage.
    text: str = ""
    text_redacted: str = ""
    pii_found: list[str] = field(default_factory=list)
    detected_language: str = ""
    language_mismatch: bool = False
    sentiment: str = ""
    sentiment_score: float = 0.0
    predicted_queue: str = ""
    predicted_queue_confidence: float = 0.0
    predicted_priority: str = ""
    embedding: list[float] = field(default_factory=list)
    enrichment_source: str = ""
    ingest_ts: str = ""
    ingest_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def make_ticket_id(row: dict[str, Any]) -> str:
    """Stable synthetic id.

    The source CSV has no primary key, and a running counter would break
    idempotency: replaying the Kafka topic would produce different ids and
    duplicate every document in Elasticsearch. Hashing the content instead makes
    the whole pipeline idempotent -- re-ingesting the same row overwrites the
    same ES document rather than creating a second one.
    """
    basis = "|".join(
        str(row.get(c, "") or "") for c in ("subject", "body", "answer", "queue", "language")
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def collect_tags(row: dict[str, Any]) -> list[str]:
    """Flatten tag_1..tag_8 into a deduplicated list, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for col in TAG_COLUMNS:
        value = row.get(col)
        if value is None:
            continue
        tag = str(value).strip()
        if not tag or tag.lower() in {"nan", "none", "null"}:
            continue
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _clean_str(value: Any) -> str:
    """CSV nulls arrive as NaN via pandas and as the string 'nan' via Spark."""
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_row(row: dict[str, Any]) -> Ticket:
    """Turn one raw CSV/JSON row into a :class:`Ticket`.

    Tolerant by design: the ``subject`` column is null in 3,838 of 28,587 rows,
    so a missing subject is normal data rather than an error.
    """
    version_raw = row.get("version", 0)
    try:
        version = int(float(version_raw)) if _clean_str(version_raw) else 0
    except (TypeError, ValueError):
        version = 0

    return Ticket(
        ticket_id=make_ticket_id(row),
        subject=_clean_str(row.get("subject")),
        body=_clean_str(row.get("body")),
        answer=_clean_str(row.get("answer")),
        type=_clean_str(row.get("type")),
        queue=_clean_str(row.get("queue")),
        priority=_clean_str(row.get("priority")).lower(),
        language=_clean_str(row.get("language")).lower(),
        version=version,
        tags=collect_tags(row),
    )


def is_valid(ticket: Ticket) -> tuple[bool, str]:
    """Validate a parsed ticket. Returns ``(ok, reason)``.

    Invalid records are routed to the dead-letter topic instead of crashing the
    stream -- one malformed row should never stop a 28k-row ingest.
    """
    if not ticket.body.strip():
        return False, "empty_body"
    if ticket.queue and ticket.queue not in QUEUES:
        return False, f"unknown_queue:{ticket.queue}"
    if ticket.priority and ticket.priority not in PRIORITIES:
        return False, f"unknown_priority:{ticket.priority}"
    if ticket.type and ticket.type not in TYPES:
        return False, f"unknown_type:{ticket.type}"
    return True, ""


# --------------------------------------------------------------------------- #
# Elasticsearch mapping
# --------------------------------------------------------------------------- #


def es_mapping(embedding_dim: int = 384) -> dict[str, Any]:
    """Index mapping for the ``tickets`` index.

    Two things matter here:

    * ``text`` is analysed for BM25 keyword search, which we keep as the baseline
      to compare semantic search against.
    * ``embedding`` is a ``dense_vector`` with ``index: true`` and cosine
      similarity, which is what enables ``knn`` search. Without ``index: true``
      the field is stored but not searchable by kNN.
    """
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    # The corpus is English + German, so we index with a language
                    # neutral analyser and lean on the multilingual embedding for
                    # cross-language recall.
                    "ticket_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "ticket_id": {"type": "keyword"},
                "subject": {"type": "text", "analyzer": "ticket_analyzer"},
                "body": {"type": "text", "analyzer": "ticket_analyzer"},
                "answer": {"type": "text", "analyzer": "ticket_analyzer"},
                "text": {"type": "text", "analyzer": "ticket_analyzer"},
                "text_redacted": {"type": "text", "analyzer": "ticket_analyzer"},
                "type": {"type": "keyword"},
                "queue": {"type": "keyword"},
                "priority": {"type": "keyword"},
                "language": {"type": "keyword"},
                "detected_language": {"type": "keyword"},
                "language_mismatch": {"type": "boolean"},
                "version": {"type": "integer"},
                "tags": {"type": "keyword"},
                "pii_found": {"type": "keyword"},
                "sentiment": {"type": "keyword"},
                "sentiment_score": {"type": "float"},
                "predicted_queue": {"type": "keyword"},
                "predicted_queue_confidence": {"type": "float"},
                "predicted_priority": {"type": "keyword"},
                "enrichment_source": {"type": "keyword"},
                "ingest_ts": {"type": "date"},
                "ingest_date": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }


def kpi_mapping() -> dict[str, Any]:
    """Index mapping for the ``ticket_kpis`` index written by the batch job."""
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "kpi": {"type": "keyword"},
                "dimension": {"type": "keyword"},
                "bucket": {"type": "keyword"},
                "value": {"type": "double"},
                "count": {"type": "long"},
                "computed_at": {"type": "date"},
            }
        },
    }
