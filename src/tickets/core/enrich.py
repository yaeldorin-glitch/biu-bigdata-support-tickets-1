"""The enrichment step, expressed as pure functions.

This is the transformation the Spark job applies to every ticket. Keeping it
here -- rather than inline in the streaming job -- means the exact same code is
exercised by the unit tests, by the offline runner and by Spark's UDFs, so a
green test suite says something real about the pipeline.

Embeddings are handled separately (:func:`attach_embeddings`) because they are
far more efficient in batches than row by row.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from .classify import classify_priority, classify_queue
from .pii import redact
from .schema import Ticket
from .sentiment import score_sentiment
from .textproc import build_text, detect_language


def enrich_ticket(ticket: Ticket, ingest_ts: datetime | None = None) -> Ticket:
    """Apply every row-local enrichment. Mutates and returns ``ticket``.

    Ordering matters: text is assembled first, redaction runs on that assembled
    text, and every downstream signal is computed from the *redacted* version so
    that placeholder tokens never leak into an embedding or a prompt.
    """
    ts = ingest_ts or datetime.now(timezone.utc)

    ticket.text = build_text(ticket.subject, ticket.body)

    redaction = redact(ticket.text)
    ticket.text_redacted = redaction.text
    ticket.pii_found = redaction.kinds

    # Language: the dataset ships a `language` label, but it disagrees with the
    # actual text often enough to be worth measuring, so we detect independently
    # and record the disagreement rather than trusting either blindly.
    ticket.detected_language = detect_language(ticket.text)
    ticket.language_mismatch = bool(
        ticket.language
        and ticket.detected_language != "unknown"
        and ticket.detected_language != ticket.language
    )

    sentiment = score_sentiment(ticket.text_redacted)
    ticket.sentiment = sentiment.label
    ticket.sentiment_score = sentiment.score

    queue_prediction = classify_queue(ticket.text_redacted)
    ticket.predicted_queue = queue_prediction.label
    ticket.predicted_queue_confidence = queue_prediction.confidence

    ticket.predicted_priority = classify_priority(ticket.text_redacted).label

    ticket.enrichment_source = "rules"
    ticket.ingest_ts = ts.isoformat()
    ticket.ingest_date = ts.date().isoformat()
    return ticket


def attach_embeddings(tickets: Sequence[Ticket], backend) -> Sequence[Ticket]:
    """Embed a batch of tickets in one call and attach the vectors."""
    if not tickets:
        return tickets
    texts = [t.text_redacted or t.text for t in tickets]
    vectors = backend.encode(texts)
    for ticket, vector in zip(tickets, vectors):
        ticket.embedding = [float(x) for x in vector]
    return tickets


def enrich_batch(
    tickets: Iterable[Ticket],
    backend=None,
    ingest_ts: datetime | None = None,
) -> list[Ticket]:
    """Convenience wrapper: enrich, then optionally embed."""
    enriched = [enrich_ticket(t, ingest_ts=ingest_ts) for t in tickets]
    if backend is not None:
        attach_embeddings(enriched, backend)
    return enriched


def embedding_matrix(tickets: Sequence[Ticket]) -> np.ndarray:
    """Stack ticket embeddings into a matrix for in-memory similarity search."""
    if not tickets:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray([t.embedding for t in tickets], dtype=np.float32)
