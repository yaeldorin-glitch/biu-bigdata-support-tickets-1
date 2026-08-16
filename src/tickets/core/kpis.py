"""KPI computation over enriched tickets.

Expressed over plain dicts so the identical logic can run on a Spark DataFrame's
``collect()`` output, on the offline runner's in-memory list, or in a test. The
Spark batch job (:mod:`tickets.spark.batch_kpis`) computes the same aggregates
with native DataFrame operations for scale; this module is the reference
implementation and the thing the tests pin.

Every KPI is emitted in one long-format shape -- ``kpi``, ``dimension``,
``bucket``, ``value``, ``count`` -- so they all land in a single Elasticsearch
index and a single Kibana data view.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


def _rows(tickets: Iterable[Any]) -> list[dict[str, Any]]:
    """Accept Ticket dataclasses or plain dicts."""
    out = []
    for t in tickets:
        out.append(t if isinstance(t, dict) else t.to_dict())
    return out


def _emit(kpi: str, dimension: str, bucket: str, value: float, count: int) -> dict[str, Any]:
    return {
        "kpi": kpi,
        "dimension": dimension,
        "bucket": str(bucket),
        "value": round(float(value), 6),
        "count": int(count),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def volume_by(tickets: Sequence[Any], field: str) -> list[dict[str, Any]]:
    """Ticket counts and shares for a categorical field."""
    rows = _rows(tickets)
    total = len(rows) or 1
    counts = Counter(r.get(field) or "unknown" for r in rows)
    return [
        _emit("volume_share", field, bucket, count / total, count)
        for bucket, count in counts.most_common()
    ]


def sentiment_by(tickets: Sequence[Any], field: str) -> list[dict[str, Any]]:
    """Mean sentiment score per bucket -- surfaces which queues run hot."""
    rows = _rows(tickets)
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grouped[r.get(field) or "unknown"].append(float(r.get("sentiment_score") or 0.0))
    out = [
        _emit("avg_sentiment", field, bucket, sum(scores) / len(scores), len(scores))
        for bucket, scores in grouped.items()
        if scores
    ]
    return sorted(out, key=lambda d: d["value"])


def negative_rate_by(tickets: Sequence[Any], field: str) -> list[dict[str, Any]]:
    """Share of tickets labelled negative per bucket."""
    rows = _rows(tickets)
    grouped: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        grouped[r.get(field) or "unknown"].append(1 if r.get("sentiment") == "negative" else 0)
    out = [
        _emit("negative_rate", field, bucket, sum(flags) / len(flags), len(flags))
        for bucket, flags in grouped.items()
        if flags
    ]
    return sorted(out, key=lambda d: d["value"], reverse=True)


def routing_accuracy(tickets: Sequence[Any], predicted_field: str = "predicted_queue") -> list[dict[str, Any]]:
    """Accuracy of predicted queue against the dataset's own label.

    Reported overall and per true queue, because a single headline number hides
    the fact that a classifier can be excellent on the majority class and
    useless on the tail -- which is exactly what a support team cares about.
    """
    rows = [r for r in _rows(tickets) if r.get("queue") and r.get(predicted_field)]
    if not rows:
        return []

    correct = sum(1 for r in rows if r[predicted_field] == r["queue"])
    out = [_emit("routing_accuracy", "overall", "all", correct / len(rows), len(rows))]

    per_queue: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        per_queue[r["queue"]].append(1 if r[predicted_field] == r["queue"] else 0)
    for queue, flags in per_queue.items():
        out.append(_emit("routing_accuracy", "queue", queue, sum(flags) / len(flags), len(flags)))
    return out


def language_mismatch_rate(tickets: Sequence[Any]) -> list[dict[str, Any]]:
    """How often the declared ``language`` disagrees with detected language.

    A data-quality KPI. It matters operationally: routing rules and canned
    replies keyed off a wrong language label send German customers English
    templates.
    """
    rows = [r for r in _rows(tickets) if r.get("detected_language") not in (None, "", "unknown")]
    if not rows:
        return []

    mismatched = sum(1 for r in rows if r.get("language_mismatch"))
    out = [_emit("language_mismatch_rate", "overall", "all", mismatched / len(rows), len(rows))]

    per_lang: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        per_lang[r.get("language") or "unknown"].append(1 if r.get("language_mismatch") else 0)
    for lang, flags in per_lang.items():
        out.append(
            _emit("language_mismatch_rate", "declared_language", lang, sum(flags) / len(flags), len(flags))
        )
    return out


def pii_exposure(tickets: Sequence[Any]) -> list[dict[str, Any]]:
    """Share of tickets in which each PII kind was detected."""
    rows = _rows(tickets)
    total = len(rows) or 1
    counts: Counter[str] = Counter()
    any_pii = 0
    for r in rows:
        kinds = r.get("pii_found") or []
        if kinds:
            any_pii += 1
        counts.update(kinds)
    out = [_emit("pii_rate", "overall", "any", any_pii / total, any_pii)]
    out.extend(_emit("pii_rate", "kind", kind, count / total, count) for kind, count in counts.most_common())
    return out


def top_tags(tickets: Sequence[Any], limit: int = 20) -> list[dict[str, Any]]:
    """Most frequent tags across the ragged tag_1..tag_8 columns."""
    rows = _rows(tickets)
    total = len(rows) or 1
    counts: Counter[str] = Counter()
    for r in rows:
        counts.update(r.get("tags") or [])
    return [
        _emit("tag_frequency", "tag", tag, count / total, count)
        for tag, count in counts.most_common(limit)
    ]


def priority_mix_by_queue(tickets: Sequence[Any]) -> list[dict[str, Any]]:
    """Share of high-priority tickets per queue -- where the pressure sits."""
    rows = _rows(tickets)
    grouped: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("queue"):
            grouped[r["queue"]].append(1 if r.get("priority") == "high" else 0)
    out = [
        _emit("high_priority_share", "queue", queue, sum(flags) / len(flags), len(flags))
        for queue, flags in grouped.items()
        if flags
    ]
    return sorted(out, key=lambda d: d["value"], reverse=True)


def compute_all(tickets: Sequence[Any]) -> list[dict[str, Any]]:
    """Every KPI, flattened into one list ready for bulk indexing."""
    out: list[dict[str, Any]] = []
    for field in ("queue", "type", "priority", "language"):
        out.extend(volume_by(tickets, field))
    out.extend(sentiment_by(tickets, "queue"))
    out.extend(negative_rate_by(tickets, "queue"))
    out.extend(priority_mix_by_queue(tickets))
    out.extend(routing_accuracy(tickets))
    out.extend(language_mismatch_rate(tickets))
    out.extend(pii_exposure(tickets))
    out.extend(top_tags(tickets))
    return out


def summarize(tickets: Sequence[Any]) -> dict[str, Any]:
    """Compact headline numbers, used by the report and the LLM narrative."""
    rows = _rows(tickets)
    accuracy = routing_accuracy(rows)
    mismatch = language_mismatch_rate(rows)
    negatives = sum(1 for r in rows if r.get("sentiment") == "negative")

    return {
        "total_tickets": len(rows),
        "languages": dict(Counter(r.get("language") for r in rows)),
        "queues": dict(Counter(r.get("queue") for r in rows).most_common()),
        "types": dict(Counter(r.get("type") for r in rows)),
        "priorities": dict(Counter(r.get("priority") for r in rows)),
        "negative_share": round(negatives / (len(rows) or 1), 4),
        "routing_accuracy_overall": next(
            (r["value"] for r in accuracy if r["dimension"] == "overall"), None
        ),
        "language_mismatch_overall": next(
            (r["value"] for r in mismatch if r["dimension"] == "overall"), None
        ),
        "tickets_with_pii": sum(1 for r in rows if r.get("pii_found")),
    }
