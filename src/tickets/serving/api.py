"""REST API over the serving layer -- the demo surface.

Endpoints:

    GET  /health              liveness + which backends are active
    GET  /search              semantic | keyword | hybrid search, side by side
    POST /ask                 RAG question answering, grounded in retrieved tickets
    POST /route               predict the queue for a new, unseen ticket
    GET  /kpis                computed KPI aggregates
    GET  /compare             one query, all three retrieval modes, for the demo

The API works against either backend: Elasticsearch when the stack is up, or an
in-memory index built from the CSV when it is not (``--offline``). That means the
demo cannot be derailed by a container failing to start.

Run it with::

    uvicorn tickets.serving.api:app --reload            # needs Elasticsearch
    OFFLINE_API=1 uvicorn tickets.serving.api:app       # self-contained
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from ..config import RAW_CSV, get_settings

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "fastapi and pydantic are required to run the API: pip install fastapi uvicorn pydantic"
    ) from exc


# --------------------------------------------------------------------------- #
# Request/response models
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=20)
    mode: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    language: str | None = None
    queue: str | None = None


class RouteRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=8000)
    k: int = Field(15, ge=1, le=50)


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #


class State:
    """Holds whichever index and backend the app was started with."""

    def __init__(self) -> None:
        self.index = None
        self.backend = None
        self.router = None
        self.mode = "uninitialised"
        self.report: dict[str, Any] | None = None


state = State()
app = FastAPI(
    title="Customer IT Support Ticket Intelligence",
    description="Semantic search, RAG and routing over a multilingual support ticket corpus.",
    version="1.0.0",
)


def _init_offline(limit: int | None = None) -> None:
    """Build an in-memory index straight from the CSV."""
    import numpy as np

    from ..ai.router import KnnRouter
    from ..ai.search import InMemoryIndex
    from ..core.embeddings import build_backend, set_backend
    from ..core.enrich import attach_embeddings, enrich_ticket
    from ..offline_pipeline import load_tickets

    settings = get_settings()
    limit = limit if limit is not None else int(os.getenv("OFFLINE_API_LIMIT", "5000"))

    tickets, _ = load_tickets(RAW_CSV, limit)
    for ticket in tickets:
        enrich_ticket(ticket)

    backend = build_backend(settings.ai, corpus=[t.text_redacted or t.text for t in tickets])
    set_backend(backend)
    attach_embeddings(tickets, backend)

    state.backend = backend
    state.index = InMemoryIndex(tickets, backend)
    labelled = [t for t in tickets if t.queue]
    state.router = KnnRouter(k=15).fit(
        np.asarray([t.embedding for t in labelled], dtype="float32"),
        [t.queue for t in labelled],
    )
    state.mode = f"offline ({len(tickets)} tickets, {backend.name})"
    logger.info("API ready in offline mode: %s", state.mode)


def _init_elastic() -> None:
    from ..ai.search import ElasticIndex
    from ..core.embeddings import build_backend, set_backend
    from .es_client import get_client

    settings = get_settings()
    client = get_client(settings)

    # The neural backend needs no corpus; the offline one must load the model
    # that the pipeline fitted, or its vectors will not match what was indexed.
    model_path = os.getenv("OFFLINE_EMBEDDING_MODEL")
    backend = build_backend(settings.ai, offline_model_path=model_path)
    set_backend(backend)

    state.backend = backend
    state.index = ElasticIndex(client, settings.elastic.index_tickets, backend)
    state.mode = f"elasticsearch ({settings.elastic.url}, {backend.name})"
    logger.info("API ready against Elasticsearch: %s", state.mode)


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(level=logging.INFO)
    if os.getenv("OFFLINE_API", "").lower() in {"1", "true", "yes"}:
        _init_offline()
        return
    try:
        _init_elastic()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Elasticsearch unavailable (%s); starting in offline mode", exc)
        _init_offline()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok" if state.index is not None else "initialising",
        "index_mode": state.mode,
        "embedding_backend": getattr(state.backend, "name", None),
        "embedding_dim": getattr(state.backend, "dim", None),
        "llm_provider": settings.ai.llm_provider,
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="natural language query"),
    k: int = Query(5, ge=1, le=50),
    mode: Literal["semantic", "keyword", "hybrid"] = "semantic",
    language: str | None = None,
    queue: str | None = None,
) -> dict[str, Any]:
    if state.index is None:
        raise HTTPException(503, "index not ready")

    filters = {key: value for key, value in (("language", language), ("queue", queue)) if value}
    retrieve = {"semantic": state.index.semantic, "keyword": state.index.keyword}.get(
        mode, state.index.hybrid
    )
    hits = retrieve(q, k=k, **filters)
    return {"query": q, "mode": mode, "count": len(hits), "results": [h.to_dict() for h in hits]}


@app.get("/compare")
def compare(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)) -> dict[str, Any]:
    """Run one query through all three modes.

    This is the endpoint to demo: it makes the difference between lexical and
    semantic matching visible on a single screen, which no amount of describing
    it does.
    """
    if state.index is None:
        raise HTTPException(503, "index not ready")
    return {
        "query": q,
        "semantic": [h.to_dict() for h in state.index.semantic(q, k=k)],
        "keyword": [h.to_dict() for h in state.index.keyword(q, k=k)],
        "hybrid": [h.to_dict() for h in state.index.hybrid(q, k=k)],
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    from ..ai.rag import answer_question

    if state.index is None:
        raise HTTPException(503, "index not ready")

    filters = {
        key: value
        for key, value in (("language", request.language), ("queue", request.queue))
        if value
    }
    answer = answer_question(
        request.question, state.index, k=request.k, mode=request.mode, **filters
    )
    return answer.to_dict()


@app.post("/route")
def route(request: RouteRequest) -> dict[str, Any]:
    """Predict the queue for an unseen ticket.

    Returns the rule-based prediction alongside the kNN one so the improvement
    is visible per request, not just in the aggregate evaluation.
    """
    from ..core.classify import classify_priority, classify_queue
    from ..core.pii import redact
    from ..core.sentiment import score_sentiment

    if state.backend is None:
        raise HTTPException(503, "index not ready")

    redaction = redact(request.text)
    text = redaction.text

    response: dict[str, Any] = {
        "pii_found": redaction.kinds,
        "rule_based": {
            "queue": classify_queue(text).label,
            "priority": classify_priority(text).label,
        },
        "sentiment": score_sentiment(text).label,
    }

    if state.router is not None:
        vector = state.backend.encode([text])[0]
        prediction = state.router.predict_one(vector)
        response["knn"] = {
            "queue": prediction.label,
            "confidence": prediction.confidence,
            "nearest_neighbours": prediction.neighbours,
        }
    elif state.index is not None:
        # Elasticsearch mode: vote over the queues of the nearest documents.
        from collections import Counter

        hits = state.index.semantic(text, k=request.k)
        votes = Counter(h.queue for h in hits if h.queue)
        if votes:
            label, count = votes.most_common(1)[0]
            response["knn"] = {
                "queue": label,
                "confidence": round(count / sum(votes.values()), 4),
                "nearest_neighbours": [(h.queue, round(h.score, 4)) for h in hits[:5]],
            }

    return response


@app.get("/kpis")
def kpis(kpi: str | None = None, limit: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    """KPI aggregates, from Elasticsearch when available, else from report.json."""
    import json
    from pathlib import Path

    settings = get_settings()
    try:
        from .es_client import get_client

        client = get_client(settings)
        query = {"term": {"kpi": kpi}} if kpi else {"match_all": {}}
        response = client.search(index=settings.elastic.index_kpis, query=query, size=limit)
        return {
            "source": "elasticsearch",
            "results": [hit["_source"] for hit in response["hits"]["hits"]],
        }
    except Exception:  # noqa: BLE001
        report_path = Path(os.getenv("OUTPUT_DIR", "output")) / "kpis.json"
        if not report_path.exists():
            raise HTTPException(
                503, "no KPIs available: run the batch job or `python -m tickets.offline_pipeline`"
            )
        rows = json.loads(report_path.read_text(encoding="utf-8"))
        if kpi:
            rows = [r for r in rows if r.get("kpi") == kpi]
        return {"source": str(report_path), "results": rows[:limit]}
