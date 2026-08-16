"""Semantic and keyword search over the ticket corpus.

Two backends implement the same interface:

``InMemoryIndex``
    numpy dot-product over the normalised embedding matrix, plus a small BM25
    implementation for the keyword baseline. Used by the offline runner, the
    evaluation harness and the tests -- no Elasticsearch required.

``ElasticIndex``
    the production path: ``knn`` against the ``dense_vector`` field for semantic
    search, ``multi_match`` for the BM25 baseline, and an RRF-style blend for
    hybrid search.

Keeping the keyword baseline as a first-class citizen is deliberate: the point
of the AI capability is not that semantic search *exists*, it is that we can
show by how much it beats lexical matching, and where it does not.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from ..core.textproc import tokenize

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    ticket_id: str
    score: float
    text: str
    queue: str = ""
    priority: str = ""
    language: str = ""
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "score": round(self.score, 4),
            "text": self.text,
            "queue": self.queue,
            "priority": self.priority,
            "language": self.language,
            "tags": self.tags or [],
        }


class SearchIndex(Protocol):
    def semantic(self, query: str, k: int = 5, **filters) -> list[SearchHit]: ...
    def keyword(self, query: str, k: int = 5, **filters) -> list[SearchHit]: ...
    def hybrid(self, query: str, k: int = 5, **filters) -> list[SearchHit]: ...


# --------------------------------------------------------------------------- #
# BM25 (used only by the in-memory index; Elasticsearch has its own)
# --------------------------------------------------------------------------- #


class BM25:
    """Textbook Okapi BM25 over an inverted index.

    The postings list matters for more than tidiness: scoring by iterating every
    document per query term is O(|query| x |corpus|), which at 28k documents and
    200 benchmark queries is minutes of pure Python. Walking only the documents
    that actually contain each term makes it milliseconds, and it is how a real
    search engine does it.
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        # term -> (document indices, term frequencies), both as numpy arrays so
        # scoring a term is a vectorised operation over its postings.
        postings: dict[str, dict[int, int]] = {}
        for index, doc in enumerate(corpus_tokens):
            for term, freq in Counter(doc).items():
                postings.setdefault(term, {})[index] = freq

        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {
            term: (
                np.fromiter(entries.keys(), dtype=np.int32, count=len(entries)),
                np.fromiter(entries.values(), dtype=np.float32, count=len(entries)),
            )
            for term, entries in postings.items()
        }
        self.doc_freq = {term: len(entries) for term, entries in postings.items()}

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        # Standard BM25 idf with the +0.5 smoothing that keeps it positive.
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        terms = tokenize(query)
        if not terms or self.avgdl == 0:
            return out

        for term in set(terms):
            entry = self.postings.get(term)
            if entry is None:
                continue
            indices, freqs = entry
            idf = self._idf(term)
            denominator = freqs + self.k1 * (
                1 - self.b + self.b * self.doc_len[indices] / self.avgdl
            )
            np.add.at(out, indices, idf * (freqs * (self.k1 + 1)) / denominator)
        return out


# --------------------------------------------------------------------------- #
# In-memory index
# --------------------------------------------------------------------------- #


class InMemoryIndex:
    """Brute-force search over the whole corpus.

    Exact rather than approximate, which is the right trade at 28k documents and
    removes ANN recall as a confound when comparing semantic against BM25.
    """

    def __init__(self, tickets: Sequence[Any], backend) -> None:
        self.tickets = [t if isinstance(t, dict) else t.to_dict() for t in tickets]
        self.backend = backend
        self.texts = [t.get("text_redacted") or t.get("text") or "" for t in self.tickets]

        vectors = [t.get("embedding") or [] for t in self.tickets]
        widths = {len(v) for v in vectors}
        if not vectors:
            raise ValueError("cannot build an index over zero tickets")
        if widths != {len(vectors[0])} or not vectors[0]:
            # Ragged or empty vectors would otherwise produce an object-dtype
            # array and fail much later, inside the first similarity computation,
            # with an error that says nothing about the real cause.
            raise ValueError(
                "every ticket must carry a non-empty embedding of equal width; "
                f"got widths {sorted(widths)} -- did you forget attach_embeddings()?"
            )

        self.matrix = np.asarray(vectors, dtype=np.float32)
        self.bm25 = BM25([tokenize(t) for t in self.texts])

    # -- helpers ----------------------------------------------------------
    def _mask(self, **filters) -> np.ndarray:
        mask = np.ones(len(self.tickets), dtype=bool)
        for field, value in filters.items():
            if value in (None, "", []):
                continue
            mask &= np.array([t.get(field) == value for t in self.tickets], dtype=bool)
        return mask

    def _hits(self, scores: np.ndarray, k: int, mask: np.ndarray) -> list[SearchHit]:
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        out = []
        for index in top:
            if not np.isfinite(scores[index]):
                continue
            ticket = self.tickets[index]
            out.append(
                SearchHit(
                    ticket_id=ticket.get("ticket_id", ""),
                    score=float(scores[index]),
                    text=self.texts[index],
                    queue=ticket.get("queue", ""),
                    priority=ticket.get("priority", ""),
                    language=ticket.get("language", ""),
                    tags=ticket.get("tags", []),
                )
            )
        return out

    # -- public API -------------------------------------------------------
    def semantic(self, query: str, k: int = 5, **filters) -> list[SearchHit]:
        vector = self.backend.encode([query])[0]
        return self._hits(self.matrix @ vector, k, self._mask(**filters))

    def keyword(self, query: str, k: int = 5, **filters) -> list[SearchHit]:
        return self._hits(self.bm25.scores(query), k, self._mask(**filters))

    def hybrid(self, query: str, k: int = 5, rrf_k: int = 60, **filters) -> list[SearchHit]:
        """Reciprocal rank fusion of the two rankings.

        RRF is used instead of score averaging because BM25 and cosine live on
        incomparable scales; fusing *ranks* sidesteps the normalisation problem
        entirely.
        """
        pool = min(k * 10, len(self.tickets))
        semantic = self.semantic(query, pool, **filters)
        keyword = self.keyword(query, pool, **filters)

        fused: dict[str, float] = {}
        holder: dict[str, SearchHit] = {}
        for ranking in (semantic, keyword):
            for rank, hit in enumerate(ranking, start=1):
                fused[hit.ticket_id] = fused.get(hit.ticket_id, 0.0) + 1.0 / (rrf_k + rank)
                holder.setdefault(hit.ticket_id, hit)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        out = []
        for ticket_id, score in ordered:
            hit = holder[ticket_id]
            out.append(
                SearchHit(ticket_id, score, hit.text, hit.queue, hit.priority, hit.language, hit.tags)
            )
        return out


# --------------------------------------------------------------------------- #
# Elasticsearch index
# --------------------------------------------------------------------------- #


class ElasticIndex:
    """Search against a live Elasticsearch cluster."""

    def __init__(self, client, index: str, backend) -> None:
        self.client = client
        self.index = index
        self.backend = backend

    @staticmethod
    def _filters(**filters) -> list[dict[str, Any]]:
        return [{"term": {k: v}} for k, v in filters.items() if v not in (None, "", [])]

    def _to_hits(self, response: dict[str, Any]) -> list[SearchHit]:
        out = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            out.append(
                SearchHit(
                    ticket_id=src.get("ticket_id", hit.get("_id", "")),
                    score=float(hit.get("_score") or 0.0),
                    text=src.get("text_redacted") or src.get("text", ""),
                    queue=src.get("queue", ""),
                    priority=src.get("priority", ""),
                    language=src.get("language", ""),
                    tags=src.get("tags", []),
                )
            )
        return out

    def semantic(self, query: str, k: int = 5, **filters) -> list[SearchHit]:
        vector = [float(x) for x in self.backend.encode([query])[0]]
        body: dict[str, Any] = {
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": k,
                # Scanning 10x the requested neighbours per shard is the usual
                # recall/latency compromise for HNSW.
                "num_candidates": max(50, k * 10),
            },
            "_source": {"excludes": ["embedding"]},
            "size": k,
        }
        clauses = self._filters(**filters)
        if clauses:
            body["knn"]["filter"] = {"bool": {"filter": clauses}}
        return self._to_hits(self.client.search(index=self.index, body=body))

    def keyword(self, query: str, k: int = 5, **filters) -> list[SearchHit]:
        must: dict[str, Any] = {
            "multi_match": {
                "query": query,
                "fields": ["subject^2", "text_redacted", "text", "answer"],
            }
        }
        body = {
            "query": {"bool": {"must": [must], "filter": self._filters(**filters)}},
            "_source": {"excludes": ["embedding"]},
            "size": k,
        }
        return self._to_hits(self.client.search(index=self.index, body=body))

    def hybrid(self, query: str, k: int = 5, rrf_k: int = 60, **filters) -> list[SearchHit]:
        pool = k * 5
        semantic = self.semantic(query, pool, **filters)
        keyword = self.keyword(query, pool, **filters)

        fused: dict[str, float] = {}
        holder: dict[str, SearchHit] = {}
        for ranking in (semantic, keyword):
            for rank, hit in enumerate(ranking, start=1):
                fused[hit.ticket_id] = fused.get(hit.ticket_id, 0.0) + 1.0 / (rrf_k + rank)
                holder.setdefault(hit.ticket_id, hit)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            SearchHit(
                tid,
                score,
                holder[tid].text,
                holder[tid].queue,
                holder[tid].priority,
                holder[tid].language,
                holder[tid].tags,
            )
            for tid, score in ordered
        ]
