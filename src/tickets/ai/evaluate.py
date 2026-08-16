"""Evaluation harness -- the evidence behind the AI claims.

The brief weights "AI capability (correctness, depth, and your understanding)"
at 25% and "results and insights" at 15%. Asserting that semantic search is
better is not evidence; this module measures it.

Two experiments:

**Routing.** Four approaches predict the ``queue`` label on a held-out split:
majority class, keyword rules, nearest centroid, and similarity-weighted kNN.
Reported as accuracy plus macro-F1, because accuracy alone flatters any model
that simply learns the majority class.

**Retrieval.** Semantic vs BM25 vs hybrid, scored with two *proxy* relevance
signals, since the corpus has no human relevance judgements:

* ``queue_purity@k`` -- share of the top k sharing the query ticket's queue.
* ``tag_overlap`` P@k / MRR -- a result is relevant if it shares >= 2 tags with
  the query ticket.

Both proxies are imperfect and we say so in the report: they reward topical
similarity, which is what routing and deduplication need, but they would
under-credit a system that retrieves a *solution* phrased differently from the
problem. Treat the numbers as comparative, not absolute.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.classify import classify_queue
from .router import CentroidRouter, KnnRouter
from .search import InMemoryIndex

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def accuracy(true: Sequence[str], pred: Sequence[str]) -> float:
    if not true:
        return 0.0
    return sum(1 for t, p in zip(true, pred) if t == p) / len(true)


def macro_f1(true: Sequence[str], pred: Sequence[str]) -> float:
    """Unweighted mean per-class F1.

    The queue distribution is heavily skewed (29% Technical Support, 1.4%
    General Inquiry), so macro-F1 is the honest headline number.
    """
    labels = sorted(set(true) | set(pred))
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def per_class_report(true: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in sorted(set(true)):
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return out


# --------------------------------------------------------------------------- #
# Routing experiment
# --------------------------------------------------------------------------- #


@dataclass
class RoutingResult:
    method: str
    accuracy: float
    macro_f1: float
    n_test: int
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "n_test": self.n_test,
        }


def evaluate_routing(
    tickets: Sequence[Any],
    test_fraction: float = 0.25,
    seed: int = 42,
    knn_k: int = 15,
) -> list[RoutingResult]:
    """Compare four routing strategies on a stratified held-out split."""
    rows = [t if isinstance(t, dict) else t.to_dict() for t in tickets]
    rows = [r for r in rows if r.get("queue") and r.get("embedding")]
    if len(rows) < 50:
        raise ValueError("need at least 50 labelled, embedded tickets to evaluate routing")

    # Stratify so every queue appears in both splits; the smallest queue has
    # only 405 rows and a random split can starve it.
    rng = random.Random(seed)
    by_queue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_queue[row["queue"]].append(row)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for queue, group in by_queue.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * test_fraction))
        test.extend(shuffled[:cut])
        train.extend(shuffled[cut:])

    train_matrix = np.asarray([r["embedding"] for r in train], dtype=np.float32)
    train_labels = [r["queue"] for r in train]
    test_matrix = np.asarray([r["embedding"] for r in test], dtype=np.float32)
    test_labels = [r["queue"] for r in test]

    results: list[RoutingResult] = []

    # 1. Majority class -- the floor any real method must clear.
    majority = Counter(train_labels).most_common(1)[0][0]
    predictions = [majority] * len(test_labels)
    results.append(
        RoutingResult(
            "majority_class", accuracy(test_labels, predictions),
            macro_f1(test_labels, predictions), len(test_labels),
        )
    )

    # 2. Keyword rules.
    predictions = [classify_queue(r.get("text_redacted") or r.get("text", "")).label for r in test]
    results.append(
        RoutingResult(
            "keyword_rules", accuracy(test_labels, predictions),
            macro_f1(test_labels, predictions), len(test_labels),
            per_class_report(test_labels, predictions),
        )
    )

    # 3. Nearest centroid over embeddings.
    centroid = CentroidRouter().fit(train_matrix, train_labels)
    predictions = [p.label for p in centroid.predict(test_matrix)]
    results.append(
        RoutingResult(
            "embedding_centroid", accuracy(test_labels, predictions),
            macro_f1(test_labels, predictions), len(test_labels),
        )
    )

    # 4. Similarity-weighted kNN over embeddings.
    knn = KnnRouter(k=knn_k).fit(train_matrix, train_labels)
    predictions = [p.label for p in knn.predict(test_matrix)]
    results.append(
        RoutingResult(
            f"embedding_knn_k{knn_k}", accuracy(test_labels, predictions),
            macro_f1(test_labels, predictions), len(test_labels),
            per_class_report(test_labels, predictions),
        )
    )

    return results


# --------------------------------------------------------------------------- #
# Retrieval experiment
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalResult:
    method: str
    queue_purity_at_k: float
    precision_at_k: float
    mrr: float
    n_queries: int
    k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "queue_purity_at_k": round(self.queue_purity_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "n_queries": self.n_queries,
            "k": self.k,
        }


def evaluate_retrieval(
    tickets: Sequence[Any],
    index: InMemoryIndex,
    n_queries: int = 200,
    k: int = 5,
    seed: int = 42,
    min_shared_tags: int = 2,
) -> list[RetrievalResult]:
    """Compare semantic, keyword and hybrid retrieval on proxy relevance.

    Each query is an existing ticket's ``subject`` (falling back to a truncated
    body). The query ticket itself is excluded from its own results, otherwise
    every method scores a trivial 1.0 at rank 1.
    """
    rows = [t if isinstance(t, dict) else t.to_dict() for t in tickets]
    candidates = [r for r in rows if (r.get("subject") or "").strip() and len(r.get("tags") or []) >= 2]
    if not candidates:
        raise ValueError("no tickets with both a subject and >=2 tags to use as queries")

    rng = random.Random(seed)
    queries = rng.sample(candidates, min(n_queries, len(candidates)))

    methods = {"semantic": index.semantic, "keyword": index.keyword, "hybrid": index.hybrid}
    results: list[RetrievalResult] = []
    by_id = {r.get("ticket_id"): r for r in rows}

    for name, retrieve in methods.items():
        purity: list[float] = []
        precision: list[float] = []
        reciprocal: list[float] = []

        for query_row in queries:
            query_text = query_row["subject"]
            query_tags = set(query_row.get("tags") or [])
            query_id = query_row.get("ticket_id")

            # Fetch k+1 so that dropping the self-match still leaves k results.
            hits = [h for h in retrieve(query_text, k=k + 1) if h.ticket_id != query_id][:k]
            if not hits:
                continue

            relevant_flags = []
            for hit in hits:
                hit_row = by_id.get(hit.ticket_id, {})
                shared = len(query_tags & set(hit_row.get("tags") or []))
                relevant_flags.append(1 if shared >= min_shared_tags else 0)

            purity.append(sum(1 for h in hits if h.queue == query_row.get("queue")) / len(hits))
            precision.append(sum(relevant_flags) / len(relevant_flags))
            first = next((i for i, flag in enumerate(relevant_flags, start=1) if flag), None)
            reciprocal.append(1.0 / first if first else 0.0)

        results.append(
            RetrievalResult(
                method=name,
                queue_purity_at_k=float(np.mean(purity)) if purity else 0.0,
                precision_at_k=float(np.mean(precision)) if precision else 0.0,
                mrr=float(np.mean(reciprocal)) if reciprocal else 0.0,
                n_queries=len(purity),
                k=k,
            )
        )
    return results


def cross_lingual_probe(index: InMemoryIndex, queries: Sequence[tuple[str, str]], k: int = 5) -> list[dict[str, Any]]:
    """Show what a multilingual embedding buys over keyword matching.

    Each entry is ``(query_text, query_language)``. We report how many of the
    top-k results are in the *other* language. BM25 scores near zero here by
    construction -- German query terms do not occur in English tickets -- so
    this is the clearest single demo of why the embedding is worth its cost.
    """
    out = []
    for text, language in queries:
        other = "en" if language == "de" else "de"
        semantic = index.semantic(text, k=k)
        keyword = index.keyword(text, k=k)
        out.append(
            {
                "query": text,
                "query_language": language,
                "semantic_cross_language_hits": sum(1 for h in semantic if h.language == other),
                "keyword_cross_language_hits": sum(1 for h in keyword if h.language == other),
                "k": k,
                "semantic_top_hit": semantic[0].to_dict() if semantic else None,
                "keyword_top_hit": keyword[0].to_dict() if keyword else None,
            }
        )
    return out
