"""Embedding-based ticket routing (k-nearest-neighbour classifier).

The operational question behind the whole project: *given the free text of a new
ticket, which queue should it go to?* This module answers it with the same
embeddings that power semantic search -- no separate model, no extra training
run, which is the cheapest possible way to turn the AI capability into business
value.

Two variants, both fitted on the labelled portion of the corpus:

``kNN``
    majority vote over the k most similar labelled tickets, weighted by
    similarity. Non-parametric, so it handles the long tail of small queues
    better than a centroid method.

``Centroid`` (nearest class mean)
    one vector per queue. Faster and smaller, and a useful ablation: if kNN does
    not beat it, the extra lookup cost is not justified.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RoutingPrediction:
    label: str
    confidence: float
    neighbours: list[tuple[str, float]]


class KnnRouter:
    """Similarity-weighted k-NN over L2-normalised embeddings."""

    def __init__(self, k: int = 15) -> None:
        self.k = k
        self._matrix: np.ndarray | None = None
        self._labels: list[str] = []

    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> "KnnRouter":
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")
        if len(labels) != matrix.shape[0]:
            raise ValueError("embeddings and labels must be the same length")
        self._matrix = matrix
        self._labels = list(labels)
        return self

    def predict_one(self, vector: np.ndarray) -> RoutingPrediction:
        if self._matrix is None:
            raise RuntimeError("KnnRouter.fit() must be called before predict")

        similarities = self._matrix @ np.asarray(vector, dtype=np.float32)
        k = min(self.k, len(self._labels))
        top = np.argpartition(-similarities, k - 1)[:k]
        top = top[np.argsort(-similarities[top])]

        votes: dict[str, float] = defaultdict(float)
        for index in top:
            # Clamp: cosine can be slightly negative, and a negative vote would
            # perversely push probability mass toward unrelated classes.
            votes[self._labels[index]] += max(float(similarities[index]), 0.0)

        total = sum(votes.values())
        if total <= 0:
            return RoutingPrediction(self._labels[top[0]], 0.0, [])

        best = max(votes, key=lambda label: votes[label])
        neighbours = [(self._labels[i], round(float(similarities[i]), 4)) for i in top[:5]]
        return RoutingPrediction(best, round(votes[best] / total, 4), neighbours)

    def predict(self, embeddings: np.ndarray) -> list[RoutingPrediction]:
        """Batched prediction.

        One (n_queries x n_train) matmul instead of n_queries separate ones.
        BLAS handles the big multiply far better than a Python loop -- on the
        28k-ticket corpus this is the difference between minutes and seconds.
        """
        if self._matrix is None:
            raise RuntimeError("KnnRouter.fit() must be called before predict")

        queries = np.asarray(embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]

        similarities = queries @ self._matrix.T
        k = min(self.k, len(self._labels))
        top_indices = np.argpartition(-similarities, k - 1, axis=1)[:, :k]

        out: list[RoutingPrediction] = []
        labels = np.asarray(self._labels)
        for row, candidates in zip(similarities, top_indices):
            order = candidates[np.argsort(-row[candidates])]
            scores = np.maximum(row[order], 0.0)

            votes: dict[str, float] = defaultdict(float)
            for label, score in zip(labels[order], scores):
                votes[str(label)] += float(score)

            total = sum(votes.values())
            if total <= 0:
                out.append(RoutingPrediction(str(labels[order[0]]), 0.0, []))
                continue

            best = max(votes, key=lambda label: votes[label])
            neighbours = [(str(labels[i]), round(float(row[i]), 4)) for i in order[:5]]
            out.append(RoutingPrediction(best, round(votes[best] / total, 4), neighbours))
        return out


class CentroidRouter:
    """Nearest class mean. One vector per queue."""

    def __init__(self) -> None:
        self._centroids: np.ndarray | None = None
        self._classes: list[str] = []

    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> "CentroidRouter":
        matrix = np.asarray(embeddings, dtype=np.float32)
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for vector, label in zip(matrix, labels):
            grouped[label].append(vector)

        self._classes = sorted(grouped)
        centroids = np.stack([np.mean(grouped[c], axis=0) for c in self._classes])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._centroids = (centroids / norms).astype(np.float32)
        return self

    def predict_one(self, vector: np.ndarray) -> RoutingPrediction:
        if self._centroids is None:
            raise RuntimeError("CentroidRouter.fit() must be called before predict")
        similarities = self._centroids @ np.asarray(vector, dtype=np.float32)
        best = int(np.argmax(similarities))

        # Softmax over cosine similarities gives a usable confidence. The
        # temperature is low because cosine values are bunched near each other.
        exponent = np.exp((similarities - similarities.max()) / 0.05)
        probabilities = exponent / exponent.sum()

        return RoutingPrediction(
            self._classes[best],
            round(float(probabilities[best]), 4),
            [(self._classes[i], round(float(similarities[i]), 4)) for i in np.argsort(-similarities)[:5]],
        )

    def predict(self, embeddings: np.ndarray) -> list[RoutingPrediction]:
        return [self.predict_one(v) for v in np.asarray(embeddings, dtype=np.float32)]
