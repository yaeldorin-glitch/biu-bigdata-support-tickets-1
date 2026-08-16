"""Embedding backends -- the heart of the graded AI capability.

Two interchangeable implementations behind one interface:

``SentenceTransformerBackend``
    ``paraphrase-multilingual-MiniLM-L12-v2``. Chosen because the corpus is 57%
    English and 43% German: a multilingual model puts both languages in a
    *shared* vector space, so a German query retrieves semantically equivalent
    English tickets. That cross-lingual recall is the single clearest win of
    semantic search over BM25 on this dataset, and it is the thing to demo.

``OfflineTfidfSvdBackend``
    Character- and word-ngram TF-IDF reduced to the same dimensionality with
    truncated SVD (i.e. LSA). No model download, no network, deterministic. It
    exists so the pipeline is never blocked on a 400MB download, and so the
    evaluation has a classical IR baseline to compare the neural model against.

Both emit L2-normalised vectors of identical width, so the Elasticsearch
``dense_vector`` mapping and the cosine kNN query are the same either way.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..config import AIConfig, get_settings
from .textproc import normalize, truncate

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Anything that turns text into fixed-width vectors."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so cosine similarity reduces to a dot product.

    Elasticsearch's ``cosine`` similarity normalises internally, but doing it
    here means the same vectors work for in-memory numpy search in the offline
    runner without a second code path.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SentenceTransformerBackend:
    """Neural sentence embeddings. Requires ``sentence-transformers``."""

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)
        actual = int(self._model.get_sentence_embedding_dimension())
        if actual != dim:
            logger.warning(
                "configured EMBEDDING_DIM=%s but %s produces %s; using %s. "
                "Recreate the Elasticsearch index if it was built with the old width.",
                dim, model_name, actual, actual,
            )
        self.dim = actual

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = [truncate(normalize(t), 1000) for t in texts]
        vectors = self._model.encode(
            cleaned, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


class OfflineTfidfSvdBackend:
    """TF-IDF + truncated SVD (LSA). Must be fitted before use.

    Word ngrams capture topical vocabulary; character ngrams give robustness to
    German compounding and to the misspellings common in support text.
    """

    def __init__(self, dim: int = 384) -> None:
        self.name = "offline:tfidf+svd"
        self.dim = dim
        self._pipeline = None

    def fit(self, corpus: Sequence[str]) -> "OfflineTfidfSvdBackend":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline

        cleaned = [normalize(t) for t in corpus if t and t.strip()]
        if not cleaned:
            raise ValueError("cannot fit the offline embedding backend on an empty corpus")

        # SVD cannot produce more components than min(n_samples, n_features) - 1.
        n_components = min(self.dim, max(2, len(cleaned) - 1))

        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.6,
                        max_features=40_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=40_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        self._pipeline = Pipeline(
            [
                ("features", features),
                ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ]
        )
        self._pipeline.fit(cleaned)

        # scikit-learn fits in float64. The SVD component matrix is
        # (n_components x n_features), so at 384 x 80k that is ~250MB of
        # float64 -- most of the serialised model, and far more precision than
        # cosine similarity over unit vectors needs. Halving it to float32 makes
        # the model small enough to ship to Spark executors comfortably.
        svd = self._pipeline.named_steps["svd"]
        svd.components_ = svd.components_.astype(np.float32)

        self.dim = n_components
        logger.info("fitted offline embedding backend: %s docs -> %s dims", len(cleaned), self.dim)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("OfflineTfidfSvdBackend.fit() must be called before encode()")
        cleaned = [normalize(t) if t else "" for t in texts]
        vectors = self._pipeline.transform(cleaned)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    # ---- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "dim": self.dim}, path)

    @classmethod
    def load(cls, path: Path) -> "OfflineTfidfSvdBackend":
        import joblib

        payload = joblib.load(path)
        backend = cls(dim=payload["dim"])
        backend._pipeline = payload["pipeline"]
        return backend


# --------------------------------------------------------------------------- #
# Factory + per-process singleton
# --------------------------------------------------------------------------- #

_LOCK = threading.Lock()
_INSTANCE: EmbeddingBackend | None = None


def build_backend(
    ai: AIConfig | None = None,
    corpus: Sequence[str] | None = None,
    offline_model_path: Path | None = None,
) -> EmbeddingBackend:
    """Construct a backend according to configuration.

    ``auto`` prefers the neural model and silently degrades to the offline one,
    which is what makes the demo robust on a machine with no internet access.
    """
    ai = ai or get_settings().ai
    choice = ai.embedding_backend.lower()

    if choice in {"sentence-transformers", "st", "auto"}:
        try:
            return SentenceTransformerBackend(ai.embedding_model, ai.embedding_dim)
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
            if choice != "auto":
                raise
            logger.warning(
                "sentence-transformers unavailable (%s); falling back to the offline backend",
                exc,
            )

    if offline_model_path and Path(offline_model_path).exists():
        return OfflineTfidfSvdBackend.load(Path(offline_model_path))

    if corpus is None:
        raise RuntimeError(
            "the offline embedding backend needs either a fitted model file or a corpus to fit on"
        )
    return OfflineTfidfSvdBackend(ai.embedding_dim).fit(corpus)


def get_backend(**kwargs) -> EmbeddingBackend:
    """Process-wide singleton.

    Spark creates one Python worker per executor core and calls the UDF once per
    row; without this, every row would reload the model.
    """
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = build_backend(**kwargs)
    return _INSTANCE


def set_backend(backend: EmbeddingBackend | None) -> None:
    """Inject a backend directly. Used by tests and by the offline runner."""
    global _INSTANCE
    _INSTANCE = backend


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one L2-normalised vector against a normalised matrix."""
    return matrix @ query
