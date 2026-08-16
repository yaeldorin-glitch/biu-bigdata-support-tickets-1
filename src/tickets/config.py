"""Central configuration.

Every tunable is read from an environment variable with a sane default, so the
same code runs unchanged in three places:

* on a laptop with nothing installed  (offline mode, see ``EMBEDDING_BACKEND``)
* inside the docker-compose stack     (service names resolve on the compose network)
* against a remote cluster            (point the URLs elsewhere)

Nothing here imports Spark, Kafka or Elasticsearch, so this module is safe to
import from tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_CSV = Path(os.getenv("RAW_CSV", DATA_DIR / "raw" / "tickets.csv"))
SAMPLE_CSV = Path(os.getenv("SAMPLE_CSV", DATA_DIR / "sample" / "tickets_sample.csv"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Kafka
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:9092"))
    topic_raw: str = field(default_factory=lambda: _env("KAFKA_TOPIC_RAW", "tickets.raw"))
    topic_dlq: str = field(default_factory=lambda: _env("KAFKA_TOPIC_DLQ", "tickets.dlq"))
    # How many messages/second the producer emits. Keeps the demo watchable
    # instead of dumping 28k rows in two seconds.
    produce_rate: float = field(default_factory=lambda: float(_env("PRODUCE_RATE", "200")))


# --------------------------------------------------------------------------- #
# MinIO / S3A object store
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjectStoreConfig:
    endpoint: str = field(default_factory=lambda: _env("MINIO_ENDPOINT", "http://localhost:9000"))
    access_key: str = field(default_factory=lambda: _env("MINIO_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: _env("MINIO_SECRET_KEY", "minioadmin"))
    bucket: str = field(default_factory=lambda: _env("MINIO_BUCKET", "lake"))

    @property
    def bronze_path(self) -> str:
        return f"s3a://{self.bucket}/bronze/tickets"

    @property
    def silver_path(self) -> str:
        return f"s3a://{self.bucket}/silver/tickets_enriched"

    @property
    def gold_path(self) -> str:
        return f"s3a://{self.bucket}/gold/kpis"

    @property
    def checkpoint_path(self) -> str:
        return f"s3a://{self.bucket}/_checkpoints"


# --------------------------------------------------------------------------- #
# Elasticsearch
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ElasticConfig:
    url: str = field(default_factory=lambda: _env("ES_URL", "http://localhost:9200"))
    index_tickets: str = field(default_factory=lambda: _env("ES_INDEX_TICKETS", "tickets"))
    index_kpis: str = field(default_factory=lambda: _env("ES_INDEX_KPIS", "ticket_kpis"))
    user: str | None = field(default_factory=lambda: os.getenv("ES_USER"))
    password: str | None = field(default_factory=lambda: os.getenv("ES_PASSWORD"))

    @property
    def host(self) -> str:
        """host:port form, which is what the Spark ES connector expects."""
        return self.url.replace("http://", "").replace("https://", "")


# --------------------------------------------------------------------------- #
# AI / embeddings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AIConfig:
    # "sentence-transformers" -> real neural embeddings (needs the model downloaded).
    # "offline"               -> deterministic TF-IDF + SVD embeddings, zero downloads.
    # "auto"                  -> try sentence-transformers, fall back to offline.
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    # 384 matches the MiniLM family. The offline backend is projected to the same
    # width so the Elasticsearch mapping never has to change between backends.
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))

    # LLM provider for enrichment / RAG: "none" | "anthropic" | "openai" | "ollama"
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-sonnet-4-20250514"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 512))
    # Cap on how many tickets get the (slow, possibly paid) LLM treatment.
    llm_sample_size: int = field(default_factory=lambda: _env_int("LLM_SAMPLE_SIZE", 200))

    # Refuse to send text to a hosted API unless it passed the redactor.
    require_redaction_before_llm: bool = field(
        default_factory=lambda: _env_bool("REQUIRE_REDACTION_BEFORE_LLM", True)
    )


@dataclass(frozen=True)
class Settings:
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    store: ObjectStoreConfig = field(default_factory=ObjectStoreConfig)
    elastic: ElasticConfig = field(default_factory=ElasticConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    spark_master: str = field(default_factory=lambda: _env("SPARK_MASTER", "local[*]"))


def get_settings() -> Settings:
    """Build settings from the current environment.

    Deliberately not cached: tests monkeypatch environment variables and expect
    the next call to reflect them.
    """
    return Settings()
