"""Elasticsearch client helpers and index management."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from ..config import Settings, get_settings
from ..core.schema import es_mapping, kpi_mapping

logger = logging.getLogger(__name__)


def get_client(settings: Settings | None = None):
    from elasticsearch import Elasticsearch

    settings = settings or get_settings()
    elastic = settings.elastic
    auth = (elastic.user, elastic.password) if elastic.user and elastic.password else None
    return Elasticsearch(elastic.url, basic_auth=auth, request_timeout=60, retry_on_timeout=True)


def create_indices(settings: Settings | None = None, recreate: bool = False) -> dict[str, str]:
    """Create the tickets and KPI indices.

    The ``embedding`` field must be declared as ``dense_vector`` *before* the
    first document is written. Elasticsearch's dynamic mapping would otherwise
    type it as a plain float array, which cannot be queried with ``knn``, and
    field types are immutable once set -- the only fix at that point is a
    reindex. Hence ``recreate``.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    results = {}

    for index, mapping in (
        (settings.elastic.index_tickets, es_mapping(settings.ai.embedding_dim)),
        (settings.elastic.index_kpis, kpi_mapping()),
    ):
        exists = client.indices.exists(index=index)
        if exists and recreate:
            client.indices.delete(index=index)
            logger.warning("deleted existing index %r", index)
            exists = False
        if exists:
            results[index] = "exists"
            continue
        client.indices.create(index=index, settings=mapping["settings"], mappings=mapping["mappings"])
        results[index] = "created"
        logger.info("created index %r", index)

    return results


def bulk_index(
    documents: Iterable[dict[str, Any]],
    index: str,
    settings: Settings | None = None,
    id_field: str | None = "ticket_id",
    chunk_size: int = 500,
) -> tuple[int, list[Any]]:
    """Bulk-index documents. Used by the offline loader, not by Spark."""
    from elasticsearch.helpers import bulk

    client = get_client(settings)

    def _actions():
        for doc in documents:
            action: dict[str, Any] = {"_index": index, "_source": doc}
            if id_field and doc.get(id_field):
                action["_id"] = doc[id_field]
            yield action

    # raise_on_error=False so one bad document does not abort the whole load;
    # the caller inspects the returned errors.
    success, errors = bulk(client, _actions(), chunk_size=chunk_size, raise_on_error=False)
    if errors:
        logger.warning("%d documents failed to index; first error: %s", len(errors), errors[0])
    return success, list(errors)


def index_stats(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    client = get_client(settings)
    out: dict[str, Any] = {}
    for index in (settings.elastic.index_tickets, settings.elastic.index_kpis):
        if client.indices.exists(index=index):
            out[index] = client.count(index=index)["count"]
        else:
            out[index] = None
    return out
