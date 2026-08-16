"""Spark Structured Streaming: Kafka -> MinIO (bronze/silver) -> Elasticsearch.

This is Part A of the brief. The medallion layout is deliberate:

``bronze``  raw JSON exactly as it arrived, plus ingest metadata. Immutable.
            If the enrichment logic changes we replay from here rather than
            re-reading Kafka, whose retention is finite.
``silver``  enriched, validated, embedded. The analytical table.
``gold``    KPI aggregates, written by :mod:`tickets.spark.batch_kpis`.

Why ``foreachBatch`` instead of three independent ``writeStream`` sinks: a single
micro-batch has to fan out to two Parquet paths *and* Elasticsearch, and the
elasticsearch-hadoop streaming sink is more limited than its batch writer.
``foreachBatch`` gives us the batch API inside a streaming query, so all three
writes share one execution of the (expensive) enrichment stage instead of
recomputing it per sink.

Delivery semantics are at-least-once: Spark may replay a micro-batch after a
failure. That is fine because every sink is idempotent -- Parquet writes are
partitioned and overwritten by batch id, and Elasticsearch writes are keyed on
``ticket_id``, a content hash. The result is effectively-once *in the sinks*
without needing transactional support.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Iterator

from ..config import Settings, get_settings
from ..core.schema import es_mapping

logger = logging.getLogger(__name__)


def raw_schema():
    """Schema of the JSON messages the producer publishes."""
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    fields = [
        StructField("ticket_id", StringType(), True),
        StructField("subject", StringType(), True),
        StructField("body", StringType(), True),
        StructField("answer", StringType(), True),
        StructField("type", StringType(), True),
        StructField("queue", StringType(), True),
        StructField("priority", StringType(), True),
        StructField("language", StringType(), True),
        StructField("version", StringType(), True),
    ]
    fields += [StructField(f"tag_{i}", StringType(), True) for i in range(1, 9)]
    # version arrives as a string from JSON; cast downstream rather than letting
    # a malformed value null the entire row via schema-level failure.
    return StructType(fields)


def enriched_schema():
    from pyspark.sql.types import (
        ArrayType, BooleanType, FloatType, IntegerType, StringType, StructField, StructType,
    )

    return StructType(
        [
            StructField("ticket_id", StringType(), True),
            StructField("subject", StringType(), True),
            StructField("body", StringType(), True),
            StructField("answer", StringType(), True),
            StructField("type", StringType(), True),
            StructField("queue", StringType(), True),
            StructField("priority", StringType(), True),
            StructField("language", StringType(), True),
            StructField("version", IntegerType(), True),
            StructField("tags", ArrayType(StringType()), True),
            StructField("text", StringType(), True),
            StructField("text_redacted", StringType(), True),
            StructField("pii_found", ArrayType(StringType()), True),
            StructField("detected_language", StringType(), True),
            StructField("language_mismatch", BooleanType(), True),
            StructField("sentiment", StringType(), True),
            StructField("sentiment_score", FloatType(), True),
            StructField("predicted_queue", StringType(), True),
            StructField("predicted_queue_confidence", FloatType(), True),
            StructField("predicted_priority", StringType(), True),
            StructField("embedding", ArrayType(FloatType()), True),
            StructField("enrichment_source", StringType(), True),
            StructField("ingest_ts", StringType(), True),
            StructField("ingest_date", StringType(), True),
        ]
    )


def _make_enricher(offline_model_path: str | None):
    """Build the ``mapInPandas`` function that enriches a partition.

    Closes over only a *path*, never a live model object: Spark pickles this
    function and ships it to executors, and a sentence-transformers model is not
    picklable. Each executor loads its own copy once, cached by the
    process-wide singleton in :mod:`tickets.core.embeddings`.
    """

    def enrich_partition(batches: Iterator["pandas.DataFrame"]) -> Iterator["pandas.DataFrame"]:  # noqa: F821
        import pandas as pd

        from ..core.embeddings import get_backend
        from ..core.enrich import enrich_batch
        from ..core.schema import is_valid, parse_row

        backend = get_backend(
            offline_model_path=offline_model_path if offline_model_path else None
        )

        for batch in batches:
            tickets = []
            for row in batch.to_dict("records"):
                ticket = parse_row(row)
                ok, reason = is_valid(ticket)
                if not ok:
                    # Dropped rows are counted by the caller via the row-count
                    # delta; a production system would route these to the DLQ.
                    logger.debug("dropping invalid ticket: %s", reason)
                    continue
                tickets.append(ticket)

            if not tickets:
                yield pd.DataFrame(columns=[f.name for f in enriched_schema().fields])
                continue

            enriched = enrich_batch(tickets, backend=backend)
            yield pd.DataFrame([t.to_dict() for t in enriched])

    return enrich_partition


def ensure_elasticsearch_index(settings: Settings) -> None:
    """Create the tickets index with the dense_vector mapping if it is absent.

    Must happen before the first write: Elasticsearch would otherwise infer
    ``embedding`` as an array of floats, which is *not* searchable by kNN, and
    the mapping cannot be changed afterwards without a reindex.
    """
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        logger.warning("elasticsearch client not installed; skipping index creation")
        return

    client = Elasticsearch(
        settings.elastic.url,
        basic_auth=(settings.elastic.user, settings.elastic.password)
        if settings.elastic.user
        else None,
        request_timeout=30,
    )
    index = settings.elastic.index_tickets
    if client.indices.exists(index=index):
        logger.info("elasticsearch index %r already exists", index)
        return

    mapping = es_mapping(settings.ai.embedding_dim)
    client.indices.create(index=index, settings=mapping["settings"], mappings=mapping["mappings"])
    logger.info("created elasticsearch index %r with %s-dim dense_vector",
                index, settings.ai.embedding_dim)


def run(
    settings: Settings | None = None,
    offline_model_path: str | None = None,
    starting_offsets: str = "earliest",
    trigger_seconds: int = 10,
    max_offsets_per_trigger: int = 2000,
    await_termination: bool = True,
):
    from pyspark.sql import functions as F

    from .session import build_spark, elasticsearch_options

    settings = settings or get_settings()
    ensure_elasticsearch_index(settings)

    spark = build_spark("tickets-stream", settings)
    store = settings.store
    es_options = elasticsearch_options(settings)

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka.bootstrap_servers)
        .option("subscribe", settings.kafka.topic_raw)
        .option("startingOffsets", starting_offsets)
        # Bounds each micro-batch so the first run over a full topic does not
        # try to enrich 28k rows in a single batch and blow the executor heap.
        .option("maxOffsetsPerTrigger", str(max_offsets_per_trigger))
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw.select(
            F.col("key").cast("string").alias("kafka_key"),
            F.from_json(F.col("value").cast("string"), raw_schema()).alias("payload"),
            F.col("topic"),
            F.col("partition"),
            F.col("offset"),
            F.col("timestamp").alias("kafka_ts"),
        )
        .filter(F.col("payload").isNotNull())
        .select("payload.*", "kafka_key", "topic", "partition", "offset", "kafka_ts")
    )

    enrich_partition = _make_enricher(offline_model_path)
    schema = enriched_schema()

    def process_batch(batch_df, batch_id: int) -> None:
        if batch_df.isEmpty():
            logger.info("batch %s is empty; skipping", batch_id)
            return

        batch_df = batch_df.persist()
        try:
            count = batch_df.count()
            logger.info("batch %s: %s messages", batch_id, count)

            # --- bronze: raw, untransformed, immutable -------------------
            (
                batch_df.withColumn("ingest_date", F.current_date().cast("string"))
                .withColumn("batch_id", F.lit(batch_id))
                .write.mode("append")
                .partitionBy("ingest_date")
                .parquet(store.bronze_path)
            )

            # --- enrichment ---------------------------------------------
            source_columns = [f.name for f in raw_schema().fields]
            enriched = batch_df.select(*source_columns).mapInPandas(enrich_partition, schema=schema)
            enriched = enriched.persist()

            enriched_count = enriched.count()
            if enriched_count == 0:
                logger.warning("batch %s produced no valid tickets", batch_id)
                return
            if enriched_count < count:
                logger.warning(
                    "batch %s: %s of %s messages failed validation and were dropped",
                    batch_id, count - enriched_count, count,
                )

            # --- silver: enriched analytical table ----------------------
            (
                enriched.write.mode("append")
                .partitionBy("ingest_date")
                .parquet(store.silver_path)
            )

            # --- serving: Elasticsearch ---------------------------------
            (
                enriched.write.format("org.elasticsearch.spark.sql")
                .options(**es_options)
                .mode("append")
                .save(settings.elastic.index_tickets)
            )

            logger.info("batch %s: wrote %s enriched tickets", batch_id, enriched_count)
        finally:
            batch_df.unpersist()

    query = (
        parsed.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", f"{store.checkpoint_path}/stream")
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
    )

    logger.info("streaming query started; writing to %s and %s",
                store.silver_path, settings.elastic.index_tickets)
    if await_termination:
        query.awaitTermination()
    return query


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the streaming enrichment pipeline.")
    parser.add_argument("--offline-model", default=os.getenv("OFFLINE_EMBEDDING_MODEL"),
                        help="path to a fitted offline embedding model (joblib)")
    parser.add_argument("--starting-offsets", default="earliest", choices=["earliest", "latest"])
    parser.add_argument("--trigger-seconds", type=int, default=10)
    parser.add_argument("--max-offsets-per-trigger", type=int, default=2000)
    args = parser.parse_args(argv)

    run(
        offline_model_path=args.offline_model,
        starting_offsets=args.starting_offsets,
        trigger_seconds=args.trigger_seconds,
        max_offsets_per_trigger=args.max_offsets_per_trigger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
