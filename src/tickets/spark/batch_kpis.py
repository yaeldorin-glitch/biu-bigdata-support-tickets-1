"""Batch KPI job: silver (MinIO) -> gold (MinIO) + Elasticsearch.

The streaming job answers "what is arriving right now". This one answers "what
does the whole corpus say", which is a different access pattern and belongs in a
scheduled batch job rather than in the stream.

Every aggregate here mirrors one in :mod:`tickets.core.kpis`. That duplication is
intentional and worth defending: the pure-Python version is the readable
reference implementation that the unit tests pin, and this version is the one
that scales. :func:`verify_against_reference` checks they agree, so the
duplication cannot silently drift.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from ..config import Settings, get_settings
from ..core.schema import kpi_mapping

logger = logging.getLogger(__name__)


def _long_format(df, kpi: str, dimension: str, bucket_col: str, value_col: str, count_col: str):
    """Reshape a grouped aggregate into the shared (kpi, dimension, bucket, value, count) shape."""
    from pyspark.sql import functions as F

    return df.select(
        F.lit(kpi).alias("kpi"),
        F.lit(dimension).alias("dimension"),
        F.col(bucket_col).cast("string").alias("bucket"),
        F.col(value_col).cast("double").alias("value"),
        F.col(count_col).cast("long").alias("count"),
        F.current_timestamp().alias("computed_at"),
    )


def compute_kpis(silver_df):
    """Compute every KPI as a single long-format DataFrame."""
    from pyspark.sql import functions as F

    total = silver_df.count()
    if total == 0:
        raise ValueError("silver layer is empty; run the streaming job first")

    frames = []

    # --- volume share by categorical dimension --------------------------
    for field in ("queue", "type", "priority", "language"):
        grouped = (
            silver_df.groupBy(field)
            .agg(F.count("*").alias("n"))
            .withColumn("share", F.col("n") / F.lit(total))
        )
        frames.append(_long_format(grouped, "volume_share", field, field, "share", "n"))

    # --- sentiment ------------------------------------------------------
    sentiment = silver_df.groupBy("queue").agg(
        F.avg("sentiment_score").alias("avg_score"),
        F.count("*").alias("n"),
    )
    frames.append(_long_format(sentiment, "avg_sentiment", "queue", "queue", "avg_score", "n"))

    negative = silver_df.groupBy("queue").agg(
        F.avg(F.when(F.col("sentiment") == "negative", 1.0).otherwise(0.0)).alias("rate"),
        F.count("*").alias("n"),
    )
    frames.append(_long_format(negative, "negative_rate", "queue", "queue", "rate", "n"))

    # --- pressure: share of high-priority tickets per queue -------------
    high_priority = silver_df.groupBy("queue").agg(
        F.avg(F.when(F.col("priority") == "high", 1.0).otherwise(0.0)).alias("rate"),
        F.count("*").alias("n"),
    )
    frames.append(_long_format(high_priority, "high_priority_share", "queue", "queue", "rate", "n"))

    # --- routing accuracy of the rule baseline --------------------------
    labelled = silver_df.filter(F.col("queue").isNotNull() & F.col("predicted_queue").isNotNull())
    correct = F.when(F.col("predicted_queue") == F.col("queue"), 1.0).otherwise(0.0)

    overall = labelled.agg(
        F.avg(correct).alias("accuracy"), F.count("*").alias("n")
    ).withColumn("bucket", F.lit("all"))
    frames.append(_long_format(overall, "routing_accuracy", "overall", "bucket", "accuracy", "n"))

    per_queue = labelled.groupBy("queue").agg(
        F.avg(correct).alias("accuracy"), F.count("*").alias("n")
    )
    frames.append(_long_format(per_queue, "routing_accuracy", "queue", "queue", "accuracy", "n"))

    # --- data quality: declared vs detected language --------------------
    detected = silver_df.filter(
        F.col("detected_language").isNotNull() & (F.col("detected_language") != "unknown")
    )
    mismatch = F.when(F.col("language_mismatch"), 1.0).otherwise(0.0)

    mismatch_overall = detected.agg(
        F.avg(mismatch).alias("rate"), F.count("*").alias("n")
    ).withColumn("bucket", F.lit("all"))
    frames.append(
        _long_format(mismatch_overall, "language_mismatch_rate", "overall", "bucket", "rate", "n")
    )

    mismatch_by_language = detected.groupBy("language").agg(
        F.avg(mismatch).alias("rate"), F.count("*").alias("n")
    )
    frames.append(
        _long_format(
            mismatch_by_language, "language_mismatch_rate", "declared_language", "language", "rate", "n"
        )
    )

    # --- PII exposure ---------------------------------------------------
    pii = silver_df.select(F.explode_outer("pii_found").alias("kind"))
    pii_by_kind = (
        pii.filter(F.col("kind").isNotNull())
        .groupBy("kind")
        .agg(F.count("*").alias("n"))
        .withColumn("share", F.col("n") / F.lit(total))
    )
    frames.append(_long_format(pii_by_kind, "pii_rate", "kind", "kind", "share", "n"))

    # --- tag frequency --------------------------------------------------
    tags = silver_df.select(F.explode_outer("tags").alias("tag")).filter(F.col("tag").isNotNull())
    top_tags = (
        tags.groupBy("tag")
        .agg(F.count("*").alias("n"))
        .withColumn("share", F.col("n") / F.lit(total))
        .orderBy(F.desc("n"))
        .limit(20)
    )
    frames.append(_long_format(top_tags, "tag_frequency", "tag", "tag", "share", "n"))

    result = frames[0]
    for frame in frames[1:]:
        result = result.unionByName(frame)
    return result


def verify_against_reference(silver_df, tolerance: float = 1e-6, sample: int = 5000) -> dict[str, Any]:
    """Cross-check the Spark aggregates against the pure-Python reference.

    Runs both implementations over the same sample and compares the headline
    numbers. A mismatch means the two have drifted and one of them is wrong.
    """
    from ..core import kpis as reference

    rows = [row.asDict(recursive=True) for row in silver_df.limit(sample).collect()]
    if not rows:
        return {"checked": 0, "ok": True, "notes": "silver sample was empty"}

    spark_rows = [r.asDict() for r in compute_kpis(silver_df.limit(sample)).collect()]
    spark_lookup = {(r["kpi"], r["dimension"], r["bucket"]): r["value"] for r in spark_rows}

    mismatches = []
    for row in reference.compute_all(rows):
        key = (row["kpi"], row["dimension"], row["bucket"])
        if key not in spark_lookup:
            continue
        if abs(spark_lookup[key] - row["value"]) > tolerance:
            mismatches.append(
                {"key": key, "spark": spark_lookup[key], "reference": row["value"]}
            )

    return {"checked": len(spark_lookup), "ok": not mismatches, "mismatches": mismatches[:10]}


def ensure_kpi_index(settings: Settings) -> None:
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        logger.warning("elasticsearch client not installed; skipping KPI index creation")
        return

    client = Elasticsearch(
        settings.elastic.url,
        basic_auth=(settings.elastic.user, settings.elastic.password)
        if settings.elastic.user
        else None,
        request_timeout=30,
    )
    index = settings.elastic.index_kpis
    if not client.indices.exists(index=index):
        mapping = kpi_mapping()
        client.indices.create(index=index, settings=mapping["settings"], mappings=mapping["mappings"])
        logger.info("created elasticsearch index %r", index)


def run(settings: Settings | None = None, verify: bool = True) -> None:
    from .session import build_spark, elasticsearch_options

    settings = settings or get_settings()
    ensure_kpi_index(settings)

    spark = build_spark("tickets-batch-kpis", settings)
    store = settings.store

    silver = spark.read.parquet(store.silver_path)
    logger.info("read %s enriched tickets from %s", silver.count(), store.silver_path)

    kpis = compute_kpis(silver).cache()
    logger.info("computed %s KPI rows", kpis.count())

    if verify:
        report = verify_against_reference(silver)
        if report["ok"]:
            logger.info("verification passed against the reference implementation (%s KPIs)",
                        report["checked"])
        else:
            logger.error("verification FAILED: %s", report["mismatches"])

    kpis.write.mode("overwrite").parquet(store.gold_path)

    # ``overwrite`` on the KPI index: these are full-corpus aggregates, so the
    # previous run's values are superseded, not appended to. KPI rows have no
    # natural key, so ``mapping_id=None`` lets Elasticsearch assign ids.
    (
        kpis.write.format("org.elasticsearch.spark.sql")
        .options(**elasticsearch_options(settings, mapping_id=None))
        .mode("overwrite")
        .save(settings.elastic.index_kpis)
    )

    kpis.orderBy("kpi", "dimension", "bucket").show(60, truncate=False)
    logger.info("KPIs written to %s and index %r", store.gold_path, settings.elastic.index_kpis)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute KPIs over the silver layer.")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip cross-checking against the reference implementation")
    args = parser.parse_args(argv)
    run(verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
