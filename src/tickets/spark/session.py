"""SparkSession construction, with S3A wired to MinIO.

Isolated here because the S3A settings are fiddly and every job needs the exact
same ones. The three that actually matter for MinIO:

``fs.s3a.path.style.access=true``
    MinIO serves buckets as ``endpoint/bucket``, not ``bucket.endpoint``. Without
    this, Hadoop tries virtual-host style and fails DNS resolution.
``fs.s3a.endpoint``
    points S3A at MinIO instead of AWS.
``fs.s3a.connection.ssl.enabled=false``
    the compose stack speaks plain HTTP internally.
"""

from __future__ import annotations

from ..config import Settings, get_settings

# Pinned so a `spark-submit` on a clean machine resolves a working, mutually
# compatible set. hadoop-aws must match the Hadoop version Spark was built
# against (3.3.4 for Spark 3.5.x) or S3A fails at class-load time.
DEFAULT_PACKAGES = [
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
    "org.apache.hadoop:hadoop-aws:3.3.4",
    "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    "org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4",
]


def build_spark(app_name: str, settings: Settings | None = None, shuffle_partitions: int = 8):
    """Create a SparkSession configured for this project."""
    from pyspark.sql import SparkSession

    settings = settings or get_settings()
    store = settings.store

    builder = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark_master)
        # 8 is a sane default for a laptop; the Spark default of 200 creates
        # hundreds of tiny files per micro-batch on a 28k-row dataset.
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.jars.packages", ",".join(DEFAULT_PACKAGES))
        # --- S3A / MinIO ---
        .config("spark.hadoop.fs.s3a.endpoint", store.endpoint)
        .config("spark.hadoop.fs.s3a.access.key", store.access_key)
        .config("spark.hadoop.fs.s3a.secret.key", store.secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", str(store.endpoint.startswith("https")).lower())
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        # Committing straight to the final path is safe on MinIO and avoids the
        # rename-heavy default committer, which is slow on object stores.
        .config("spark.hadoop.fs.s3a.committer.name", "directory")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def elasticsearch_options(
    settings: Settings | None = None, mapping_id: str | None = "ticket_id"
) -> dict[str, str]:
    """Options for the elasticsearch-hadoop connector.

    ``es.mapping.id`` is what makes writes idempotent: the connector issues an
    index request keyed on our content hash, so replaying Kafka updates existing
    documents instead of duplicating them. Pass ``mapping_id=None`` for rows with
    no natural key (the KPI aggregates) and let Elasticsearch generate ids.
    """
    settings = settings or get_settings()
    elastic = settings.elastic
    options = {
        "es.nodes": elastic.host.split(":")[0],
        "es.port": elastic.host.split(":")[1] if ":" in elastic.host else "9200",
        "es.nodes.wan.only": "true",
        "es.write.operation": "index",
        "es.spark.dataframe.write.null": "true",
    }
    if mapping_id:
        options["es.mapping.id"] = mapping_id
    if elastic.user and elastic.password:
        options["es.net.http.auth.user"] = elastic.user
        options["es.net.http.auth.pass"] = elastic.password
    return options
