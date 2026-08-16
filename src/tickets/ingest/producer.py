"""CSV -> Kafka producer.

Replays the Kaggle export onto the ``tickets.raw`` topic as one JSON message per
ticket, at a configurable rate so the streaming job has something to actually
stream. This is the "E" and "L" of the ELT: extract from the file, load to the
broker untouched. All transformation happens downstream in Spark, which is what
makes it ELT rather than ETL.

Messages are keyed by ``ticket_id`` (a content hash) so that all versions of a
ticket land on the same partition and ordering is preserved per ticket.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

from ..config import get_settings
from ..core.schema import RAW_COLUMNS, make_ticket_id

logger = logging.getLogger(__name__)


def iter_csv_rows(path: Path) -> Iterator[dict]:
    """Stream rows from the CSV without loading it all into memory.

    ``csv.DictReader`` rather than pandas: the file is 26MB here but the point of
    a producer is that it should not care how big the source is.
    """
    import csv

    # The corpus contains fields well over the default 128KB field limit only
    # rarely, but raising it costs nothing and avoids a crash mid-replay.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(RAW_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
        for row in reader:
            yield row


def build_producer(bootstrap: str):
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=bootstrap.split(","),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        # 'all' costs latency but guarantees the message is on every in-sync
        # replica before we call it sent. For a demo of correctness over speed,
        # that is the right default.
        acks="all",
        linger_ms=50,
        retries=3,
        compression_type="gzip",
    )


def produce(
    csv_path: Path,
    bootstrap: str,
    topic: str,
    rate: float = 200.0,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Publish tickets to Kafka. Returns the number of messages sent."""
    producer = None if dry_run else build_producer(bootstrap)
    interval = 1.0 / rate if rate > 0 else 0.0

    sent = 0
    started = time.time()
    for row in iter_csv_rows(csv_path):
        if limit is not None and sent >= limit:
            break
        payload = {column: row.get(column) for column in RAW_COLUMNS}
        key = make_ticket_id(payload)
        payload["ticket_id"] = key

        if dry_run:
            if sent < 3:
                logger.info("dry-run message: %s", json.dumps(payload, ensure_ascii=False)[:300])
        else:
            producer.send(topic, key=key, value=payload)

        sent += 1
        if sent % 1000 == 0:
            logger.info("sent %s messages (%.0f msg/s)", sent, sent / max(time.time() - started, 1e-9))
        if interval:
            time.sleep(interval)

    if producer is not None:
        producer.flush()
        producer.close()

    logger.info("done: %s messages in %.1fs", sent, time.time() - started)
    return sent


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Replay the ticket CSV onto a Kafka topic.")
    parser.add_argument("--csv", type=Path, default=None, help="path to the source CSV")
    parser.add_argument("--bootstrap", default=settings.kafka.bootstrap_servers)
    parser.add_argument("--topic", default=settings.kafka.topic_raw)
    parser.add_argument("--rate", type=float, default=settings.kafka.produce_rate,
                        help="messages per second; 0 means as fast as possible")
    parser.add_argument("--limit", type=int, default=None, help="stop after N messages")
    parser.add_argument("--dry-run", action="store_true", help="parse and print without connecting to Kafka")
    args = parser.parse_args(argv)

    from ..config import RAW_CSV

    csv_path = args.csv or RAW_CSV
    if not csv_path.exists():
        parser.error(
            f"CSV not found at {csv_path}. Download the dataset (see data/README.md) "
            "or pass --csv."
        )

    produce(csv_path, args.bootstrap, args.topic, args.rate, args.limit, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
