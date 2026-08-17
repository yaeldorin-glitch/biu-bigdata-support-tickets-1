# Design document

**Ticket Intelligence — BIU 8688697201 Big Data and AI**

## 1. Problem and dataset

A support organisation receives tickets as free text and must route each to the
right queue while avoiding re-solving problems it has already solved — both are
text-understanding problems. This project ingests tickets as a stream, enriches
each with derived signals, and serves two AI capabilities: automatic routing and
semantic retrieval of similar past tickets.

**Dataset:** *Customer IT Support — Ticket Dataset* (Tobias Bueck, CC BY-NC 4.0),
28,587 tickets in English and German. Semi-structured: three free-text fields
(`subject`, `body`, `answer`) plus a ragged tag list flattened across eight
sparse columns (1,255 distinct values).

## 2. Architecture

```
Kaggle CSV ──▶ producer.py ──▶ Kafka (tickets.raw) ──▶ Spark Structured Streaming
                                                                │
                                          ┌─────────────────────┼──────────────────┐
                                          ▼                     ▼                  ▼
                                   MinIO bronze/         enrichment UDFs    MinIO silver/
                                   raw parquet           normalise · PII    enriched parquet
                                   immutable             · language ·             │
                                                         sentiment · embed          ▼
                                                                │            batch_kpis.py
                                                                ▼                   │
                                                        Elasticsearch               ▼
                                                   dense_vector 384, kNN     MinIO gold/
                                                   + BM25 over same docs            │
                                                                └─────────┬─────────┘
                                                                          ▼
                                                                Kibana · FastAPI
                                                          /search /ask /route /kpis
```

| technology | role |
|---|---|
| Apache Kafka | streaming ingest — the pipeline's front door and replay buffer |
| Spark Structured Streaming | transformation engine, micro-batch `foreachBatch` fan-out |
| MinIO (S3-compatible) | the data lake — bronze / silver / gold |
| Elasticsearch | NoSQL store *and* vector index — BM25 and kNN over identical docs |
| Docker Compose | the whole stack, reproducible |

**Data flow.** `producer.py` streams the CSV to Kafka with no transformation —
this is deliberately ELT, not ETL: the broker holds raw truth and Spark does the
transform downstream. Each micro-batch is written verbatim to bronze
(immutable), then enrichment UDFs normalise text, redact PII, detect language,
score sentiment, rule-classify, and embed each ticket; one `foreachBatch` call
writes the result to silver *and* Elasticsearch so enrichment runs once per
batch rather than once per sink. `batch_kpis.py` aggregates silver into gold and
the `ticket_kpis` index; FastAPI serves search, RAG, routing and the KPIs
directly (`/kpis`). The full compose file adds Kibana as a dashboard over the
same index; the memory-constrained slim file that was actually run and
verified (section 6) drops it and serves KPIs through the API instead.

## 3. The AI capability

Four of the brief's §6.2 options are implemented, with **(b) embeddings and
semantic search** as the graded centrepiece: `paraphrase-multilingual-MiniLM-L12-v2`
produces 384-d vectors in an Elasticsearch `dense_vector` field (cosine kNN); a
second offline TF-IDF+SVD backend behind the same interface needs no download
and doubles as the classical baseline. **Routing (f):** a similarity-weighted
k-NN over the search embeddings and a linear SVM over word+char TF-IDF are
exposed side by side on `POST /route`. **RAG (c):** `POST /ask` retrieves the
k most similar tickets and grounds the answer, stripping any citation the model
invents. **LLM enrichment (a):** optional per-ticket classification, entity
extraction and summarisation, validated against known label vocabularies before
reaching the index. With `LLM_PROVIDER=none` and no model weights, the pipeline
still produces every number below — a deliberate design property, not a
limitation.

## 4. Results

Routing, stratified 25% held-out split, n = 7,145:

| method | accuracy | macro-F1 |
|---|---|---|
| majority class | 29.2% | 0.045 |
| keyword rules | 27.4% | 0.208 |
| k-NN, neural embeddings | 64.6% | 0.605 |
| **linear SVM, word+char TF-IDF** | **64.8%** | **0.645** |

Retrieval, 250 queries, k=5, proxy relevance (≥2 shared tags): semantic 0.346
queue purity vs. BM25's 0.322, and **hybrid RRF 0.352** beats both. The neural
embedding is worth 6 points of routing accuracy and flips the retrieval result —
the offline LSA backend *loses* to BM25 (0.294 vs. 0.322), since LSA is a linear
projection of the same statistics BM25 already scores, adding no outside
semantic knowledge. Full tables and operational findings: [`RESULTS.md`](RESULTS.md).

## 5. Trade-offs

**`foreachBatch` over three streaming sinks** — one micro-batch fans out to two
Parquet paths and Elasticsearch; separate `writeStream` queries would re-run the
expensive enrichment per sink. **Content-addressed `ticket_id`** (a SHA-1 of the
ticket's content, not a counter) makes every sink idempotent, so Kafka's
at-least-once replay overwrites documents instead of duplicating them —
effectively-once without transactions. **Pure-Python core** — every
transformation is a plain function the Spark UDFs call, which is what makes the
unit tests meaningful and lets the whole pipeline run with no infrastructure;
`batch_kpis.py` cross-checks its Spark aggregates against this same reference so
the two implementations cannot silently drift. **Macro-F1 alongside accuracy**
— with a 29.3% majority class, accuracy alone would make the keyword rules look
almost as good as k-NN; reporting both is what exposes that the rules score
*below* a constant.

## 6. Honest scope

Verified: all 87 unit tests pass; the full pipeline ran end to end over all
28,587 real rows; the PII gate, the LLM label validator and the RAG citation
checker are each pinned by a test. The Docker Compose stack, the Spark
streaming and batch jobs, and the Elasticsearch writes and kNN queries have
since been run against the live stack: real tickets through Kafka → Spark →
MinIO (bronze then silver) → Elasticsearch, the `dense_vector` mapping
confirmed correct against the live index, and the batch job's self-check
against the pure-Python reference passing. Six real bugs surfaced on that
first run — two retired Docker Hub image tags, a missing `pyarrow` dependency,
a volume-permission issue, a relative-import bug in how `spark-submit` invokes
a package module, and the offline embedding backend needing a pre-fitted model
rather than one streaming batch — all fixed; see `docs/DEMO.md`'s
troubleshooting table.
