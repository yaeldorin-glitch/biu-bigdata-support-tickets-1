# Design document

**Ticket Intelligence — BIU 8688697201 Big Data and AI**

---

## 1. Problem

A support organisation receives tickets as free text. Two costs follow: someone
must decide which queue each ticket belongs to, and agents re-solve problems the
organisation has already solved. Both are text-understanding problems, which is
why they resist the SQL-shaped tooling that handles the rest of a support desk's
data.

This project builds a pipeline that ingests tickets as a stream, enriches each
one with derived signals, and serves two capabilities: **automatic routing** and
**semantic retrieval of similar past tickets**.

**Dataset.** *Customer IT Support — Ticket Dataset* (Tobias Bueck, CC BY-NC 4.0),
28,587 tickets in English and German. It qualifies as semi-structured on two
counts: three natural-language fields (`subject`, `body`, `answer`, averaging 387
characters), and a ragged tag list flattened across eight sparse columns
(`tag_1`…`tag_8`, 1,255 distinct values, 98% null by `tag_8`).

---

## 2. Architecture

```
Kaggle CSV ──▶ producer.py ──▶ Kafka (tickets.raw) ──▶ Spark Structured Streaming
                                                                │
                                          ┌─────────────────────┼──────────────────┐
                                          ▼                     ▼                  ▼
                                   MinIO bronze/         enrichment UDFs    MinIO silver/
                                   raw parquet           · normalise        enriched parquet
                                   immutable             · redact PII              │
                                                         · detect language         ▼
                                                         · sentiment         batch_kpis.py
                                                         · rule classify           │
                                                         · embed 384-d             ▼
                                                                │            MinIO gold/
                                                                ▼
                                                        Elasticsearch
                                                   dense_vector 384, cosine kNN
                                                   + BM25 over the same docs
                                                                │
                                                   ┌────────────┴────────────┐
                                                   ▼                         ▼
                                              Kibana                    FastAPI
                                                                  /search /compare
                                                                  /ask /route /kpis
```

Diagram source: [`architecture.mmd`](architecture.mmd).

### Course technologies used

| technology | role |
|---|---|
| **Apache Kafka** | streaming ingest; the topic is the pipeline's front door and its replay buffer |
| **Apache Spark Structured Streaming** | the transformation engine; micro-batch, `foreachBatch` fan-out |
| **MinIO (S3-compatible object store)** | the data lake — bronze / silver / gold |
| **Elasticsearch** | NoSQL document store *and* the vector index; serves BM25 and kNN over identical documents |
| **Kibana** | dashboards over the KPI index |
| **Docker Compose** | the whole stack, reproducible |
| Parquet | columnar storage in every lake layer |

### Data flow

1. **Extract / load.** `producer.py` streams the CSV row by row and publishes one
   JSON message per ticket to `tickets.raw`, keyed by `ticket_id`. No
   transformation happens here — this is deliberately **ELT**, not ETL. The
   broker holds raw truth; Spark does the T.
2. **Bronze.** Every micro-batch is written to `s3a://lake/bronze/tickets`
   partitioned by ingest date, exactly as received. Immutable.
3. **Transform.** Enrichment UDFs normalise text, redact PII, detect language,
   score sentiment, apply the rule classifier, and produce a 384-dimension
   embedding.
4. **Silver.** The enriched, validated records land in
   `s3a://lake/silver/tickets_enriched` and simultaneously in Elasticsearch.
5. **Gold.** `batch_kpis.py` reads silver, computes aggregates, and writes them to
   `s3a://lake/gold/kpis` and the `ticket_kpis` index.
6. **Serve.** FastAPI exposes search, RAG and routing; Kibana reads the KPIs.

---

## 3. The AI capability

Four of the brief's section-6.2 options are implemented, with **(b) embeddings and
semantic search** as the graded centrepiece.

**Embeddings.** `paraphrase-multilingual-MiniLM-L12-v2` produces 384-dimension
vectors stored in an Elasticsearch `dense_vector` field with `index: true` and
cosine similarity, which is what makes `knn` queries possible. The model is
multilingual by design: the corpus is 57% English and 43% German, and a shared
vector space is what would let a German query retrieve an equivalent English
ticket. A second **offline backend** (word + character TF-IDF reduced to 384
dimensions by truncated SVD) implements the same interface with no download, so
the pipeline never hard-depends on model weights.

**Routing (option f).** The same vectors feed a similarity-weighted k-NN
classifier that predicts `queue`. No separate model, no training run — the
cheapest possible route from "we have embeddings" to "we have business value".

**RAG (option c).** `POST /ask` retrieves the k most similar tickets and asks an
LLM to answer strictly from them. Citations are post-checked against the
retrieved ids and stripped if invented.

**LLM enrichment (option a).** Optional per-ticket classification, entity
extraction and summarisation, with model output validated against the known
label vocabularies before it can reach the index.

**Fallbacks are first-class.** With `LLM_PROVIDER=none` and no model weights, the
pipeline still produces every number in this document. That is a deliberate
design property, not a limitation: the demo cannot be broken by a failed
download.

---

## 4. Results

Routing, stratified 25% held-out split, n = 7,145:

| method | accuracy | macro-F1 |
|---|---|---|
| majority class | 29.2% | 0.045 |
| keyword rules | 27.4% | 0.208 |
| embedding centroid | 28.4% | 0.252 |
| **embedding k-NN (k=15)** | **48.4%** | **0.384** |

Retrieval, 250 queries, k=5, proxy relevance (≥2 shared tags):

| method | queue purity@5 | precision@5 | MRR |
|---|---|---|---|
| semantic | 0.294 | 0.713 | 0.823 |
| keyword (BM25) | **0.322** | 0.728 | **0.859** |
| hybrid (RRF) | 0.294 | **0.745** | 0.844 |

Selected operational findings:

- Tickets labelled `de` are mislabelled **27.2%** of the time; tickets labelled
  `en`, **0.05%**. The 11.7% headline rate hides a one-directional defect.
- Service Outages and Maintenance is 4.0% of volume but **71.0%** high priority.
- Negative sentiment ranges from 53.8% (Technical Support) to 21.2% (Sales).

Full tables: [`RESULTS.md`](RESULTS.md), generated from `output/report.json`.

---

## 5. Trade-offs and what we would do differently

**`foreachBatch` over multiple streaming sinks.** A micro-batch fans out to two
Parquet paths and Elasticsearch. Three independent `writeStream` queries would
re-execute the expensive enrichment for each sink; `foreachBatch` runs it once.
The cost is that we hand-manage the writes rather than letting Spark do it.

**At-least-once, made effectively-once in the sinks.** `ticket_id` is a SHA-1 of
the ticket's content rather than a counter, so a replayed micro-batch overwrites
the same Elasticsearch documents instead of duplicating them. This is why the id
is content-addressed — it is the cheapest way to get idempotency without
transactions.

**Exact kNN in the offline path, approximate (HNSW) in Elasticsearch.** At 28k
documents, brute force is fast enough and removes ANN recall as a confound when
comparing semantic against BM25. Elasticsearch uses HNSW because a real corpus
would not fit in memory.

**Pure-Python core, Spark as a thin wrapper.** Every transformation is a pure
function; the UDFs call them. This is what makes the unit tests meaningful and
what lets the whole pipeline run without Spark. The cost is a second, native
implementation of the KPI aggregates for scale — mitigated by
`verify_against_reference()`, which cross-checks the two.

**Macro-F1 alongside accuracy.** With a 29.3% majority class, accuracy alone
would have made the keyword rules look almost as good as the k-NN. Reporting
both is what exposed that the rules score *below* a constant.

**The proxy relevance metric is the weakest link in the evaluation.** Shared tags
reward topical similarity, which is what routing and deduplication need — but
they would under-credit a system that retrieves a *solution* phrased differently
from the problem. Human relevance judgements on a few hundred queries would be
the honest fix, and it is the first thing we would add.

**The retrieval result contradicted the design.** LSA does not beat BM25, because
it is a linear projection of the same term-document statistics rather than
outside semantic knowledge. The multilingual neural model is expected to change
this. **We have not measured it, so we do not claim it** — reproducing that
comparison is a one-line environment change.

---

## 6. Honest scope

Verified: all 80 unit tests pass; the full pipeline ran end to end over all
28,587 real rows; the PII gate, the LLM label validator and the RAG citation
checker are each pinned by a test.

Not executed in the build environment: the Docker Compose stack, the Spark
streaming and batch jobs, the Elasticsearch writes and kNN queries, and the
neural embedding backend — no Docker daemon, no `pyspark` install, no reachable
cluster, no model download. Those paths follow the documented APIs and standard
patterns, but they have not been run and should be exercised before the demo.
