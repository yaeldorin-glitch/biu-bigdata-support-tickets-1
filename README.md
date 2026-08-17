# Ticket Intelligence — Big Data & AI course project

**BIU 8688697201 — Big Data and AI**

An end-to-end ELT pipeline over 28,587 multilingual customer IT support tickets:
Kafka → Spark Structured Streaming → MinIO (bronze/silver/gold) → Elasticsearch,
with an embedding-based AI layer that does semantic search, retrieval-augmented
question answering, and automatic ticket routing.

The operational question the project answers: **given only the free text of an
incoming ticket, which support queue should it go to, and what similar tickets
have we already solved?**

---

## Headline result

Predicting the `queue` label from ticket text alone, on a stratified 25%
held-out split (7,145 tickets):

| method | accuracy | macro-F1 |
|---|---|---|
| always guess the majority class | 29.2% | 0.045 |
| keyword rules (hand-written) | 27.4% | 0.208 |
| nearest class centroid (embeddings) | 22.0% | 0.205 |
| k-NN over neural embeddings (k=5) | 64.6% | 0.605 |
| **linear SVM, word+char TF-IDF** | **64.8%** | **0.645** |

Two things worth saying out loud, because they are the interesting part:

- **Hand-written keyword rules lose to guessing.** 27.4% accuracy is *below* the
  29.3% you get by answering "Technical Support" every time. The rules do carry
  real signal — their macro-F1 is 0.208 against the majority baseline's 0.045 —
  but on raw accuracy they are worse than useless. This is exactly why the
  evaluation reports both numbers.
- **`k` mattered more than the model did.** k-NN at k=15 scores 48.2%; at k=5 it
  scores 58.4% on the offline backend. Accuracy falls monotonically as k grows,
  because with ten classes and a long tail a wide neighbourhood drags every
  prediction toward the majority queue.
- **The neural embedding is worth 6 points to k-NN, and nothing to the SVM.**
  Swapping the offline TF-IDF+SVD backend for
  `paraphrase-multilingual-MiniLM-L12-v2` moved k-NN from 58.4% to **64.6%**
  (macro-F1 0.529 to 0.605) and left the linear SVM at 64.8% — as it must, since
  that model reads sparse term counts and never touches an embedding. Worth
  stating plainly: a better representation only helps the methods that consume
  it.

Every figure here is generated from the pipeline's own output rather than typed
by hand. See [`docs/RESULTS.md`](docs/RESULTS.md) for the full tables,
[`docs/CEILING.md`](docs/CEILING.md) for how far accuracy can actually go and why,
and [`docs/DEMO.md`](docs/DEMO.md) for the demo runbook.

---

## Quick start

The fastest path needs **no Docker, no API key and no network**. Install the
package once, and every command below works from any directory:

```bash
cd biu-bigdata-support-tickets     # you must be INSIDE the project folder for this step
pip install -e .
tickets-pipeline --limit 5000
```

<details>
<summary><b>Windows / PowerShell</b></summary>

Easiest route — open PowerShell **inside the project folder** (in File Explorer,
type `powershell` in the address bar and press Enter) and run:

```powershell
.\run.ps1
```

That checks Python, installs the package, warns you if the dataset or Docker
memory is wrong, then runs the pipeline. Other modes: `.\run.ps1 -Full`,
`.\run.ps1 -Api`, `.\run.ps1 -Stack`.

Or do it by hand:

```powershell
cd C:\path\to\biu-bigdata-support-tickets
pip install -e .
tickets-pipeline --limit 5000
```

Environment variables use different syntax from Linux — `VAR=value` on one line
does **not** work in PowerShell:

```powershell
$env:EMBEDDING_BACKEND = "sentence-transformers"   # PowerShell
```
```cmd
set EMBEDDING_BACKEND=sentence-transformers        :: cmd.exe
```

`ModuleNotFoundError: No module named 'tickets'` means one of two things: you are
not in the project folder, or you skipped `pip install -e .`. Running
`python -m tickets.offline_pipeline` from your home directory will always fail —
use the `tickets-pipeline` command instead, which works anywhere.
</details>

Without installing, you can still run it from the project root with
`PYTHONPATH=src python -m tickets.offline_pipeline`.

That runs the complete logic — parse, validate, redact, enrich, embed, index,
evaluate, aggregate — over the committed 300-row sample or the full CSV if you
have it, and writes `output/report.json`. It takes about a minute for 5,000
tickets.

For the full stack (on an 8GB machine, `make up` uses `docker-compose.slim.yml`,
which drops Kibana and caps memory to fit):

```bash
cp .env.example .env
EMBEDDING_BACKEND=offline tickets-pipeline --limit 3000  # one-time: fits the offline model make stream needs
make up          # kafka + minio + elasticsearch + spark
make indices     # create the dense_vector mapping BEFORE any writes
make produce     # replay the CSV onto the kafka topic
make stream      # spark structured streaming: kafka -> minio + elasticsearch
make kpis        # spark batch job: silver -> gold + elasticsearch
make api         # http://localhost:8000/docs
```

MinIO's console is on <http://localhost:9001> (`minioadmin` / `minioadmin`).
Verified against this exact sequence — see [`docs/DEMO.md`](docs/DEMO.md) for
the bugs that surfaced the first time and how each was fixed.

> **Dataset.** The 26MB CSV is CC BY-NC 4.0 and is not committed. See
> [`data/README.md`](data/README.md) for how to fetch it. A stratified 300-row
> sample **is** committed, and the runner falls back to it automatically, so
> everything above works immediately after cloning.
>
> **Expect different numbers from the sample.** 300 rows is far too few to
> reproduce the results below — the k-NN classifier in particular needs
> neighbours to vote, and on the sample it underperforms the keyword rules. The
> sample is there to prove the pipeline runs, not to reproduce the findings.
> Every figure in this README and in `docs/RESULTS.md` comes from the full
> 28,587-row dataset.

---

## Architecture

```
Kaggle CSV ──▶ producer.py ──▶ Kafka (tickets.raw)
                                    │
                                    ▼
                    Spark Structured Streaming (foreachBatch)
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          MinIO bronze/       enrichment UDFs    MinIO silver/
          (raw parquet,        · normalise        (enriched parquet)
           immutable)          · redact PII              │
                               · detect language         ▼
                               · sentiment         batch_kpis.py
                               · rule classify           │
                               · embed 384-d             ▼
                                    │             MinIO gold/ (KPIs)
                                    ▼
                            Elasticsearch
                       (dense_vector 384, cosine kNN
                        + BM25 on the same documents)
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   Kibana dashboards      FastAPI
                                          /search /compare
                                          /ask /route /kpis
```

The full diagram source is [`docs/architecture.mmd`](docs/architecture.mmd)
(Mermaid). Design rationale and trade-offs: [`docs/design.md`](docs/design.md).

### Why a medallion layout

`bronze` keeps the raw JSON exactly as it arrived and is never rewritten. When
the enrichment logic changes — and it changed four times while building this —
we replay from bronze rather than from Kafka, whose retention is finite. `silver`
is the enriched analytical table. `gold` holds the KPI aggregates the dashboards
read.

### Why `foreachBatch` rather than three streaming sinks

One micro-batch has to fan out to two Parquet paths *and* Elasticsearch. Three
independent `writeStream` queries would each re-execute the expensive enrichment
stage. `foreachBatch` gives us the batch API inside a streaming query, so
enrichment runs once per batch and all three sinks share it.

### Delivery semantics

At-least-once from Kafka; Spark may replay a micro-batch after a failure. That
is safe here because every sink is idempotent — `ticket_id` is a SHA-1 of the
ticket's content, and Elasticsearch writes are keyed on it via `es.mapping.id`.
Replaying updates documents in place instead of duplicating them. The result is
effectively-once *in the sinks* without needing transactions.

---

## The AI capability (Part B)

The brief asks for at least one option from section 6.2. This implements four,
with **(b) embeddings and semantic search** as the graded centrepiece:

| # | option | where |
|---|---|---|
| **b** | **Embeddings + semantic search** — 384-d vectors in an Elasticsearch `dense_vector` field, cosine kNN | `core/embeddings.py`, `ai/search.py` |
| c | **RAG** — grounded question answering with citation checking | `ai/rag.py`, `POST /ask` |
| a | **LLM enrichment** — classify, tag entities, summarise, score sentiment | `ai/llm.py` |
| f | **ML / predictive analytics** — k-NN and centroid routing classifiers | `ai/router.py` |
| g | automated insight narrative over computed KPIs | `ai/rag.py::narrate_insights` |

### Two embedding backends, one interface

`paraphrase-multilingual-MiniLM-L12-v2` is the intended model: the corpus is 57%
English and 43% German, and a multilingual model puts both languages in a shared
vector space, so a German query can retrieve an equivalent English ticket.

There is also an **offline backend** — word + character TF-IDF reduced to 384
dimensions with truncated SVD (i.e. LSA). No download, no network, deterministic.
Both emit L2-normalised vectors of identical width, so the Elasticsearch mapping
and the kNN query are unchanged between them. `EMBEDDING_BACKEND=auto` prefers
the neural model and silently falls back.

**All numbers in this repository were produced with the offline backend**, because
the environment the project was built in could not download model weights. What
that means for the results is set out honestly below.

### Safety: PII never reaches a hosted model

`core/pii.py` is the enforcement point, and `assert_safe_for_external()` runs
immediately before every hosted-LLM request. It **fails closed** — if redaction
has not removed emails, phone numbers, IBANs or card numbers, the call raises
rather than degrading.

The dataset publisher had already substituted placeholders like `<name>` and
`<tel_num>`, but relying on an upstream promise is not a control, so we re-scan.
What the scan actually found, stated precisely: PII patterns in 1,053 tickets
(3.7%), of which **1,052 are the publisher's own placeholder tokens** and exactly
**one** is a real phone-number pattern that escaped their substitution. So the
upstream anonymisation is close to airtight — but "close to" is why the gate
exists, and the one leak it caught is the argument for keeping it.

### RAG grounding is enforced, not requested

Asking a model to cite only what it was given is a prompt, not a guarantee.
`ai/rag.py` post-checks every citation against the set of retrieved ticket ids
and strips any the model invented. With `LLM_PROVIDER=none` the same retrieval
runs and the answer becomes extractive — less fluent, still grounded, and the
demo works with no API key.

---

## The retrieval result, and how it changed

This is the measurement worth reading twice, because the answer flipped when the
embedding did.

**With the offline TF-IDF + SVD backend (i.e. LSA):**

| method | queue purity@5 | precision@5 | MRR |
|---|---|---|---|
| semantic | 0.294 | 0.713 | 0.823 |
| keyword (BM25) | **0.322** | 0.728 | **0.859** |
| hybrid (RRF) | 0.294 | **0.745** | 0.844 |

Semantic search *lost*. That was the expected outcome once you look at what LSA
is: a linear projection of the same term-document statistics BM25 already scores.
It compresses lexical information; it adds no semantic information from outside
the corpus, so it cannot systematically beat the lexical baseline at lexical
matching.

**With `paraphrase-multilingual-MiniLM-L12-v2`:**

| method | queue purity@5 | precision@5 | MRR |
|---|---|---|---|
| semantic | **0.346** | 0.744 | 0.835 |
| keyword (BM25) | 0.322 | 0.728 | **0.859** |
| **hybrid (RRF)** | **0.352** | **0.758** | **0.876** |

Semantic now beats BM25 on queue purity and precision, and **hybrid wins on all
three**. The neural model brings outside knowledge — it was trained on
paraphrase pairs, not on this corpus — which is exactly what LSA could not do.

BM25 still leads on MRR (0.859 vs 0.835). That is consistent rather than
contradictory: lexical matching is very good at putting an *exact* term match at
rank 1, while the embedding is better across the whole top-5. Fusing the two with
RRF beats either alone, which is why hybrid is the default in the API.

### The cross-lingual probe did not work, and we know why — and fixed it

| query | semantic hits in the other language | keyword |
|---|---|---|
| `Rechnung falsch berechnet` | 0 | 0 |
| `Server ist ausgefallen` | 1 | 0 |
| `cannot reset my password` | 0 | 0 |

Near zero, even with the multilingual model. **This is a flaw in the probe, not
a verdict on the model.** The corpus holds 12,249 German tickets, so a German
query finds excellent German matches and no English ticket ever reaches the top
5 — the measurement is swamped by same-language competition before cross-lingual
ability can show.

Measuring it properly means restricting the candidate set: run a German query
with `language=en` filtered on, and check whether the retrieved English tickets
are topically right. The API already supports that filter
(`/search?q=...&language=en`); `evaluate_cross_lingual_retrieval`
(`src/tickets/ai/evaluate.py`) now does exactly that, scored with the same
shared-tag proxy as the main retrieval table so the numbers are comparable —
on the full 28,587-ticket corpus, 250 queries:

| method | queue purity@5 | precision@5 | MRR |
|---|---|---|---|
| semantic | 0.313 | **0.685** | 0.787 |
| keyword | 0.266 | 0.660 | 0.768 |
| hybrid | 0.300 | 0.698 | 0.799 |

Forced to match across the language boundary, semantic retrieval still beats
keyword search, and by a **wider** margin than in the same-pool table above —
2.5 points of precision@5 (0.685 vs. 0.660) versus 1.6 points when both search
the same mixed pool (0.744 vs. 0.728). BM25 cannot match German query terms
against English ticket text at all; this is the scenario the embedding earns
its cost on. An earlier run of this same evaluation on a 5,000-ticket sample
showed a much larger gap (11.5 points) — a reminder that a small sample can
overstate an effect even when the direction is right. Full table and
methodology note: `docs/RESULTS.md`.

## Insights from the data

- **The `language` label is broken in one direction only.** Overall mismatch is
  11.7%, which sounds like uniform noise. It is not: tickets labelled `de` are
  mislabelled **27.2%** of the time, while tickets labelled `en` are mislabelled
  **0.05%** of the time (n=12,210 and 16,234). More than a quarter of the
  "German" corpus is actually English text. That asymmetry points at a specific
  upstream defect — most likely a generation or import step that stamped `de` on
  English rows — and it is invisible if you only look at the headline rate.
  Anything keyed off this field, such as canned replies or language-based
  routing, is misfiring on those rows today.
- **Urgency is concentrated, and not where volume is.** Service Outages and
  Maintenance is only 4.0% of tickets but **71.0%** of them are high priority;
  Technical Support is 58.6%. At the other end, Human Resources is 9.5% and
  General Inquiry 12.8%. Staffing to volume would badly under-resource the
  outage queue.
- **Sentiment tracks urgency closely.** Technical Support (53.8% negative) and
  Service Outages (53.5%) run hottest; Sales and Pre-Sales is coldest at 21.2%.
  The ranking is nearly the same as the high-priority ranking, which is a useful
  sanity check that the lexicon scorer is measuring something real.
- **Priority does not vary by language.** `de` 39.4% high, `en` 38.8%, corpus
  39.1%. The skew lives entirely in the queue dimension.
- **Keyword rules are a trap.** They look reasonable, they are easy to explain to
  a stakeholder, and they perform worse than a constant. Worth showing to anyone
  proposing to hand-write a routing rulebook.

---

## Project layout

```
src/tickets/
  config.py               all configuration, environment-driven
  core/                   PURE PYTHON — no Spark, no ES, fully unit tested
    schema.py             canonical ticket, validation, ES mappings
    textproc.py           normalisation, tokenisation, language detection
    pii.py                redaction + the fail-closed external-call gate
    sentiment.py          bilingual lexicon scorer
    classify.py           rule-based zero-shot baseline
    embeddings.py         neural + offline backends behind one interface
    enrich.py             the transformation the Spark UDFs apply
    kpis.py               reference KPI implementation
  ingest/producer.py      CSV -> Kafka
  spark/
    session.py            SparkSession + S3A/MinIO wiring
    stream_job.py         Kafka -> bronze/silver -> Elasticsearch
    batch_kpis.py         silver -> gold + Elasticsearch, self-verifying
  ai/
    search.py             BM25, in-memory and Elasticsearch indexes
    router.py             k-NN and centroid routing classifiers
    rag.py                grounded question answering
    llm.py                pluggable provider, offline fallback
    evaluate.py           the evaluation harness
  serving/
    es_client.py          index management + bulk loading
    api.py                FastAPI serving layer
  offline_pipeline.py     the whole pipeline with no infrastructure
tests/                    87 unit tests
docs/                     design document, results, architecture diagram
scripts/                  sample builder, results renderer, push helper
```

### Why `core/` contains no Spark

Every transformation is a pure function that takes and returns plain Python. The
Spark job calls exactly these functions inside its UDFs. That is what makes the
test suite meaningful — a green run says something about the pipeline, not just
about a helper module — and it is why the entire thing can also run without Spark
at all. `batch_kpis.py` goes further and cross-checks its native DataFrame
aggregates against the pure-Python reference, so the two implementations cannot
silently drift.

---

## Testing and verification

```bash
make test        # 87 unit tests
```

Honest scope, since it affects how much you should trust what is here:

| verified | how |
|---|---|
| all `core/` and `ai/` logic | 87 unit tests, all passing |
| the full pipeline over all 28,587 real rows | `offline_pipeline`, end to end, numbers in `docs/RESULTS.md` |
| PII gate blocks unredacted text | unit test asserts it raises |
| hallucinated LLM labels are rejected | unit test with a stub returning an invented queue |
| hallucinated RAG citations are stripped | unit test |
| KPI maths | unit tests + Spark/reference cross-check in `batch_kpis.py` |

The Docker Compose stack, the Spark streaming and batch jobs, and the
Elasticsearch writes and kNN queries were authored with no reachable cluster to
test against. They have since been **run against the live stack and verified**:
50 real tickets through Kafka → Spark → MinIO (bronze then silver) →
Elasticsearch, the `dense_vector` mapping confirmed correct via the live index
(not assumed), the batch job's self-verification against the pure-Python
reference passing, and `/search`, `/route`, `/ask` all returning correct
results against the live index. `docs/DEMO.md`'s troubleshooting table lists
every real bug that surfaced on the first run and how each was fixed — six of
them, from two retired Docker Hub image tags to a missing `pyarrow` dependency.
Read it before re-running; it will save you the hour it cost the first time.

**Run `make up`** (uses `docker-compose.slim.yml`, sized for an 8GB machine) to
bring the stack up yourself, and budget time for the first `spark-submit`,
which resolves several hundred MB of Ivy jars.

---

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example). The three
that matter most:

| variable | default | note |
|---|---|---|
| `EMBEDDING_BACKEND` | `auto` | `auto` \| `sentence-transformers` \| `offline` |
| `EMBEDDING_DIM` | `384` | **changing this requires recreating the ES index** — `dense_vector` dimensions are fixed at mapping time |
| `LLM_PROVIDER` | `none` | `none` \| `anthropic` \| `openai` \| `ollama` |

If you switch embedding backends, delete and recreate the index. Vectors from
different models are not comparable, and Elasticsearch will happily mix them and
return nonsense.

---

## Credits

- Dataset: Tobias Bueck, *Customer IT Support — Ticket Dataset*, CC BY-NC 4.0.
  See [`data/README.md`](data/README.md).
- Code in this repository: MIT, see [`LICENSE`](LICENSE).

Per the course's honest-use section: an AI assistant was used to help write and
review this code. Every design decision, trade-off and limitation documented here
is one we can explain and defend, and the sections above deliberately record what
was measured, what was not, and where the results contradicted the design.
