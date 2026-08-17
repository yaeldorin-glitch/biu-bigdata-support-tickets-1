# Demo runbook

The brief requires "a live demo, or a recorded demo when a live one is not
possible" (section 7) and weights presentation + demo at 10%. This is the script
for it.

**Record it in advance even if you plan to present live.** A recording is an
accepted deliverable, and it means a container that refuses to start on the day
costs you nothing. Windows has a recorder built in: `Win + G`.

Total runtime once images are pulled: about 8 minutes. Budget 30-40 minutes for
the very first run — Docker pulls ~3GB of images and Spark resolves ~200MB of
Ivy jars.

---

## Before you record

```powershell
cd C:\path\to\biu-bigdata-support-tickets
pip install -e .
```

Check three things:

1. **Docker Desktop is running.** On an 8GB machine, always use
   `docker-compose.slim.yml`, never the default file — every command below
   assumes it (`-f docker-compose.slim.yml`). The slim file drops Kibana and
   caps Elasticsearch's heap; the default file assumes 8GB free inside Docker
   and gets Elasticsearch OOM-killed on startup.
2. **The full dataset is at `data\raw\tickets.csv`.** Without it everything runs
   on the 300-row sample and the numbers will not match your slides.
3. **Nothing else is using ports** 9092, 9000, 9001, 9200, 4040, 8000. (4040 is
   the Spark UI; 5601 would be Kibana, but it is not present in the slim stack.)

**One-time setup: fit the offline embedding model.** The Spark container runs
with `EMBEDDING_BACKEND=offline` (loading the 471MB neural model into every
executor does not fit the memory budget), and that backend needs a *fitted*
model file — it cannot fit itself on a 50-ticket streaming batch. Fit it once
against a real sample and it is reused on every future run:

```powershell
EMBEDDING_BACKEND=offline tickets-pipeline --limit 3000
```

This takes seconds (it is TF-IDF+SVD, not the neural model) and writes
`output/offline_embedding_model.joblib`, which the Spark container picks up
automatically via `OFFLINE_EMBEDDING_MODEL` in `docker-compose.slim.yml`.
Skipping this step fails with `RuntimeError: the offline embedding backend
needs either a fitted model file or a corpus to fit on`.

Do a full dry run start to finish before recording. The first run is the one
that finds the problems.

---

## The sequence

### 0. Safety net first (30 seconds)

Run this *before* touching Docker, and leave the output on screen:

```powershell
tickets-pipeline --full
```

This produces every number in your slides with no infrastructure at all. If
anything later fails, you still have a working demo. Say so out loud — "this is
the same logic the Spark job runs, without the cluster" — because it is true and
it is a point in your favour, not an excuse.

### 1. Bring up the stack (2 minutes, mostly waiting)

```powershell
docker compose -f docker-compose.slim.yml up -d
docker compose -f docker-compose.slim.yml ps
```

Talk over the wait. The services are: Kafka (broker), MinIO (object store),
Elasticsearch (NoSQL + vector index), Spark (local mode). That is four of the
course technologies in one screen — worth pointing at explicitly, since "use of
course technologies" is 20% of the grade. (The slim stack drops Kibana to fit
an 8GB machine; the FastAPI `/kpis` endpoint covers the same numbers.)

Wait until Elasticsearch answers:

```powershell
curl http://localhost:9200/_cluster/health
```

### 2. Create the index (10 seconds)

```powershell
python -c "from tickets.serving.es_client import create_indices; print(create_indices())"
```

**Say why this step exists**, it is a good detail: the `embedding` field has to
be declared as `dense_vector` *before* the first write. If you let Elasticsearch
infer the mapping it types it as a plain float array, kNN search stops working,
and field types are immutable — the only fix is a full reindex.

### 3. Stream the data (3 minutes)

Two terminals. First the producer:

```powershell
tickets-producer --rate 200
```

Then the streaming job. Note the target is `docker/spark/run_module.py`, not
`stream_job.py` directly: `spark-submit` executes its target file as a bare
script with no parent package, and `stream_job.py` uses relative imports
(`from ..config import ...`) that only resolve when Python loads it as part of
the `tickets` package. The launcher runs it as a real module import instead —
the same effect as `python -m tickets.spark.stream_job`, which `spark-submit`
has no flag for:

```powershell
docker compose -f docker-compose.slim.yml exec spark spark-submit `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 `
  /opt/project/docker/spark/run_module.py tickets.spark.stream_job --trigger-seconds 5
```

While it runs, show the data landing in three places:

- **MinIO** — <http://localhost:9001> (`minioadmin` / `minioadmin`) — the `lake`
  bucket — `bronze/` filling with Parquet, then `silver/`
- **Elasticsearch** — `curl http://localhost:9200/tickets/_count`
- **Spark UI** — <http://localhost:4040> — the streaming query, batch by batch

The bronze/silver split is worth one sentence: bronze is raw and immutable, so
when the enrichment logic changes you replay from there rather than from Kafka,
whose retention is finite.

### 4. Batch KPIs (1 minute)

Stop the streaming job first (`Ctrl+C`, or `docker compose -f
docker-compose.slim.yml exec spark pkill -f SparkSubmit` — it is a persistent
query and will not exit on its own), then:

```powershell
docker compose -f docker-compose.slim.yml exec spark spark-submit `
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 `
  /opt/project/docker/spark/run_module.py tickets.spark.batch_kpis
```

Point out the line in the log that says verification passed — the Spark
aggregates are cross-checked against the pure-Python reference implementation, so
the two cannot silently drift.

### 5. The AI capability (2 minutes) — this is the part that matters

```powershell
uvicorn tickets.serving.api:app --port 8000
```

Open <http://localhost:8000/docs>.

**a. Semantic vs keyword, side by side.** This is the single most convincing
screen in the demo:

```
GET /compare?q=I was charged twice for my subscription
```

Three rankings for one query. Show a result that semantic search found and BM25
missed because it shares no words with the query.

**b. Routing a ticket that does not exist yet.** Type something new:

```
POST /route
{"text": "Our production database has been unreachable for two hours and customers cannot check out"}
```

The response gives the rule-based guess, the k-NN guess and the linear SVM guess
with its confidence. Say the number out loud: **64.8% accuracy against a 29.2%
majority-class baseline**, and macro-F1 0.645 against 0.045.

**c. RAG.**

```
POST /ask
{"question": "What are the most common billing complaints?"}
```

Point out that with no API key configured it answers extractively from the
retrieved tickets rather than failing — and that invented citations are stripped
before you ever see them.

### 6. KPIs (1 minute)

```
GET /kpis
```

The slim stack drops Kibana (see "Before you record" above), so this is the
real demo path for the aggregates, not a fallback — it reads the same
`ticket_kpis` data straight from Elasticsearch (or `output/kpis.json` offline).

Show two findings that change a decision:

- Tickets labelled `de` are mislabelled **27.2%** of the time; tickets labelled
  `en`, **0.05%**. A one-directional upstream defect, invisible in the 11.7%
  headline.
- Service Outages is 4.0% of volume but **71.0%** high priority. Staffing to
  volume would under-resource it badly.

---

## Shutting down

```powershell
docker compose -f docker-compose.slim.yml down        # keeps the data
docker compose -f docker-compose.slim.yml down -v     # deletes the volumes too
```

---

## When something breaks

Found and fixed once this stack was actually started for the first time —
listed here so a repeat is a two-minute fix, not a re-investigation:

| symptom | cause | fix |
|---|---|---|
| Elasticsearch exits right after start | Docker memory below 8GB, or using `docker-compose.yml` instead of the slim file | always `-f docker-compose.slim.yml`; Settings → Resources → Memory |
| Producer connects then times out | wrong bootstrap address | host → `localhost:9092`, container → `kafka:29092` |
| `spark-submit` hangs on first run | Ivy resolving jars | wait it out once; they are cached afterwards |
| `bitnami/kafka:3.7` / `bitnami/spark:3.5.1: not found` | Bitnami retired free-tier version tags | already fixed in `docker-compose.slim.yml` / `docker/spark/Dockerfile` → `bitnamilegacy/*` |
| `FileNotFoundError` in `.ivy2/cache` during Ivy resolve | the `ivy-cache` named volume is created root-owned but the container runs as non-root UID 1001 | one-time: `docker compose -f docker-compose.slim.yml exec -u root spark chown -R 1001:0 /opt/bitnami/spark/.ivy2` |
| `ImportError: attempted relative import with no known parent package` | `spark-submit` runs its target as a bare script, not a package module | target `docker/spark/run_module.py <module>` instead of the `.py` file directly (see step 3) |
| `ModuleNotFoundError: No module named 'pyarrow'` | Spark's `mapInPandas` needs Arrow; it was missing from `requirements.txt` | already fixed — `pyarrow` is now a dependency |
| `RuntimeError: the offline embedding backend needs either a fitted model file or a corpus to fit on` | streaming a batch alone gives the offline backend nothing to fit on | run the one-time `EMBEDDING_BACKEND=offline tickets-pipeline --limit 3000` step above first |
| kNN query rejected by Elasticsearch | index created by dynamic mapping | delete the index, run step 2, re-stream |
| `No module named 'tickets'` | not installed, or wrong folder | `pip install -e .`, then use `tickets-pipeline` |
| Port already in use | leftover containers | `docker compose -f docker-compose.slim.yml down` then retry |

**If the stack fails mid-demo**, fall back to step 0 and to
`OFFLINE_API=1 uvicorn tickets.serving.api:app` — the same API, same endpoints,
in-memory index, no infrastructure. Practise this fallback once so you can do it
calmly.

---

## Questions you should expect

The brief allows 3 minutes of questions and weights understanding above
everything else. The likely ones:

**"Why ELT and not ETL?"** The broker holds raw untransformed messages and Spark
does the transformation downstream. That means we can change the enrichment logic
and replay from bronze without re-extracting from source.

**"What happens if a Spark batch fails halfway?"** Kafka gives at-least-once, so
the batch replays. Every sink is idempotent because `ticket_id` is a SHA-1 of the
ticket's content — a replay overwrites the same Elasticsearch documents instead
of duplicating them. Effectively-once in the sinks, without needing transactions.

**"Why is semantic search not beating keyword search?"** (if asked about the
offline-backend numbers in `docs/CEILING.md`) Because those come from the
offline TF-IDF+SVD backend, which is LSA — a linear projection of the same
statistics BM25 already scores. It compresses lexical information and adds no
outside semantic knowledge. With the neural model — what the live demo and
`docs/RESULTS.md` actually use — semantic wins: 0.744 vs. 0.728 precision@5,
and the gap widens to 0.685 vs. 0.660 when the query and its match are forced
to be in different languages, where BM25 cannot share a single word with a
German query against an English ticket.

**"Why is accuracy only 65%?"** Because four of the ten queues — Technical,
Product, IT Support and Customer Service — overlap semantically, and every large
confusion is inside that cluster. The queues that are semantically distinct do
fine: Billing recall 0.82, Service Outages 0.75. Top-3 accuracy is 87.3%, so as a
suggestion tool rather than an auto-filer it is already strong.

**"Did you write this code?"** Answer honestly — the brief explicitly permits an
AI assistant and explicitly requires that you can explain every line. Read
through `src/tickets/core/` before the demo; it is eight files of plain Python
with no framework, and it is where all the real logic lives.
