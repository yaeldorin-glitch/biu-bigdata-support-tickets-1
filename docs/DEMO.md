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

1. **Docker Desktop is running**, with Settings → Resources → Memory at **8GB or
   more**. Below that, Elasticsearch is OOM-killed during startup and the whole
   demo dies at step 2.
2. **The full dataset is at `data\raw\tickets.csv`.** Without it everything runs
   on the 300-row sample and the numbers will not match your slides.
3. **Nothing else is using ports** 9092, 9000, 9001, 9200, 5601, 8080, 8000.

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
docker compose up -d
docker compose ps
```

Talk over the wait. The services are: Kafka (broker), MinIO (object store),
Elasticsearch (NoSQL + vector index), Kibana (dashboards), Spark (master +
worker). That is five of the course technologies in one screen — worth pointing
at explicitly, since "use of course technologies" is 20% of the grade.

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

Then the streaming job:

```powershell
docker compose exec spark spark-submit `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 `
  /opt/project/src/tickets/spark/stream_job.py
```

While it runs, show the data landing in three places:

- **MinIO** → <http://localhost:9001> (`minioadmin` / `minioadmin`) → the `lake`
  bucket → `bronze/` filling with Parquet, then `silver/`
- **Elasticsearch** → `curl http://localhost:9200/tickets/_count`
- **Spark UI** → <http://localhost:4040> → the streaming query, batch by batch

The bronze/silver split is worth one sentence: bronze is raw and immutable, so
when the enrichment logic changes you replay from there rather than from Kafka,
whose retention is finite.

### 4. Batch KPIs (1 minute)

```powershell
docker compose exec spark spark-submit `
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 `
  /opt/project/src/tickets/spark/batch_kpis.py
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

### 6. Kibana (1 minute)

<http://localhost:5601> → Discover → create a data view on `ticket_kpis`.

Show two findings that change a decision:

- Tickets labelled `de` are mislabelled **27.2%** of the time; tickets labelled
  `en`, **0.05%**. A one-directional upstream defect, invisible in the 11.7%
  headline.
- Service Outages is 4.0% of volume but **71.0%** high priority. Staffing to
  volume would under-resource it badly.

---

## Shutting down

```powershell
docker compose down        # keeps the data
docker compose down -v     # deletes the volumes too
```

---

## When something breaks

| symptom | cause | fix |
|---|---|---|
| Elasticsearch exits right after start | Docker memory below 8GB | Settings → Resources → Memory |
| Producer connects then times out | wrong bootstrap address | host → `localhost:9092`, container → `kafka:29092` |
| `spark-submit` hangs on first run | Ivy resolving jars | wait it out once; they are cached afterwards |
| kNN query rejected by Elasticsearch | index created by dynamic mapping | delete the index, run step 2, re-stream |
| `No module named 'tickets'` | not installed, or wrong folder | `pip install -e .`, then use `tickets-pipeline` |
| Port already in use | leftover containers | `docker compose down` then retry |

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

**"Why is semantic search not beating keyword search?"** Because these numbers
come from the offline TF-IDF+SVD backend, which is LSA — a linear projection of
the same statistics BM25 already scores. It compresses lexical information and
adds no outside semantic knowledge. The multilingual neural model is expected to
change that; we have not measured it, so we do not claim it.

**"Why is accuracy only 65%?"** Because four of the ten queues — Technical,
Product, IT Support and Customer Service — overlap semantically, and every large
confusion is inside that cluster. The queues that are semantically distinct do
fine: Billing recall 0.82, Service Outages 0.75. Top-3 accuracy is 87.3%, so as a
suggestion tool rather than an auto-filer it is already strong.

**"Did you write this code?"** Answer honestly — the brief explicitly permits an
AI assistant and explicitly requires that you can explain every line. Read
through `src/tickets/core/` before the demo; it is eight files of plain Python
with no framework, and it is where all the real logic lives.
