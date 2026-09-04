# Demo runbook

Everything you actually need for the recording/presentation. If a line here
doesn't make sense once you're doing it, ask about that one line.

For *understanding* the project (concepts, why things are built this way),
read "מסע הכרטיס" — that's a separate guide, not this file:
<https://claude.ai/code/artifact/a1d650aa-679c-423d-8010-856292f1f0f7>

---

## Before the presentation (once, whenever you have 20-30 min)

1. Read "מסע הכרטיס" once, start to finish.
2. Open the demo link yourself (see below) and try the 4 examples in the
   table below.
3. Skim "Questions you should expect" at the bottom of this file.

---

## During the presentation

| # | Where | What | Say / show |
|---|---|---|---|
| 1 | Open Docker Desktop | click it, wait ~20-30s | — |
| 2 | PowerShell, inside the project folder | `.\run.ps1 -Stack` — wait ~2-3 min | while waiting: "this brings up Kafka, MinIO, Elasticsearch and Spark — four of the course technologies" |
| 3 | Same window, once it says "Uvicorn running" | leave this window open for the rest of the demo | this window *is* now serving the link |
| 4 | Browser | open `http://localhost:8000/docs` | "this is the live API, backed by 17,354 real tickets already indexed" |
| 5 | Browser, `/compare` | `q`: `I was charged twice for my subscription` | point at a semantic result sharing no words with the query — "this is what the AI capability buys us" |
| 6 | Browser, `/route` | `text`: `Our production database has been unreachable for two hours and customers cannot check out` | say out loud: **64.8% accuracy vs. a 29.2% majority-class baseline** |
| 7 | Browser, `/ask` | `question`: `What are the most common billing complaints?` | point at `citations` — "only built from real retrieved tickets" |
| 8 | Browser, `/kpis` | click Execute, no input | say: "German-labelled tickets are wrong 27.2% of the time vs. 0.05% for English; Service Outages is 4% of volume but 71% high-priority" |

That's the whole thing. You don't need the producer, the Spark streaming
job, or anything else — the index already has real data in it, and showing
already-real data honestly is enough.

---

## If it doesn't work

| symptom | fix |
|---|---|
| `docker compose ... ps` shows nothing, or everything says `Exited` | the machine was restarted or slept — data is safe, just run `.\run.ps1 -Stack` again |
| The link refuses to connect even after step 2 finished | wait a full minute — the first request loads the AI model, that delay is normal, not a hang |
| Docker Desktop is installed, but only some of the four services ever show up (Kafka, MinIO, Elasticsearch, Spark) | `.\run.ps1 -Stack` was never actually run to completion on this computer — opening Docker Desktop by itself doesn't create any of the four, only the command does. Run `.\run.ps1 -Stack` and let it finish; first time on a new computer takes ~30-40 min (~3GB download) |
| Nothing here works, and the files on this computer don't match what's described | this computer doesn't have the actual project — it has other, unrelated files (get the real thing: `git clone https://github.com/yaeldorin-glitch/biu-bigdata-support-tickets-1.git`, or the green **Code → Download ZIP** button on that page). None of these commands work against anything else |
| `/kpis` shows stale/small numbers that don't match your slides | the KPI batch job needs re-running after new data streams in — it does not update itself: `docker compose -f docker-compose.slim.yml exec spark spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 --conf spark.hadoop.es.http.timeout=5m docker/spark/run_module.py tickets.spark.batch_kpis` (the longer timeout matters — on this machine's tight memory, Elasticsearch can be too busy to answer the default timeout and the write silently fails, leaving old numbers in place) |
| Still stuck | `OFFLINE_API=1 uvicorn tickets.serving.api:app` — same API, no Docker needed, works instantly |

---

## Questions you should expect

**"Why ELT and not ETL?"** The broker holds raw untransformed messages;
Spark transforms them downstream. That means the enrichment logic can change
and be replayed from raw data without going back to the source.

**"What happens if a Spark batch fails halfway?"** Kafka gives
at-least-once, so it replays. Every sink is idempotent because `ticket_id`
is a SHA-1 of the ticket's content — a replay overwrites the same document
instead of duplicating it.

**"Why is accuracy only 65%?"** Four of the ten queues (Technical, Product,
IT Support, Customer Service) overlap semantically, and nearly all confusion
is inside that cluster. Queues that are semantically distinct do well —
Billing recall 0.82. Top-3 accuracy is 87.3%, so as a suggestion tool rather
than an auto-filer it's already strong.

**"Did you write this code?"** Answer honestly — the brief explicitly
permits an AI assistant and requires that you can explain every line. Read
`src/tickets/core/` before presenting; eight plain Python files, no
framework, where the real logic lives.
