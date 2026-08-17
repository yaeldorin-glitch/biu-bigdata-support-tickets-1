# Project context

Read this first. It is the handoff note for anyone — human or assistant — picking
this project up.

## What this is

A university course project: **BIU 8688697201, Big Data and AI**. An end-to-end
ELT pipeline over 28,587 multilingual customer support tickets, with an
embedding-based AI layer. Kafka → Spark Structured Streaming → MinIO
(bronze/silver/gold) → Elasticsearch, plus a FastAPI serving layer doing semantic
search, RAG and ticket routing.

The owner is a student who needs to **present this to a class and defend every
design decision**. Optimising for cleverness at the cost of explainability is the
wrong trade here. The course brief says outright: *"The single most important
factor is understanding. A simpler project that you fully grasp and can discuss
beats a complex one you cannot explain."*

Start with `README.md`, then `docs/design.md`.

---

## The machine this runs on

These constraints are load-bearing. Ignoring them wastes hours.

- **Windows 11, PowerShell.** Not bash. `VAR=value` on one line does not work;
  use `$env:VAR = "value"`.
- **7.8 GB RAM total.** WSL 2 is capped at 5GB via `%USERPROFILE%\.wslconfig`.
  **Always use `docker-compose.slim.yml`, never `docker-compose.yml`** — the
  default file assumes Docker can have 8GB and Elasticsearch is OOM-killed on
  startup. The slim file peaks at ~3.8GB, drops Kibana, and runs Spark in local
  mode instead of master + worker.
- **The Windows username contains a space and Hebrew characters.** Paths break in
  three ways: they must be quoted, they get mangled when copied through a
  terminal because of RTL rendering, and `cd` to a hand-typed path frequently
  fails with "does not exist" even when the path is right. **Do not construct
  absolute paths by hand.** Use `$PSScriptRoot`, `$HOME`, or find the folder by
  searching for a known file.
- **CPU only, no GPU.** Embedding all 28,587 tickets with the neural model takes
  **~25 minutes**. Do not trigger a full re-embed casually — it is the single
  most expensive operation in the project.

---

## Current state

### Verified working

- `pip install -e .` and the `tickets-pipeline` console script.
- **The full pipeline has been run end to end on all 28,587 tickets with the
  neural backend** (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  471MB, now cached). Every figure in `README.md`, `docs/RESULTS.md` and
  `docs/presentation.pptx` comes from that run.
- 87 unit tests pass (`python -m pytest tests/`).
- The repo is published at `github.com/yaeldorin-glitch/biu-bigdata-support-tickets-1` (public).
- **The Docker stack has been started and verified end to end**, for the first
  time, against `docker-compose.slim.yml`: 50 real tickets sent through Kafka →
  Spark Structured Streaming → MinIO (bronze then silver, medallion layout
  intact) → Elasticsearch (`dense_vector`, 384 dims, cosine, kNN-indexed —
  confirmed via the actual mapping, not assumed). The batch KPI job's
  self-verification against the pure-Python reference passed (83 KPIs, no
  mismatches). `/search`, `/route`, and `/ask` all returned correct results
  against the live index. See `docs/DEMO.md`'s troubleshooting table for the
  real bugs this surfaced and how each was fixed — read it before re-running,
  it will save you the same hour it cost the first time.

### Still open

- **`demo.ps1` never existed as a file** — it was the planned name in early
  notes, but what actually got built and committed is `run.ps1`, and the
  README documents `run.ps1`. Nobody has run `run.ps1` against a live stack;
  the verification above was done with the raw `docker compose` /
  `spark-submit` commands directly. Run `run.ps1` once before relying on it.
- **Nobody has recorded the demo yet** — required deliverable, 10% of the
  grade, and the stack is now in a state where that should go smoothly.

---

## What is left to do

| # | task | status |
|---|---|---|
| 1 | Get the Docker stack running | **done — see "Verified working" above** |
| 2 | Run `run.ps1` once to confirm the guided script itself works | **done — `run.ps1 -Stack` brought up all 4 containers healthy; the producer and Spark streaming job were then run manually and streamed 17,354 real tickets through to Elasticsearch before being stopped** |
| 3 | Record a screen demo (required deliverable, 10% of the grade) | open, no longer blocked — the stack has real data in it and is ready to go |
| 4 | Owner reads and understands `src/tickets/core/` | ongoing, hers to do |
| 5 | Confirm team size | **done — team of 3: Yael Cohen, Itay Nuri, Yahel Menachem** |

`docs/presentation_he/` holds a separate Hebrew classroom deck (`מצגת.pptx`,
10 slides with real charts) and a full speaking script (`תסריט.md`, split
across three presenters with a live-demo section and an anticipated-Q&A list)
for presenting this to the class. This is distinct from `docs/presentation.pptx`,
which is the official English deliverable generated from `output/report.json`
by `scripts/build_deck.js` — the Hebrew deck is hand-authored and not
regenerated by that script.

`docs/DEMO.md` has the full runbook, a troubleshooting table with every bug
found so far and its fix, and the questions to expect in the viva. When
something fails, read the container logs before changing code:

```powershell
docker compose -f docker-compose.slim.yml logs elasticsearch --tail 50
docker compose -f docker-compose.slim.yml ps
```

`exit code 137` means the OOM killer took it — reduce memory, do not retry.

---

## Design decisions to preserve

Do not "simplify" these away. Each was deliberate and each is something the owner
has to be able to defend.

- **`src/tickets/core/` contains no Spark, no Kafka, no Elasticsearch.** Every
  transformation is a pure function; the Spark UDFs call exactly those functions.
  This is what makes the test suite meaningful and what lets the whole pipeline
  run without any infrastructure. Keep it that way.
- **`ticket_id` is a SHA-1 of the ticket's content**, not a counter. Kafka gives
  at-least-once; content addressing makes every sink idempotent, so a replayed
  micro-batch overwrites rather than duplicates. Effectively-once in the sinks
  without transactions.
- **`foreachBatch` rather than three `writeStream` sinks** — one micro-batch fans
  out to two Parquet paths and Elasticsearch, and enrichment must run once per
  batch, not once per sink.
- **The `dense_vector` mapping is created before the first write.** Elasticsearch
  would otherwise infer a plain float array, which cannot be kNN-searched, and
  field types are immutable. `create_indices()` must run before streaming.
- **Two embedding backends behind one interface.** The offline TF-IDF+SVD backend
  is not dead code — it is why the pipeline runs with no downloads, and it is the
  classical baseline the neural model is measured against.
- **macro-F1 is reported alongside accuracy** everywhere. The majority class is
  29.3%; accuracy alone hides that the keyword rules score *below* a constant.
- **Three demonstrated AI capabilities, not four.** `ai/llm.py` genuinely
  supports Anthropic/OpenAI/Ollama for per-ticket classification, entity
  extraction and summarisation — it is real, tested code, not a stub. But every
  run reported anywhere in this project (the full 28,587-ticket run, the Docker
  verification) used `LLM_PROVIDER=none`, so enrichment always took the
  deterministic rule-based fallback, never the live-model path. Do not count it
  as a fourth capability alongside embeddings/semantic search, RAG and ML
  routing in README, design.md, either presentation deck, or the artifacts —
  that would be claiming something that was never actually run. If it is ever
  exercised against a real provider and the result is measured, promote it then.

---

## Honesty rules for this project

The brief requires that the owner can explain everything, and the documentation
is deliberately written to record what was *not* verified. Preserve that.

- If you add a claim, measure it first. If you cannot measure it, say so in the
  same sentence.
- `docs/CEILING.md` records experiments that **failed** — ensembles, trees,
  per-language models, a two-stage hierarchy. Those negative results are worth
  marks. Do not delete them.
- The unrestricted cross-lingual probe returns ~0 on the full corpus, and that
  is a property of the *probe*, not the model: with 12,249 German tickets, a
  German query's unrestricted top-5 is dominated by German tickets on volume
  alone, before cross-lingual ability gets a chance to show up. This gap is no
  longer just documented — it is fixed and measured.
  `evaluate_cross_lingual_retrieval` (`src/tickets/ai/evaluate.py`) restricts
  the candidate pool to the *other* language before ranking (the same fix
  `/search?q=...&language=en` gives a live user), then scores it with the
  identical shared-tag proxy as the main retrieval table. On the full
  28,587-ticket corpus, 250 queries: semantic scores 0.685 precision@5
  cross-lingually vs. keyword's 0.660 — a *wider* gap (2.5 points) than the
  1.6-point gap in same-pool retrieval, because BM25 cannot match German query
  terms against English ticket text at all. (An earlier run on a 5,000-ticket
  sample showed an 11.5-point gap — real direction, overstated by sample size.)
  See `docs/RESULTS.md`'s "Cross-lingual retrieval, language-filtered" section.

---

## Key numbers

Routing, stratified 25% held-out split, n = 7,145:

| method | accuracy | macro-F1 |
|---|---|---|
| majority class | 29.2% | 0.045 |
| keyword rules | 27.4% | 0.208 |
| embedding centroid | 22.0% | 0.205 |
| k-NN over neural embeddings (k=5) | 64.6% | 0.605 |
| **linear SVM, word+char TF-IDF** | **64.8%** | **0.645** |

Retrieval, 250 queries, k=5, proxy relevance (≥2 shared tags):

| method | queue purity@5 | precision@5 | MRR |
|---|---|---|---|
| semantic | 0.346 | 0.744 | 0.835 |
| keyword (BM25) | 0.322 | 0.728 | 0.859 |
| **hybrid (RRF)** | **0.352** | **0.758** | **0.876** |

Operational findings: German-labelled tickets are mislabelled **27.2%** of the
time vs **0.05%** for English ones; Service Outages is 4.0% of volume but
**71.0%** high priority.

**If you re-run the pipeline, regenerate the documents too** — every figure above
is produced from `output/report.json`, not typed by hand:

```powershell
python scripts/render_results.py --write   # docs/RESULTS.md
node scripts/build_deck.js                 # docs/presentation.pptx (needs Node)
```

---

## Data

`data/raw/tickets.csv` (26MB, 28,587 rows) is **gitignored** — it is CC BY-NC 4.0
and not ours to redistribute. A stratified 300-row sample is committed at
`data/sample/tickets_sample.csv`, and the runner falls back to it automatically.

**The sample will not reproduce the numbers above.** 300 rows is far too few —
k-NN in particular needs neighbours to vote and underperforms the keyword rules
there. The sample proves the pipeline runs; it does not reproduce the findings.
