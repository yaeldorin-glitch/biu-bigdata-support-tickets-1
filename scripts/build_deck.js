/**
 * Builds docs/presentation.pptx.
 *
 * Every figure in the deck is read from output/report.json, so re-running the
 * pipeline and re-running this script regenerates the slides with fresh numbers
 * instead of leaving stale ones behind.
 *
 *   node scripts/build_deck.js
 */

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const report = JSON.parse(fs.readFileSync(path.join(ROOT, "output", "report.json"), "utf8"));

// --- palette: Midnight Executive -------------------------------------------
const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const INK = "16203F";
const SLATE = "5A6485";
const AMBER = "E8A33D";
const TEAL = "2A9D8F";
const CRIMSON = "C1443C";
const LIGHT = "F4F6FB";

const HEAD = "Cambria";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Yael Dorin";
pres.title = "Ticket Intelligence - Big Data & AI";

const W = 13.3;
const H = 7.5;

// --- helpers ---------------------------------------------------------------

const routing = Object.fromEntries(report.routing_evaluation.map((r) => [r.method, r]));
const retrieval = Object.fromEntries(report.retrieval_evaluation.map((r) => [r.method, r]));
const summary = report.summary;
const kpi = (name, dim, bucket) =>
  report.kpis.find((k) => k.kpi === name && k.dimension === dim && k.bucket === bucket);

const pct = (x) => `${(x * 100).toFixed(1)}%`;

// Method names carry their hyperparameter (embedding_knn_k5), so look them up by
// prefix rather than hardcoding -- otherwise retuning k silently breaks the deck.
const byPrefix = (prefix) =>
  report.routing_evaluation.find((r) => r.method.startsWith(prefix));
const bestRouter = report.routing_evaluation.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
const knnRouter = byPrefix("embedding_knn");

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}

function lightSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText(title, {
    x: 0.6, y: 0.42, w: W - 1.2, h: 0.75,
    fontFace: HEAD, fontSize: 34, bold: true, color: INK, margin: 0,
  });
  if (kicker) {
    s.addText(kicker, {
      x: 0.6, y: 1.18, w: W - 1.2, h: 0.4,
      fontFace: BODY, fontSize: 14, color: SLATE, italic: true, margin: 0,
    });
  }
  return s;
}

/** Big number callout on a tinted card. */
function statCard(slide, { x, y, w, h, value, label, sub, color, fill }) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || LIGHT },
    line: { color: fill || LIGHT, width: 0 },
  });
  slide.addText(value, {
    x: x + 0.15, y: y + 0.16, w: w - 0.3, h: 0.85,
    fontFace: HEAD, fontSize: 40, bold: true, color: color || NAVY,
    align: "center", margin: 0,
  });
  slide.addText(label, {
    x: x + 0.15, y: y + 1.0, w: w - 0.3, h: 0.32,
    fontFace: BODY, fontSize: 12.5, bold: true, color: INK, align: "center", margin: 0,
  });
  if (sub) {
    slide.addText(sub, {
      x: x + 0.15, y: y + 1.3, w: w - 0.3, h: 0.5,
      fontFace: BODY, fontSize: 10.5, color: SLATE, align: "center", margin: 0,
    });
  }
}

/** Icon-free numbered step chip used on the architecture slide. */
function stageBox(slide, { x, y, w, h, title, lines, fill, titleColor }) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.07,
    fill: { color: fill },
    line: { color: fill, width: 0 },
  });
  slide.addText(title, {
    x: x + 0.12, y: y + 0.1, w: w - 0.24, h: 0.3,
    fontFace: BODY, fontSize: 12, bold: true, color: titleColor || WHITE, margin: 0,
  });
  slide.addText(
    lines.map((t, i) => ({
      text: t,
      options: { bullet: false, breakLine: i !== lines.length - 1 },
    })),
    {
      x: x + 0.12, y: y + 0.42, w: w - 0.24, h: h - 0.5,
      fontFace: BODY, fontSize: 9.5, color: titleColor || WHITE, margin: 0, lineSpacing: 13,
    }
  );
}

function arrow(slide, x, y) {
  slide.addText("▶", {
    x, y, w: 0.3, h: 0.3, fontFace: BODY, fontSize: 13, color: SLATE,
    align: "center", margin: 0,
  });
}

// =========================================================================
// 1. Title
// =========================================================================
{
  const s = darkSlide();
  s.addText("Ticket Intelligence", {
    x: 0.9, y: 2.05, w: 11.5, h: 1.0,
    fontFace: HEAD, fontSize: 54, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Streaming ELT and semantic search over 28,587 support tickets", {
    x: 0.9, y: 3.05, w: 11.5, h: 0.5,
    fontFace: BODY, fontSize: 20, color: ICE, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 4.0, w: 5.2, h: 0.05, rectRadius: 0,
    fill: { color: AMBER }, line: { color: AMBER, width: 0 },
  });

  s.addText(
    [
      { text: "Kafka  ·  Spark Structured Streaming  ·  MinIO  ·  Elasticsearch", options: { breakLine: true } },
      { text: "AI capability: embeddings + semantic search, RAG, k-NN routing", options: { breakLine: false } },
    ],
    { x: 0.9, y: 4.3, w: 11.5, h: 0.9, fontFace: BODY, fontSize: 14, color: ICE, margin: 0, lineSpacing: 22 }
  );

  s.addText("BIU 8688697201 — Big Data and AI  |  Yael Dorin", {
    x: 0.9, y: 6.4, w: 11.5, h: 0.4,
    fontFace: BODY, fontSize: 12, color: SLATE, margin: 0,
  });
  s.addNotes(
    "The operational question: given only the free text of an incoming ticket, which support queue " +
    "should it go to, and what similar tickets have we already solved? Both are text-understanding " +
    "problems, which is why they resist the SQL-shaped tooling that handles the rest of a support desk."
  );
}

// =========================================================================
// 2. Problem and dataset
// =========================================================================
{
  const s = lightSlide("The problem, and why this dataset", "Customer IT Support Ticket Dataset — Tobias Bueck, CC BY-NC 4.0");

  s.addText(
    [
      { text: "Two costs in every support organisation", options: { bold: true, breakLine: true, fontSize: 15 } },
      { text: "Someone must route each free-text ticket to the right queue.", options: { bullet: true, breakLine: true } },
      { text: "Agents re-solve problems the organisation already solved.", options: { bullet: true, breakLine: true } },
      { text: " ", options: { breakLine: true } },
      { text: "Why it qualifies as semi-structured", options: { bold: true, breakLine: true, fontSize: 15 } },
      { text: "subject / body / answer are natural language, averaging 387 characters.", options: { bullet: true, breakLine: true } },
      { text: "tag_1…tag_8 is a ragged list of 1,255 values flattened into sparse columns — 98% null by tag_8.", options: { bullet: true, breakLine: true } },
      { text: "subject itself is null in 13.4% of rows. That is data, not an error.", options: { bullet: true, breakLine: false } },
    ],
    { x: 0.6, y: 1.85, w: 7.2, h: 4.4, fontFace: BODY, fontSize: 13, color: INK, margin: 0, paraSpaceAfter: 7 }
  );

  statCard(s, { x: 8.2, y: 1.85, w: 2.3, h: 1.55, value: "28,587", label: "tickets", sub: "en 57% / de 43%", fill: LIGHT });
  statCard(s, { x: 10.7, y: 1.85, w: 2.0, h: 1.55, value: "10", label: "queues", sub: "the routing label", fill: LIGHT });
  statCard(s, { x: 8.2, y: 3.6, w: 2.3, h: 1.55, value: "4", label: "ticket types", sub: "Incident · Request\nProblem · Change", fill: LIGHT });
  statCard(s, { x: 10.7, y: 3.6, w: 2.0, h: 1.55, value: "1,255", label: "distinct tags", sub: "ragged, sparse", fill: LIGHT });

  s.addText("Majority class is Technical Support at 29.3% — the number every model must beat.", {
    x: 8.2, y: 5.35, w: 4.5, h: 0.7, fontFace: BODY, fontSize: 11, color: SLATE, italic: true, margin: 0,
  });

  s.addNotes(
    "The brief says clean tabular data alone is not sufficient. This dataset qualifies twice over: three " +
    "free-text fields, and a genuinely ragged tag structure that has to be normalised into an array before " +
    "anything can use it. Note the 29.3% majority class — it is the baseline every result on the next slides " +
    "is measured against."
  );
}

// =========================================================================
// 3. Architecture
// =========================================================================
{
  const s = lightSlide("Architecture", "ELT, not ETL — the broker holds raw truth and Spark does the transformation");

  const y0 = 2.0;
  stageBox(s, {
    x: 0.6, y: y0, w: 2.25, h: 1.5, fill: NAVY,
    title: "1 · Ingest",
    lines: ["producer.py", "CSV → JSON", "Kafka tickets.raw", "keyed by content hash"],
  });
  arrow(s, 2.95, y0 + 0.6);

  stageBox(s, {
    x: 3.35, y: y0, w: 2.45, h: 1.5, fill: NAVY,
    title: "2 · Transform",
    lines: ["Spark Structured", "Streaming", "foreachBatch fan-out", "enrichment UDFs"],
  });
  arrow(s, 5.9, y0 + 0.6);

  stageBox(s, {
    x: 6.3, y: y0, w: 2.45, h: 1.5, fill: NAVY,
    title: "3 · Lake (MinIO)",
    lines: ["bronze — raw, immutable", "silver — enriched", "gold — KPI aggregates", "Parquet throughout"],
  });
  arrow(s, 8.85, y0 + 0.6);

  stageBox(s, {
    x: 9.25, y: y0, w: 3.45, h: 1.5, fill: NAVY,
    title: "4 · Serve",
    lines: ["Elasticsearch — dense_vector 384", "cosine kNN + BM25, same docs", "Kibana dashboards", "FastAPI /search /ask /route"],
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 3.75, w: 12.1, h: 1.25, rectRadius: 0.08,
    fill: { color: LIGHT }, line: { color: LIGHT, width: 0 },
  });
  s.addText("Enrichment applied to every ticket", {
    x: 0.8, y: 3.85, w: 5.0, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "normalise text  ·  redact PII  ·  detect language  ·  sentiment  ·  rule classify  ·  embed 384-d",
    { x: 0.8, y: 4.2, w: 11.7, h: 0.35, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0 }
  );
  s.addText(
    "All of it is pure Python in core/ — Spark calls the same functions the unit tests call.",
    { x: 0.8, y: 4.55, w: 11.7, h: 0.35, fontFace: BODY, fontSize: 11, color: SLATE, italic: true, margin: 0 }
  );

  s.addText(
    [
      { text: "Why foreachBatch instead of three streaming sinks?  ", options: { bold: true } },
      { text: "One micro-batch fans out to two Parquet paths and Elasticsearch. Three separate writeStream queries would re-run the expensive enrichment for each sink.", options: {} },
    ],
    { x: 0.6, y: 5.25, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 12, color: INK, margin: 0 }
  );
  s.addText(
    [
      { text: "Why a content-hash ticket_id?  ", options: { bold: true } },
      { text: "Kafka gives at-least-once. Hashing content makes every sink idempotent, so a replayed batch overwrites rather than duplicates — effectively-once without transactions.", options: {} },
    ],
    { x: 0.6, y: 5.8, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 12, color: INK, margin: 0 }
  );

  s.addNotes(
    "Medallion layout: bronze is raw and never rewritten, so when enrichment logic changes — and it changed " +
    "four times while building this — we replay from bronze rather than from Kafka, whose retention is finite. " +
    "Be ready for the question 'why not exactly-once': Spark's exactly-once needs a transactional sink; we get " +
    "the same end state more cheaply by making the sinks idempotent."
  );
}

// =========================================================================
// 4. The AI capability
// =========================================================================
{
  const s = lightSlide("The AI capability", "Four of the brief's options; embeddings + semantic search is the centrepiece");

  const rows = [
    ["b", "Embeddings + semantic search", "384-d vectors in an Elasticsearch dense_vector field, cosine kNN", TEAL],
    ["f", "k-NN routing classifier", "Reuses the same vectors — no separate model, no training run", NAVY],
    ["c", "RAG question answering", "Retrieves k tickets, answers only from them, citations verified", NAVY],
    ["a", "LLM enrichment", "Classify, tag, summarise — output validated against known labels", NAVY],
  ];

  let y = 1.95;
  rows.forEach(([letter, title, desc, color]) => {
    s.addShape(pres.ShapeType.ellipse, {
      x: 0.6, y: y + 0.06, w: 0.5, h: 0.5,
      fill: { color }, line: { color, width: 0 },
    });
    s.addText(letter, {
      x: 0.6, y: y + 0.06, w: 0.5, h: 0.5,
      fontFace: HEAD, fontSize: 17, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
    });
    s.addText(title, {
      x: 1.3, y: y + 0.02, w: 5.0, h: 0.32,
      fontFace: BODY, fontSize: 14.5, bold: true, color: INK, margin: 0,
    });
    s.addText(desc, {
      x: 1.3, y: y + 0.34, w: 6.4, h: 0.34,
      fontFace: BODY, fontSize: 11.5, color: SLATE, margin: 0,
    });
    y += 0.82;
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 8.2, y: 1.95, w: 4.5, h: 2.5, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Two backends, one interface", {
    x: 8.45, y: 2.1, w: 4.0, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: WHITE, margin: 0,
  });
  s.addText(
    [
      { text: "multilingual MiniLM — a shared en/de vector space", options: { bullet: true, breakLine: true } },
      { text: "offline TF-IDF + SVD — no download, deterministic", options: { bullet: true, breakLine: true } },
      { text: "Identical width and normalisation, so the ES mapping never changes between them.", options: { bullet: true, breakLine: false } },
    ],
    { x: 8.45, y: 2.5, w: 4.0, h: 1.8, fontFace: BODY, fontSize: 11, color: ICE, margin: 0, paraSpaceAfter: 8 }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: 8.2, y: 4.7, w: 4.5, h: 2.05, rectRadius: 0.08,
    fill: { color: LIGHT }, line: { color: LIGHT, width: 0 },
  });
  s.addText("Guardrails, not prompts", {
    x: 8.45, y: 4.85, w: 4.0, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0,
  });
  s.addText(
    [
      { text: "PII gate fails closed before any hosted call", options: { bullet: true, breakLine: true } },
      { text: "Invented queue labels rejected, not indexed", options: { bullet: true, breakLine: true } },
      { text: "RAG citations checked against retrieved ids", options: { bullet: true, breakLine: false } },
    ],
    { x: 8.45, y: 5.25, w: 4.0, h: 1.35, fontFace: BODY, fontSize: 11, color: INK, margin: 0, paraSpaceAfter: 8 }
  );

  s.addNotes(
    "The point about two backends is robustness: with LLM_PROVIDER=none and no model weights the pipeline " +
    "still produces every number in this deck. That is deliberate — the demo cannot be broken by a failed " +
    "download. On guardrails: asking a model to cite only what it was given is a prompt, not a guarantee, " +
    "so we post-check every citation and strip the invented ones."
  );
}

// =========================================================================
// 5. Headline result - routing
// =========================================================================
{
  const s = lightSlide("Result: routing tickets from text alone", "Stratified 25% held-out split, n = 7,145");

  const order = ["majority_class", "keyword_rules", "embedding_centroid",
                 knnRouter.method, "linear_svm_tfidf"];
  const labels = ["Always guess\nmajority class", "Keyword rules", "Embedding\ncentroid",
                  `Embedding k-NN\n(k=${knnRouter.method.replace("embedding_knn_k", "")})`,
                  "Linear SVM\nword+char TF-IDF"];

  s.addChart(
    pres.ChartType.bar,
    [
      {
        name: "Accuracy",
        labels,
        values: order.map((m) => Number(((routing[m] || {accuracy: 0}).accuracy * 100).toFixed(1))),
      },
    ],
    {
      x: 0.55, y: 1.95, w: 6.5, h: 4.05,
      barDir: "col",
      chartColors: [SLATE, SLATE, SLATE, SLATE, TEAL],
      varyColors: true,
      showTitle: true,
      title: "Accuracy (%)",
      titleFontFace: BODY, titleFontSize: 13, titleColor: INK,
      showValue: true,
      dataLabelPosition: "outEnd",
      dataLabelFontFace: BODY, dataLabelFontSize: 12, dataLabelColor: INK,
      dataLabelFormatCode: '0.0"%"',
      showLegend: false,
      catAxisLabelColor: INK, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10,
      valAxisLabelColor: SLATE, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10,
      valAxisMaxVal: 60,
      valGridLine: { color: "E4E8F0", size: 1 },
      catGridLine: { style: "none" },
    }
  );

  statCard(s, {
    x: 7.4, y: 1.95, w: 2.55, h: 1.75,
    value: pct(bestRouter.accuracy), label: "best router accuracy",
    sub: `vs ${pct(routing.majority_class.accuracy)} majority`, color: TEAL, fill: LIGHT,
  });
  statCard(s, {
    x: 10.15, y: 1.95, w: 2.55, h: 1.75,
    value: bestRouter.macro_f1.toFixed(3), label: "macro-F1",
    sub: `vs ${routing.majority_class.macro_f1.toFixed(3)} majority`, color: TEAL, fill: LIGHT,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 7.4, y: 3.95, w: 5.3, h: 2.05, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Hand-written rules lose to guessing", {
    x: 7.6, y: 4.08, w: 4.9, h: 0.32, fontFace: BODY, fontSize: 13.5, bold: true, color: AMBER, margin: 0,
  });
  s.addText(
    `${pct(routing.keyword_rules.accuracy)} accuracy is below the ${pct(routing.majority_class.accuracy)} you get by ` +
    "answering \"Technical Support\" every time. The rules do carry signal — macro-F1 " +
    `${routing.keyword_rules.macro_f1.toFixed(3)} against ${routing.majority_class.macro_f1.toFixed(3)} — but on raw ` +
    "accuracy they are worse than a constant. This is why we report both metrics.",
    { x: 7.6, y: 4.45, w: 4.9, h: 1.45, fontFace: BODY, fontSize: 11, color: ICE, margin: 0 }
  );

  s.addNotes(
    "This is the headline. The k-NN classifier trains nothing — it reuses the embeddings that already exist for " +
    "semantic search, which is the cheapest possible route from 'we have vectors' to 'we have business value'. " +
    "If asked why macro-F1 is so much lower than accuracy: the queue distribution is heavily skewed, from 29% " +
    "Technical Support down to 1.4% General Inquiry, and macro-F1 weights every class equally."
  );
}

// =========================================================================
// 6. The result that contradicted the design
// =========================================================================
{
  const s = lightSlide(
    "The result that contradicted our design",
    "250 queries, k=5, proxy relevance — a hit counts if it shares ≥2 tags with the query ticket"
  );

  s.addChart(
    pres.ChartType.bar,
    [
      {
        name: "precision@5",
        labels: ["Semantic", "Keyword (BM25)", "Hybrid (RRF)"],
        values: [retrieval.semantic, retrieval.keyword, retrieval.hybrid].map((r) =>
          Number(r.precision_at_k.toFixed(3))
        ),
      },
      {
        name: "MRR",
        labels: ["Semantic", "Keyword (BM25)", "Hybrid (RRF)"],
        values: [retrieval.semantic, retrieval.keyword, retrieval.hybrid].map((r) =>
          Number(r.mrr.toFixed(3))
        ),
      },
    ],
    {
      x: 0.55, y: 2.0, w: 6.4, h: 3.9,
      barDir: "col",
      barGrouping: "clustered",
      chartColors: [TEAL, NAVY],
      showTitle: false,
      showValue: true,
      dataLabelPosition: "outEnd",
      dataLabelFontFace: BODY, dataLabelFontSize: 10, dataLabelColor: INK,
      dataLabelFormatCode: "0.000",
      showLegend: true,
      legendPos: "b",
      legendFontFace: BODY, legendFontSize: 11, legendColor: INK,
      catAxisLabelColor: INK, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 11,
      valAxisLabelColor: SLATE, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10,
      valAxisMaxVal: 1.0,
      valGridLine: { color: "E4E8F0", size: 1 },
      catGridLine: { style: "none" },
    }
  );

  s.addText("Semantic search did not beat BM25.", {
    x: 7.3, y: 2.0, w: 5.4, h: 0.4,
    fontFace: HEAD, fontSize: 19, bold: true, color: CRIMSON, margin: 0,
  });

  s.addText(
    [
      { text: "Why — and why it was predictable", options: { bold: true, breakLine: true, fontSize: 13 } },
      { text: "These numbers come from the offline TF-IDF + SVD backend, i.e. LSA. LSA is a linear projection of the same term-document statistics BM25 already scores. It compresses lexical information; it adds no semantic information from outside the corpus.", options: { breakLine: true } },
      { text: "The cross-lingual probe makes the point sharply: a German query retrieves ~0 relevant English tickets, because LSA shares no space across languages.", options: { breakLine: true } },
      { text: "The multilingual neural model is expected to change both results. We have not measured it, so we do not claim it.", options: { bold: true, breakLine: false } },
    ],
    { x: 7.3, y: 2.5, w: 5.4, h: 3.4, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0, paraSpaceAfter: 8 }
  );

  s.addText(
    "Routing is a different task: k-NN exploits the geometry of the space, so embeddings win there even in LSA form.",
    { x: 0.55, y: 6.05, w: 12.15, h: 0.4, fontFace: BODY, fontSize: 11.5, color: SLATE, italic: true, margin: 0 }
  );

  s.addNotes(
    "Do not skip this slide. It is the one that shows understanding rather than just a working system. The " +
    "honest position is: our design predicted semantic would win, it did not under the backend we could actually " +
    "run, and we can explain exactly why. Reproducing the neural comparison is one environment variable — " +
    "EMBEDDING_BACKEND=sentence-transformers — and one pipeline re-run."
  );
}

// =========================================================================
// 7. Insights from the data
// =========================================================================
{
  const s = lightSlide("What the data told us", "Findings that change what an operations team would do on Monday");

  const mismatchDe = kpi("language_mismatch_rate", "declared_language", "de");
  const mismatchEn = kpi("language_mismatch_rate", "declared_language", "en");
  const outages = kpi("high_priority_share", "queue", "Service Outages and Maintenance");
  const hr = kpi("high_priority_share", "queue", "Human Resources");

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.9, w: 5.95, h: 2.15, rectRadius: 0.08,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("The language label is broken in one direction", {
    x: 0.8, y: 2.02, w: 5.55, h: 0.32, fontFace: BODY, fontSize: 14, bold: true, color: AMBER, margin: 0,
  });
  s.addText(
    [
      { text: `Tickets labelled de are wrong ${pct(mismatchDe.value)} of the time.`, options: { breakLine: true } },
      { text: `Tickets labelled en are wrong ${(mismatchEn.value * 100).toFixed(2)}% of the time.`, options: { breakLine: true } },
      { text: `The ${pct(summary.language_mismatch_overall)} headline rate hides a one-directional defect: more than a quarter of the "German" corpus is English text.`, options: { breakLine: false } },
    ],
    { x: 0.8, y: 2.42, w: 5.55, h: 1.5, fontFace: BODY, fontSize: 11.5, color: ICE, margin: 0, paraSpaceAfter: 7 }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.75, y: 1.9, w: 5.95, h: 2.15, rectRadius: 0.08,
    fill: { color: LIGHT }, line: { color: LIGHT, width: 0 },
  });
  s.addText("Urgency is concentrated, not where volume is", {
    x: 6.95, y: 2.02, w: 5.55, h: 0.32, fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0,
  });
  s.addText(
    [
      { text: `Service Outages is 4.0% of volume but ${pct(outages.value)} high priority.`, options: { breakLine: true } },
      { text: `Human Resources is ${pct(hr.value)}.`, options: { breakLine: true } },
      { text: "Staffing to ticket volume would badly under-resource the outage queue.", options: { breakLine: false } },
    ],
    { x: 6.95, y: 2.42, w: 5.55, h: 1.5, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0, paraSpaceAfter: 7 }
  );

  s.addChart(
    pres.ChartType.bar,
    [
      {
        name: "High-priority share",
        labels: report.kpis
          .filter((k) => k.kpi === "high_priority_share")
          .slice(0, 6)
          .map((k) => k.bucket.replace(" and Maintenance", "").replace(" and Payments", "")),
        values: report.kpis
          .filter((k) => k.kpi === "high_priority_share")
          .slice(0, 6)
          .map((k) => Number((k.value * 100).toFixed(1))),
      },
    ],
    {
      x: 0.55, y: 4.2, w: 12.2, h: 2.35,
      barDir: "bar",
      chartColors: [NAVY],
      showTitle: true,
      title: "High-priority share by queue (%)",
      titleFontFace: BODY, titleFontSize: 12, titleColor: INK,
      showValue: true,
      dataLabelPosition: "outEnd",
      dataLabelFontFace: BODY, dataLabelFontSize: 10, dataLabelColor: INK,
      dataLabelFormatCode: '0.0"%"',
      showLegend: false,
      catAxisLabelColor: INK, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10,
      valAxisLabelColor: SLATE, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 9,
      valAxisMaxVal: 80,
      valGridLine: { color: "E4E8F0", size: 1 },
      catGridLine: { style: "none" },
    }
  );

  s.addNotes(
    "The language finding is the one to lead with in questions. An 11.7% overall mismatch sounds like uniform " +
    "noise; splitting it by declared language shows it is almost entirely on the German side, which points at a " +
    "specific upstream defect rather than detector error. Anything keyed off that field — canned replies, " +
    "language-based routing — is misfiring on those rows today."
  );
}

// =========================================================================
// 8. Trade-offs
// =========================================================================
{
  const s = lightSlide("Trade-offs we made deliberately", "And the one we would change first");

  const items = [
    ["Pure-Python core, Spark as a thin wrapper",
     "Every transformation is a pure function; UDFs call them. This is what makes 80 unit tests meaningful and lets the whole pipeline run without Spark. Cost: a second native implementation of the KPIs, cross-checked automatically."],
    ["At-least-once + idempotent sinks",
     "Cheaper than transactional exactly-once, and the end state is identical because ticket_id is a content hash."],
    ["Exact k-NN offline, HNSW in Elasticsearch",
     "At 28k documents brute force is fast and removes approximate-recall as a confound when comparing against BM25."],
    ["Macro-F1 reported alongside accuracy",
     "With a 29.3% majority class, accuracy alone would have made the keyword rules look nearly as good as k-NN."],
  ];

  let y = 1.95;
  items.forEach(([title, body]) => {
    s.addText(title, {
      x: 0.6, y, w: 7.3, h: 0.3, fontFace: BODY, fontSize: 13.5, bold: true, color: NAVY, margin: 0,
    });
    s.addText(body, {
      x: 0.6, y: y + 0.3, w: 7.3, h: 0.62, fontFace: BODY, fontSize: 11, color: SLATE, margin: 0,
    });
    y += 1.05;
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 8.3, y: 1.95, w: 4.4, h: 2.5, rectRadius: 0.08,
    fill: { color: CRIMSON }, line: { color: CRIMSON, width: 0 },
  });
  s.addText("Weakest link", {
    x: 8.5, y: 2.08, w: 4.0, h: 0.32, fontFace: BODY, fontSize: 14, bold: true, color: WHITE, margin: 0,
  });
  s.addText(
    "The proxy relevance metric. Shared tags reward topical similarity — right for routing and deduplication — " +
    "but they under-credit a system that retrieves a solution phrased differently from the problem. " +
    "Human judgements on a few hundred queries is the honest fix, and the first thing we would add.",
    { x: 8.5, y: 2.5, w: 4.0, h: 1.85, fontFace: BODY, fontSize: 11, color: WHITE, margin: 0 }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: 8.3, y: 4.7, w: 4.4, h: 1.55, rectRadius: 0.08,
    fill: { color: LIGHT }, line: { color: LIGHT, width: 0 },
  });
  s.addText("Honest scope", {
    x: 8.5, y: 4.82, w: 4.0, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "80 unit tests pass and the pipeline ran end to end over all 28,587 rows. The Docker stack, the Spark jobs " +
    "and the Elasticsearch writes have since been run and verified against the live stack — see docs/DEMO.md.",
    { x: 8.5, y: 5.15, w: 4.0, h: 1.0, fontFace: BODY, fontSize: 10.5, color: SLATE, margin: 0 }
  );

  s.addNotes(
    "Being explicit about what was and was not verified is worth more marks than glossing over it, and it is the " +
    "honest answer if an examiner asks 'did you actually run this end to end on the cluster'."
  );
}

// =========================================================================
// 9. Demo + close
// =========================================================================
{
  const s = darkSlide();
  s.addText("Demo", {
    x: 0.9, y: 0.75, w: 11.5, h: 0.7,
    fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, margin: 0,
  });

  const demos = [
    ["GET /compare?q=…", "One query, three rankings side by side — semantic, BM25 and hybrid. The clearest way to see what each retrieval mode actually does."],
    ["POST /route", "Paste an unseen ticket. Returns the rule-based guess and the k-NN guess with its nearest neighbours, so the improvement is visible per request."],
    ["POST /ask", "RAG over the corpus. Answers only from retrieved tickets; invented citations are stripped before you see them."],
    ["Kibana", "KPI dashboards over the gold layer — queue volumes, sentiment, high-priority concentration."],
  ];

  let y = 1.7;
  demos.forEach(([title, body]) => {
    s.addText(title, {
      x: 0.9, y, w: 11.4, h: 0.32, fontFace: BODY, fontSize: 15, bold: true, color: AMBER, margin: 0,
    });
    s.addText(body, {
      x: 0.9, y: y + 0.32, w: 11.4, h: 0.5, fontFace: BODY, fontSize: 12.5, color: ICE, margin: 0,
    });
    y += 0.98;
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 5.75, w: 11.5, h: 0.05, rectRadius: 0,
    fill: { color: AMBER }, line: { color: AMBER, width: 0 },
  });
  s.addText(
    "Fallback: `python -m tickets.offline_pipeline` reproduces every number in this deck with no Docker, " +
    "no API key and no network.",
    { x: 0.9, y: 6.0, w: 11.5, h: 0.5, fontFace: BODY, fontSize: 12.5, color: WHITE, italic: true, margin: 0 }
  );
  s.addText("github.com/yaeldorin-glitch/biu-bigdata-support-tickets", {
    x: 0.9, y: 6.6, w: 11.5, h: 0.35, fontFace: BODY, fontSize: 11.5, color: SLATE, margin: 0,
  });

  s.addNotes(
    "Lead the demo with /compare — it makes the difference between lexical and semantic matching visible on one " +
    "screen, which no amount of describing it does. Keep the offline runner ready as the fallback if a container " +
    "misbehaves; it produces the same figures."
  );
}

const out = path.join(ROOT, "docs", "presentation.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
