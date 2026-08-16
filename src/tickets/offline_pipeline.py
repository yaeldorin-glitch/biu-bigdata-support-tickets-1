"""Offline end-to-end runner -- the same logic, no infrastructure.

Runs parse -> validate -> enrich -> embed -> index -> evaluate -> KPIs against
the CSV directly, using the identical functions the Spark job calls. It exists
for three reasons:

1. **The demo always works.** If Docker will not start on the presentation
   machine, this still produces every number in the report.
2. **It is the correctness oracle.** The Spark job's aggregates are verified
   against the same reference implementation this runner uses.
3. **It generates the report.** The figures in ``docs/`` and in the slides come
   from ``output/report.json``, produced here, so they are reproducible rather
   than hand-copied.

Usage::

    python -m tickets.offline_pipeline --limit 5000
    python -m tickets.offline_pipeline --full --output output/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import OUTPUT_DIR, RAW_CSV, SAMPLE_CSV, get_settings
from .core import kpis as kpi_module
from .core.embeddings import OfflineTfidfSvdBackend, build_backend, set_backend
from .core.enrich import attach_embeddings, enrich_ticket
from .core.schema import Ticket, is_valid, parse_row

logger = logging.getLogger(__name__)


def load_tickets(csv_path: Path, limit: int | None = None) -> tuple[list[Ticket], dict[str, int]]:
    """Read and validate the CSV. Returns tickets plus a rejection tally."""
    import csv
    import sys

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    tickets: list[Ticket] = []
    rejected: dict[str, int] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if limit is not None and len(tickets) >= limit:
                break
            ticket = parse_row(row)
            ok, reason = is_valid(ticket)
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            tickets.append(ticket)

    return tickets, rejected


def run(
    csv_path: Path,
    output_dir: Path,
    limit: int | None = None,
    eval_queries: int = 200,
    knn_k: int = 15,
    skip_retrieval_eval: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # --- extract --------------------------------------------------------
    t0 = time.time()
    tickets, rejected = load_tickets(csv_path, limit)
    timings["load"] = time.time() - t0
    if not tickets:
        raise SystemExit(f"no valid tickets found in {csv_path}")
    logger.info("loaded %s tickets in %.1fs (rejected: %s)", len(tickets), timings["load"], rejected or "none")

    # --- transform ------------------------------------------------------
    t0 = time.time()
    for ticket in tickets:
        enrich_ticket(ticket)
    timings["enrich"] = time.time() - t0
    logger.info("enriched in %.1fs", timings["enrich"])

    # --- embed ----------------------------------------------------------
    t0 = time.time()
    corpus = [t.text_redacted or t.text for t in tickets]
    backend = build_backend(settings.ai, corpus=corpus)
    set_backend(backend)
    attach_embeddings(tickets, backend)
    timings["embed"] = time.time() - t0
    logger.info("embedded with %s (dim=%s) in %.1fs", backend.name, backend.dim, timings["embed"])

    # Persist the fitted offline model so the Spark job can reuse the *same*
    # vector space instead of refitting and producing incompatible embeddings.
    if isinstance(backend, OfflineTfidfSvdBackend):
        model_path = output_dir / "offline_embedding_model.joblib"
        backend.save(model_path)
        logger.info("saved fitted offline embedding model to %s", model_path)

    # --- index ----------------------------------------------------------
    from .ai.search import InMemoryIndex

    t0 = time.time()
    index = InMemoryIndex(tickets, backend)
    timings["index"] = time.time() - t0

    # --- evaluate -------------------------------------------------------
    from .ai.evaluate import cross_lingual_probe, evaluate_retrieval, evaluate_routing

    t0 = time.time()
    routing = [r.to_dict() for r in evaluate_routing(tickets, knn_k=knn_k)]
    timings["eval_routing"] = time.time() - t0

    retrieval: list[dict[str, Any]] = []
    if not skip_retrieval_eval:
        t0 = time.time()
        retrieval = [r.to_dict() for r in evaluate_retrieval(tickets, index, n_queries=eval_queries)]
        timings["eval_retrieval"] = time.time() - t0

    probes = cross_lingual_probe(
        index,
        [
            ("Rechnung falsch berechnet", "de"),
            ("Server ist ausgefallen", "de"),
            ("cannot reset my password", "en"),
            ("data breach security incident", "en"),
        ],
    )

    # --- KPIs -----------------------------------------------------------
    t0 = time.time()
    all_kpis = kpi_module.compute_all(tickets)
    summary = kpi_module.summarize(tickets)
    timings["kpis"] = time.time() - t0

    # --- optional LLM narrative (clearly separated from computed values) -
    from .ai.rag import narrate_insights

    narrative = narrate_insights(summary)

    report = {
        "dataset": {
            "path": str(csv_path),
            "tickets_loaded": len(tickets),
            "rejected": rejected,
        },
        "embedding_backend": {"name": backend.name, "dim": backend.dim},
        "summary": summary,
        "routing_evaluation": routing,
        "retrieval_evaluation": retrieval,
        "cross_lingual_probe": probes,
        "kpis": all_kpis,
        "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        "generated_narrative": narrative,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", report_path)

    kpi_path = output_dir / "kpis.json"
    kpi_path.write_text(json.dumps(all_kpis, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(report)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\n" + "=" * 68)
    print("PIPELINE SUMMARY")
    print("=" * 68)
    print(f"tickets              : {summary['total_tickets']:,}")
    print(f"embedding backend    : {report['embedding_backend']['name']} "
          f"(dim={report['embedding_backend']['dim']})")
    print(f"negative sentiment   : {summary['negative_share']:.1%}")
    print(f"language mismatch    : {summary['language_mismatch_overall']:.1%}")
    print(f"tickets with PII     : {summary['tickets_with_pii']:,}")

    print("\nROUTING (predicting `queue` from free text)")
    print("-" * 68)
    print(f"{'method':<26} {'accuracy':>10} {'macro-F1':>10} {'n_test':>8}")
    for row in report["routing_evaluation"]:
        print(f"{row['method']:<26} {row['accuracy']:>9.1%} {row['macro_f1']:>10.3f} {row['n_test']:>8,}")

    if report["retrieval_evaluation"]:
        print("\nRETRIEVAL (proxy relevance; comparative, not absolute)")
        print("-" * 68)
        print(f"{'method':<26} {'queue@k':>10} {'P@k':>10} {'MRR':>8}")
        for row in report["retrieval_evaluation"]:
            print(f"{row['method']:<26} {row['queue_purity_at_k']:>10.3f} "
                  f"{row['precision_at_k']:>10.3f} {row['mrr']:>8.3f}")

    print("\nCROSS-LINGUAL RETRIEVAL (hits in the other language, k=5)")
    print("-" * 68)
    for probe in report["cross_lingual_probe"]:
        print(f"  {probe['query'][:38]:<40} semantic={probe['semantic_cross_language_hits']}  "
              f"keyword={probe['keyword_cross_language_hits']}")
    print("=" * 68 + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the full pipeline offline, without Docker.")
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5000,
                        help="number of tickets to process (default 5000; use --full for all)")
    parser.add_argument("--full", action="store_true", help="process the entire dataset")
    parser.add_argument("--eval-queries", type=int, default=200)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--skip-retrieval-eval", action="store_true",
                        help="skip the retrieval benchmark (it is the slowest step)")
    args = parser.parse_args(argv)

    # A fresh clone has no data/raw/tickets.csv (the full export is CC BY-NC and
    # not committed), so fall back to the committed sample rather than failing.
    # This is what makes `make offline` work immediately after cloning.
    if not args.csv.exists():
        if args.csv == RAW_CSV and SAMPLE_CSV.exists():
            logger.warning(
                "%s not found; falling back to the committed sample at %s. "
                "See data/README.md to obtain the full 28,587-row dataset.",
                args.csv, SAMPLE_CSV,
            )
            args.csv = SAMPLE_CSV
        else:
            parser.error(f"CSV not found at {args.csv}. See data/README.md for how to obtain it.")

    run(
        csv_path=args.csv,
        output_dir=args.output,
        limit=None if args.full else args.limit,
        eval_queries=args.eval_queries,
        knn_k=args.knn_k,
        skip_retrieval_eval=args.skip_retrieval_eval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
