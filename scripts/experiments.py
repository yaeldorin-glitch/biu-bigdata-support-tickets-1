"""Routing accuracy sweep: what actually moves the number?

Runs several candidate improvements over the identical stratified split used by
tickets.ai.evaluate.evaluate_routing (seed 42, 25% held out), so every result is
directly comparable to the figures in docs/RESULTS.md.

    python scripts/experiments.py --full

Anything labelled LEAKAGE is measured for information only -- it uses fields that
do not exist yet when a *new* ticket arrives, so it cannot be deployed. It is
included to show where the ceiling is.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tickets.ai.evaluate import accuracy, macro_f1  # noqa: E402
from tickets.ai.router import CentroidRouter, KnnRouter  # noqa: E402
from tickets.core.classify import classify_queue  # noqa: E402
from tickets.core.embeddings import OfflineTfidfSvdBackend  # noqa: E402
from tickets.core.enrich import attach_embeddings, enrich_ticket  # noqa: E402
from tickets.offline_pipeline import load_tickets  # noqa: E402

RESULTS: list[dict] = []


def record(name, true, pred, note=""):
    row = {
        "method": name,
        "accuracy": round(accuracy(true, pred), 4),
        "macro_f1": round(macro_f1(true, pred), 4),
        "note": note,
    }
    RESULTS.append(row)
    print(f"  {name:<44} acc={row['accuracy']:.1%}  macroF1={row['macro_f1']:.3f}  {note}")
    return row


def stratified_split(tickets, test_fraction=0.25, seed=42):
    """Identical logic to evaluate_routing, so results line up with RESULTS.md."""
    rng = random.Random(seed)
    by_queue = defaultdict(list)
    for t in tickets:
        by_queue[t.queue].append(t)

    train, test = [], []
    for queue in by_queue:
        group = by_queue[queue][:]
        rng.shuffle(group)
        cut = max(1, int(len(group) * test_fraction))
        test.extend(group[:cut])
        train.extend(group[cut:])
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "tickets.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "experiments.json")
    args = parser.parse_args()

    limit = None if args.full else args.limit

    print("loading + enriching...")
    t0 = time.time()
    tickets, _ = load_tickets(args.csv, limit)
    for t in tickets:
        enrich_ticket(t)
    tickets = [t for t in tickets if t.queue]
    print(f"  {len(tickets):,} labelled tickets in {time.time()-t0:.0f}s")

    train, test = stratified_split(tickets)
    y_train = [t.queue for t in train]
    y_test = [t.queue for t in test]
    print(f"  train={len(train):,}  test={len(test):,}\n")

    # Text variants -------------------------------------------------------
    text_train = [t.text_redacted or t.text for t in train]
    text_test = [t.text_redacted or t.text for t in test]

    # =====================================================================
    print("BASELINES")
    majority = Counter(y_train).most_common(1)[0][0]
    record("majority_class", y_test, [majority] * len(y_test))
    record("keyword_rules", y_test, [classify_queue(x).label for x in text_test])

    # =====================================================================
    print("\nEMBEDDINGS (offline TF-IDF+SVD, 384-d) — vary k")
    t0 = time.time()
    backend = OfflineTfidfSvdBackend(dim=384).fit(text_train)   # fit on TRAIN ONLY
    attach_embeddings(train, backend)
    attach_embeddings(test, backend)
    X_train = np.asarray([t.embedding for t in train], dtype=np.float32)
    X_test = np.asarray([t.embedding for t in test], dtype=np.float32)
    print(f"  (embedding fit+encode {time.time()-t0:.0f}s)")

    record("embedding_centroid", y_test, [p.label for p in CentroidRouter().fit(X_train, y_train).predict(X_test)])
    for k in (5, 15, 30, 50, 100):
        preds = [p.label for p in KnnRouter(k=k).fit(X_train, y_train).predict(X_test)]
        record(f"embedding_knn_k{k}", y_test, preds)

    # =====================================================================
    print("\nSUPERVISED LINEAR MODELS on sparse TF-IDF (no SVD)")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.6,
                          sublinear_tf=True, max_features=200_000)
    Xs_train = vec.fit_transform(text_train)
    Xs_test = vec.transform(text_test)
    print(f"  sparse features: {Xs_train.shape[1]:,}")

    for name, model in [
        ("linear_svc", LinearSVC(C=1.0, max_iter=3000)),
        ("linear_svc_balanced", LinearSVC(C=1.0, class_weight="balanced", max_iter=3000)),
        ("logreg_balanced", LogisticRegression(max_iter=1500, class_weight="balanced", n_jobs=-1)),
    ]:
        t0 = time.time()
        model.fit(Xs_train, y_train)
        record(f"tfidf_{name}", y_test, list(model.predict(Xs_test)), f"({time.time()-t0:.0f}s train)")

    # word+char union, which helps German compounds
    from sklearn.pipeline import FeatureUnion
    union = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.6, sublinear_tf=True, max_features=150_000)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150_000)),
    ])
    Xu_train = union.fit_transform(text_train)
    Xu_test = union.transform(text_test)
    model = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000).fit(Xu_train, y_train)
    record("tfidf_word+char_svc_balanced", y_test, list(model.predict(Xu_test)),
           f"({Xu_train.shape[1]:,} features)")

    # =====================================================================
    print("\nCEILING PROBES — not deployable, measured for information only")

    # The agent's reply exists only AFTER the ticket is handled.
    text_ans_train = [f"{t.text_redacted} {t.answer}" for t in train]
    text_ans_test = [f"{t.text_redacted} {t.answer}" for t in test]
    v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.6, sublinear_tf=True, max_features=200_000)
    m = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000).fit(v2.fit_transform(text_ans_train), y_train)
    record("LEAKAGE +answer_field", y_test, list(m.predict(v2.transform(text_ans_test))),
           "answer does not exist for a new ticket")

    # Tags are assigned by the same labelling process as the queue.
    tag_train = [f"{t.text_redacted} {' '.join(t.tags)}" for t in train]
    tag_test = [f"{t.text_redacted} {' '.join(t.tags)}" for t in test]
    v3 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.6, sublinear_tf=True, max_features=200_000)
    m = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000).fit(v3.fit_transform(tag_train), y_train)
    record("LEAKAGE +tags", y_test, list(m.predict(v3.transform(tag_test))),
           "tags co-generated with the queue label")

    # =====================================================================
    print("\nWHERE THE ERRORS ARE (best deployable model)")
    best = max((r for r in RESULTS if not r["method"].startswith("LEAKAGE")),
               key=lambda r: r["macro_f1"])
    print(f"  best deployable: {best['method']}")

    model = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000).fit(Xu_train, y_train)
    preds = list(model.predict(Xu_test))

    confusion = Counter()
    for true_label, pred_label in zip(y_test, preds):
        if true_label != pred_label:
            confusion[(true_label, pred_label)] += 1
    print("\n  top confusions (true -> predicted):")
    for (a, b), n in confusion.most_common(10):
        print(f"    {n:>4}  {a}  ->  {b}")

    per_class = {}
    for label in sorted(set(y_test)):
        tp = sum(1 for t, p in zip(y_test, preds) if t == label and p == label)
        support = sum(1 for t in y_test if t == label)
        per_class[label] = {"recall": round(tp / support, 3), "support": support}
    print("\n  per-class recall:")
    for label, d in sorted(per_class.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"    {d['recall']:.3f}  ({d['support']:>5,})  {label}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "results": RESULTS,
        "top_confusions": [{"true": a, "pred": b, "n": n} for (a, b), n in confusion.most_common(15)],
        "per_class_recall": per_class,
        "n_train": len(train), "n_test": len(test),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
