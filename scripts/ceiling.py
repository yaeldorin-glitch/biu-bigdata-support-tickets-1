"""How high can routing accuracy actually go?

`experiments.py` asked "which model is best". This asks the prior question:
**is the remaining error the model's fault, or the labels'?**

Four measurements:

1. **Label disagreement on near-duplicate tickets.** If two tickets with nearly
   identical text carry different `queue` labels, no model can get both right.
   This is a direct empirical estimate of the Bayes error -- the floor.
2. **Top-1 vs top-2 vs top-3 accuracy.** If the product is "suggest a queue to a
   human agent" rather than "decide unattended", top-3 is the honest metric and
   it is far higher.
3. **Accuracy vs coverage.** Route automatically only above a confidence
   threshold, send the rest to a human. Trades volume for precision.
4. **Merged labels.** Collapse the four overlapping generic support queues into
   one and re-measure. Shows how much of the error is that specific ambiguity.

    python scripts/ceiling.py --full
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
from tickets.core.embeddings import OfflineTfidfSvdBackend  # noqa: E402
from tickets.core.enrich import attach_embeddings, enrich_ticket  # noqa: E402
from tickets.offline_pipeline import load_tickets  # noqa: E402

# The four queues the confusion matrix shows are mutually entangled.
GENERIC_SUPPORT = {
    "Technical Support",
    "Product Support",
    "IT Support",
    "Customer Service",
}


def stratified_split(tickets, test_fraction=0.25, seed=42):
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
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "ceiling.json")
    args = parser.parse_args()

    out: dict = {}

    print("loading + enriching...")
    tickets, _ = load_tickets(args.csv, None if args.full else args.limit)
    for t in tickets:
        enrich_ticket(t)
    tickets = [t for t in tickets if t.queue]
    print(f"  {len(tickets):,} labelled tickets\n")

    train, test = stratified_split(tickets)
    y_train = [t.queue for t in train]
    y_test = [t.queue for t in test]
    text_train = [t.text_redacted or t.text for t in train]
    text_test = [t.text_redacted or t.text for t in test]

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    def build():
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.6,
                                     sublinear_tf=True, max_features=150_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     sublinear_tf=True, max_features=150_000)),
        ])
        return Pipeline([
            ("features", features),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=3000), cv=3)),
        ])

    # =====================================================================
    print("=" * 66)
    print("1. LABEL DISAGREEMENT ON NEAR-DUPLICATE TICKETS  (the Bayes floor)")
    print("=" * 66)

    t0 = time.time()
    backend = OfflineTfidfSvdBackend(dim=256).fit([t.text_redacted or t.text for t in tickets])
    attach_embeddings(tickets, backend)
    X = np.asarray([t.embedding for t in tickets], dtype=np.float32)
    labels = np.asarray([t.queue for t in tickets])
    print(f"  (embedded {len(tickets):,} in {time.time()-t0:.0f}s)")

    # For a sample of tickets, find the single nearest OTHER ticket and check
    # whether its label agrees. Chunked so the similarity matrix stays in RAM.
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(tickets), size=min(6000, len(tickets)), replace=False)

    agree_by_band: dict[str, list[int]] = defaultdict(list)
    CHUNK = 500
    for start in range(0, len(sample_idx), CHUNK):
        idx = sample_idx[start:start + CHUNK]
        sims = X[idx] @ X.T
        for row, i in zip(sims, idx):
            row[i] = -np.inf                      # exclude self
            j = int(np.argmax(row))
            similarity = float(row[j])
            band = (
                ">=0.95" if similarity >= 0.95 else
                "0.90-0.95" if similarity >= 0.90 else
                "0.80-0.90" if similarity >= 0.80 else
                "<0.80"
            )
            agree_by_band[band].append(1 if labels[j] == labels[i] else 0)

    print(f"\n  {'similarity of nearest neighbour':<34} {'pairs':>8} {'same queue':>12}")
    dup = {}
    for band in (">=0.95", "0.90-0.95", "0.80-0.90", "<0.80"):
        flags = agree_by_band.get(band, [])
        if not flags:
            continue
        rate = sum(flags) / len(flags)
        dup[band] = {"pairs": len(flags), "same_queue_rate": round(rate, 4)}
        print(f"  {band:<34} {len(flags):>8,} {rate:>11.1%}")
    out["nearest_neighbour_label_agreement"] = dup
    print("\n  Read this as: even for tickets that are near-textual-twins, the")
    print("  labels disagree often. That share is error no model can remove.")

    # =====================================================================
    print("\n" + "=" * 66)
    print("2. TOP-1 vs TOP-2 vs TOP-3 ACCURACY")
    print("=" * 66)

    t0 = time.time()
    model = build().fit(text_train, y_train)
    probabilities = model.predict_proba(text_test)
    classes = list(model.named_steps["clf"].classes_)
    print(f"  (trained in {time.time()-t0:.0f}s)")

    order = np.argsort(-probabilities, axis=1)
    topk = {}
    for k in (1, 2, 3):
        hits = sum(
            1 for row, true in zip(order, y_test)
            if true in {classes[j] for j in row[:k]}
        )
        topk[f"top{k}"] = round(hits / len(y_test), 4)
        print(f"  top-{k} accuracy: {hits/len(y_test):.1%}")
    out["topk_accuracy"] = topk
    print("\n  If the product suggests a queue to an agent rather than deciding")
    print("  unattended, top-3 is the metric that matters.")

    # =====================================================================
    print("\n" + "=" * 66)
    print("3. ACCURACY vs COVERAGE  (abstain when unsure)")
    print("=" * 66)

    confidence = probabilities.max(axis=1)
    predictions = [classes[row[0]] for row in order]
    correct = np.array([p == t for p, t in zip(predictions, y_test)])

    print(f"\n  {'threshold':>10} {'coverage':>10} {'accuracy on covered':>22}")
    curve = []
    for threshold in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        mask = confidence >= threshold
        if mask.sum() == 0:
            continue
        coverage = float(mask.mean())
        acc = float(correct[mask].mean())
        curve.append({"threshold": threshold, "coverage": round(coverage, 4),
                      "accuracy": round(acc, 4)})
        print(f"  {threshold:>10.1f} {coverage:>9.1%} {acc:>21.1%}")
    out["accuracy_vs_coverage"] = curve
    print("\n  This is the deployable answer to 'I want very high accuracy':")
    print("  auto-route the confident slice, send the rest to a human.")

    # =====================================================================
    print("\n" + "=" * 66)
    print("4. MERGED LABELS  (collapse the four overlapping support queues)")
    print("=" * 66)

    def merge(label: str) -> str:
        return "General Support" if label in GENERIC_SUPPORT else label

    ym_train = [merge(y) for y in y_train]
    ym_test = [merge(y) for y in y_test]

    t0 = time.time()
    merged_model = build().fit(text_train, ym_train)
    merged_predictions = list(merged_model.predict(text_test))
    acc = accuracy(ym_test, merged_predictions)
    f1 = macro_f1(ym_test, merged_predictions)
    majority = Counter(ym_train).most_common(1)[0][0]
    baseline = accuracy(ym_test, [majority] * len(ym_test))

    print(f"  classes: {len(set(ym_train))} (was {len(set(y_train))})")
    print(f"  majority baseline : {baseline:.1%}")
    print(f"  linear svm        : {acc:.1%}   macro-F1 {f1:.3f}   ({time.time()-t0:.0f}s)")
    out["merged_labels"] = {
        "n_classes": len(set(ym_train)),
        "majority_baseline": round(baseline, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "merged_into_one": sorted(GENERIC_SUPPORT),
    }
    print("\n  Same text, same model -- only the label taxonomy changed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
