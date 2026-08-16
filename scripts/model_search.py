"""Wide model search for routing accuracy.

Everything in `experiments.py` was a first pass. This goes further: regularisation
sweeps, other model families, feature additions, per-language models, a two-stage
hierarchy, and stacked ensembles.

Same stratified split as everywhere else (seed 42, 25% held out), so every number
is comparable to docs/RESULTS.md and docs/CEILING.md.

    python scripts/model_search.py --full
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tickets.ai.evaluate import accuracy, macro_f1  # noqa: E402
from tickets.core.enrich import enrich_ticket  # noqa: E402
from tickets.offline_pipeline import load_tickets  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402

RESULTS: list[dict] = []
BEST = {"score": -1.0, "name": None}


def record(name, true, pred, note=""):
    acc = accuracy(true, pred)
    f1 = macro_f1(true, pred)
    RESULTS.append({"method": name, "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4), "note": note})
    flag = ""
    if f1 > BEST["score"]:
        BEST.update(score=f1, name=name)
        flag = "  <-- best macro-F1 so far"
    print(f"  {name:<46} acc={acc:.1%}  macroF1={f1:.3f} {note}{flag}")


def stratified_split(items, key, test_fraction=0.25, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    train, test = [], []
    for g in groups:
        arr = groups[g][:]
        rng.shuffle(arr)
        cut = max(1, int(len(arr) * test_fraction))
        test.extend(arr[:cut])
        train.extend(arr[cut:])
    return train, test


def make_features(max_word=150_000, max_char=150_000, min_df=2, word_ngram=(1, 2)):
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=word_ngram, min_df=min_df, max_df=0.6,
                                 sublinear_tf=True, max_features=max_word)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=max_char)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "tickets.csv")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "model_search.json")
    args = parser.parse_args()

    print("loading + enriching...")
    tickets, _ = load_tickets(args.csv, None if args.full else args.limit)
    for t in tickets:
        enrich_ticket(t)
    tickets = [t for t in tickets if t.queue]
    print(f"  {len(tickets):,} labelled tickets\n")

    train, test = stratified_split(tickets, key=lambda t: t.queue)
    y_train = [t.queue for t in train]
    y_test = [t.queue for t in test]
    txt_train = [t.text_redacted or t.text for t in train]
    txt_test = [t.text_redacted or t.text for t in test]
    print(f"train={len(train):,}  test={len(test):,}\n")

    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC

    # =====================================================================
    print("=" * 74)
    print("A. REFERENCE — the current best (linear SVM, word+char, balanced)")
    print("=" * 74)
    feats = make_features()
    Xtr = feats.fit_transform(txt_train)
    Xte = feats.transform(txt_test)
    print(f"  features: {Xtr.shape[1]:,}")
    base = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000).fit(Xtr, y_train)
    record("linear_svm_C1_balanced", y_test, list(base.predict(Xte)))

    # =====================================================================
    print("\n" + "=" * 74)
    print("B. REGULARISATION SWEEP — is C=1 even right?")
    print("=" * 74)
    for C in (0.1, 0.3, 0.5, 2.0, 5.0, 10.0):
        m = LinearSVC(C=C, class_weight="balanced", max_iter=4000).fit(Xtr, y_train)
        record(f"linear_svm_C{C}_balanced", y_test, list(m.predict(Xte)))

    # =====================================================================
    print("\n" + "=" * 74)
    print("C. OTHER MODEL FAMILIES on the same sparse features")
    print("=" * 74)
    for name, model in [
        ("complement_nb", ComplementNB(alpha=0.3)),
        ("ridge_balanced", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ("sgd_modified_huber", SGDClassifier(loss="modified_huber", alpha=1e-5,
                                             class_weight="balanced", max_iter=40, tol=1e-4,
                                             random_state=42)),
        ("logreg_saga_balanced", LogisticRegression(solver="saga", C=4.0, max_iter=400,
                                                    class_weight="balanced")),
    ]:
        t0 = time.time()
        try:
            m = model.fit(Xtr, y_train)
            record(name, y_test, list(m.predict(Xte)), f"({time.time()-t0:.0f}s)")
        except Exception as exc:
            print(f"  {name:<46} FAILED: {exc}")

    # =====================================================================
    print("\n" + "=" * 74)
    print("D. RICHER FEATURES — more vocabulary, trigrams, min_df=1")
    print("=" * 74)
    for label, kwargs in [
        ("bigger_vocab", dict(max_word=300_000, max_char=300_000)),
        ("min_df1", dict(min_df=1)),
        ("word_trigrams", dict(word_ngram=(1, 3), max_word=300_000)),
    ]:
        f = make_features(**kwargs)
        A = f.fit_transform(txt_train)
        B = f.transform(txt_test)
        m = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000).fit(A, y_train)
        record(f"svm_{label}", y_test, list(m.predict(B)), f"({A.shape[1]:,} feats)")

    # =====================================================================
    print("\n" + "=" * 74)
    print("E. ADD SUBMISSION METADATA — type / priority / language")
    print("=" * 74)
    print("  These are chosen by the customer at submission time in most ticketing")
    print("  systems, so unlike `answer` and `tags` they are legitimately available.")
    print("  Flagged as an ASSUMPTION about the source system, not a certainty.")

    from sklearn.preprocessing import OneHotEncoder
    meta_train = [[t.type, t.priority, t.detected_language] for t in train]
    meta_test = [[t.type, t.priority, t.detected_language] for t in test]
    enc = OneHotEncoder(handle_unknown="ignore")
    Mtr = enc.fit_transform(meta_train)
    Mte = enc.transform(meta_test)

    Xtr_meta = sp.hstack([Xtr, Mtr * 2.0]).tocsr()   # upweight: 17 dense cols vs 140k sparse
    Xte_meta = sp.hstack([Xte, Mte * 2.0]).tocsr()
    m = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000).fit(Xtr_meta, y_train)
    record("svm_plus_type_priority_lang", y_test, list(m.predict(Xte_meta)),
           "ASSUMES metadata available at submission")

    # type alone, the most defensible of the three
    enc2 = OneHotEncoder(handle_unknown="ignore")
    T1 = enc2.fit_transform([[t.type] for t in train])
    T2 = enc2.transform([[t.type] for t in test])
    m = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000).fit(
        sp.hstack([Xtr, T1 * 2.0]).tocsr(), y_train)
    record("svm_plus_type_only", y_test, list(m.predict(sp.hstack([Xte, T2 * 2.0]).tocsr())),
           "ASSUMES type available at submission")

    # =====================================================================
    print("\n" + "=" * 74)
    print("F. PER-LANGUAGE MODELS — one model for en, one for de")
    print("=" * 74)
    print("  Uses DETECTED language, not the declared field, which is wrong on")
    print("  27.2% of German-labelled rows.")

    preds_by_lang = {}
    for lang in ("en", "de"):
        idx_tr = [i for i, t in enumerate(train) if t.detected_language == lang]
        idx_te = [i for i, t in enumerate(test) if t.detected_language == lang]
        if len(idx_tr) < 200 or not idx_te:
            continue
        f = make_features()
        A = f.fit_transform([txt_train[i] for i in idx_tr])
        B = f.transform([txt_test[i] for i in idx_te])
        m = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000).fit(
            A, [y_train[i] for i in idx_tr])
        for i, p in zip(idx_te, m.predict(B)):
            preds_by_lang[i] = p
        print(f"    {lang}: train={len(idx_tr):,} test={len(idx_te):,}")

    # Anything the per-language models did not cover falls back to the global model.
    global_preds = list(base.predict(Xte))
    combined = [preds_by_lang.get(i, global_preds[i]) for i in range(len(y_test))]
    record("per_language_models", y_test, combined,
           f"({len(preds_by_lang):,}/{len(y_test):,} covered by a language model)")

    # =====================================================================
    print("\n" + "=" * 74)
    print("G. TWO-STAGE HIERARCHY — split the entangled cluster separately")
    print("=" * 74)
    GENERIC = {"Technical Support", "Product Support", "IT Support", "Customer Service"}

    coarse_train = ["GENERIC" if y in GENERIC else y for y in y_train]
    coarse_test = ["GENERIC" if y in GENERIC else y for y in y_test]

    stage1 = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000).fit(Xtr, coarse_train)
    coarse_pred = list(stage1.predict(Xte))
    print(f"    stage 1 (7 classes): {accuracy(coarse_test, coarse_pred):.1%}")

    gi_tr = [i for i, y in enumerate(y_train) if y in GENERIC]
    f2 = make_features()
    A2 = f2.fit_transform([txt_train[i] for i in gi_tr])
    stage2 = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000).fit(
        A2, [y_train[i] for i in gi_tr])

    gi_te = [i for i, p in enumerate(coarse_pred) if p == "GENERIC"]
    fine = list(stage2.predict(f2.transform([txt_test[i] for i in gi_te]))) if gi_te else []
    print(f"    stage 2 (4 classes) on {len(gi_te):,} routed tickets")

    hier = list(coarse_pred)
    for i, p in zip(gi_te, fine):
        hier[i] = p
    record("two_stage_hierarchy", y_test, hier)

    # =====================================================================
    print("\n" + "=" * 74)
    print("H. DENSE-FEATURE MODELS — trees on SVD embeddings")
    print("=" * 74)
    from sklearn.decomposition import TruncatedSVD

    t0 = time.time()
    svd = TruncatedSVD(n_components=200, random_state=42)
    Dtr = svd.fit_transform(Xtr)
    Dte = svd.transform(Xte)
    print(f"    SVD to 200 dims in {time.time()-t0:.0f}s "
          f"(explained variance {svd.explained_variance_ratio_.sum():.1%})")

    for name, model in [
        ("extra_trees_400", ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                                 n_jobs=-1, random_state=42)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=200, random_state=42)),
    ]:
        t0 = time.time()
        try:
            m = model.fit(Dtr, y_train)
            record(name, y_test, list(m.predict(Dte)), f"({time.time()-t0:.0f}s)")
        except Exception as exc:
            print(f"  {name:<46} FAILED: {exc}")

    # =====================================================================
    print("\n" + "=" * 74)
    print("I. STACKED ENSEMBLE — combine sparse-linear + dense-tree + NB")
    print("=" * 74)
    print("  Meta-learner trained on out-of-fold probabilities so the stack never")
    print("  sees its own base models' training predictions.")

    from sklearn.model_selection import cross_val_predict

    t0 = time.time()
    svm_prob = LogisticRegression(solver="saga", C=4.0, max_iter=300, class_weight="balanced")
    nb = ComplementNB(alpha=0.3)

    oof_svm = cross_val_predict(svm_prob, Xtr, y_train, cv=3, method="predict_proba", n_jobs=1)
    oof_nb = cross_val_predict(nb, Xtr, y_train, cv=3, method="predict_proba", n_jobs=1)
    oof_gb = cross_val_predict(HistGradientBoostingClassifier(max_iter=120, random_state=42),
                               Dtr, y_train, cv=3, method="predict_proba", n_jobs=1)
    print(f"    out-of-fold probabilities in {time.time()-t0:.0f}s")

    meta_X = np.hstack([oof_svm, oof_nb, oof_gb])
    meta = LogisticRegression(max_iter=1000, class_weight="balanced").fit(meta_X, y_train)

    full_svm = svm_prob.fit(Xtr, y_train).predict_proba(Xte)
    full_nb = nb.fit(Xtr, y_train).predict_proba(Xte)
    full_gb = HistGradientBoostingClassifier(max_iter=120, random_state=42).fit(
        Dtr, y_train).predict_proba(Dte)
    record("stacked_ensemble", y_test,
           list(meta.predict(np.hstack([full_svm, full_nb, full_gb]))))

    # =====================================================================
    print("\n" + "=" * 74)
    print("SUMMARY — ranked by macro-F1")
    print("=" * 74)
    for r in sorted(RESULTS, key=lambda d: -d["macro_f1"])[:12]:
        print(f"  {r['macro_f1']:.3f}  {r['accuracy']:>6.1%}  {r['method']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"results": RESULTS, "best": BEST, "n_train": len(train), "n_test": len(test)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
