# How high can routing accuracy actually go?

Two scripts answer two different questions. `scripts/experiments.py` asks *which
model is best*. `scripts/ceiling.py` asks the prior question: **is the remaining
error the model's fault, or the labels'?**

Reproduce both with:

```bash
python scripts/experiments.py --full
python scripts/ceiling.py --full
```

All figures below are on the full 28,587-ticket corpus, on the identical
stratified split used everywhere else (seed 42, 25% held out, n=7,145).

---

## 1. What improved the number

| method | accuracy | macro-F1 |
|---|---|---|
| majority class | 29.2% | 0.045 |
| keyword rules | 27.4% | 0.208 |
| embedding centroid | 28.6% | 0.252 |
| k-NN, k=100 | 40.4% | 0.235 |
| k-NN, k=50 | 42.3% | 0.267 |
| k-NN, k=30 | 44.4% | 0.299 |
| k-NN, k=15 | 48.2% | 0.380 |
| **k-NN, k=5** | **57.8%** | **0.524** |
| logistic regression, TF-IDF, balanced | 49.4% | 0.493 |
| linear SVM, TF-IDF | 61.6% | 0.597 |
| linear SVM, TF-IDF, balanced | 60.9% | 0.608 |
| **linear SVM, word+char TF-IDF, balanced** | **61.7%** | **0.615** |

Two lessons:

- **`k` dominated.** Accuracy falls monotonically as `k` grows. With ten classes
  and a long tail, a wide neighbourhood pulls every vote toward the majority
  queue. `k=15` was simply a bad default.
- **A linear SVM on sparse TF-IDF beats k-NN on dense embeddings**, trains in
  four seconds, and needs no embedding model at all. Character n-grams add a
  further point of macro-F1 by handling German compounds and misspellings that
  word n-grams tokenise badly.

`class_weight="balanced"` costs 0.7 points of accuracy and buys 0.011 macro-F1 —
the right trade when the smallest queue is 1.4% of the corpus.

## 2. Leakage probes — where the ceiling looked like it was

Adding fields that do **not** exist when a new ticket arrives:

| probe | accuracy | macro-F1 |
|---|---|---|
| + `answer` field (agent's reply) | 64.7% | 0.656 |
| + `tags` | 61.0% | 0.610 |

Neither is deployable. `answer` is written after the ticket is handled; `tags`
are produced by the same labelling process as `queue`. They are measured only to
bound the headroom — and adding the agent's own reply, which is about as strong a
hint as exists, buys just **three points**.

## 3. The Bayes floor — do near-identical tickets even agree?

For a 6,000-ticket sample we found each ticket's nearest neighbour in embedding
space and checked whether its `queue` label matched.

| similarity of nearest neighbour | pairs | same queue |
|---|---|---|
| ≥ 0.95 | 1,084 | **99.3%** |
| 0.90 – 0.95 | 1,544 | 94.8% |
| 0.80 – 0.90 | 2,121 | 72.7% |
| < 0.80 | 1,251 | 51.0% |

**This is the result that corrected our earlier conclusion.** When two tickets
are genuine textual twins, their labels agree 99.3% of the time. The labels are
*not* noisy. So the irreducible error is low, and the gap between 61.7% and the
top of the table is mostly **model capacity**, not label ambiguity.

The honest revised position: a fine-tuned multilingual transformer should
plausibly reach the low-to-mid 70s. We have not run it, so that is a projection,
not a measurement.

What *is* ambiguous is a specific cluster. The top confusions are entirely among
four semantically overlapping queues:

```
243  Product Support    →  Technical Support
217  Technical Support  →  Product Support
185  IT Support         →  Technical Support
184  Technical Support  →  Customer Service
174  Technical Support  →  IT Support
169  Customer Service   →  Technical Support
```

Meanwhile the semantically distinct queues are handled well — **Billing and
Payments recall 0.821**, **Service Outages 0.749**. The model is not weak
everywhere; it is weak exactly where a human would also hesitate.

## 4. Three ways to get genuinely high accuracy

If the requirement is "very high accuracy", the answer is to change the question,
and each option is measured rather than argued.

### a. Suggest instead of decide

| | accuracy |
|---|---|
| top-1 | 60.6% |
| top-2 | **77.7%** |
| top-3 | **87.3%** |

If the interface offers an agent three ranked queues rather than auto-filing,
the right metric is top-3, and it is already 87.3%.

### b. Abstain when unsure

Auto-route only above a confidence threshold; send the rest to a human.

| threshold | coverage | accuracy on the covered slice |
|---|---|---|
| 0.0 | 100.0% | 60.6% |
| 0.3 | 91.3% | 63.8% |
| 0.4 | 64.8% | 72.1% |
| 0.5 | 37.6% | 81.4% |
| 0.6 | 20.4% | **90.9%** |
| 0.7 | 9.7% | **96.3%** |
| 0.8 | 4.1% | 99.0% |

This is the deployable answer. Automating the confident fifth of the queue at
91% accuracy is worth far more operationally than automating everything at 61%.

### c. Fix the taxonomy

Collapse the four entangled queues into one `General Support`:

| | value |
|---|---|
| classes | 7 (was 10) |
| majority baseline | 74.6% |
| linear SVM | **85.1%** (macro-F1 0.522) |

Same text, same model — only the label taxonomy changed. The 24-point jump is a
measurement of how much of the original problem was the taxonomy rather than the
data.

Note the honest caveat: the majority baseline also jumps to 74.6%, so the *lift*
over baseline shrinks. Merged labels look better on a slide and are genuinely
more useful operationally, but they are an easier problem, not a better model.

---

## What we would try next, in order

1. **Fine-tuned multilingual transformer** (`sentence-transformers` embeddings, or
   full fine-tuning of XLM-R). The near-duplicate analysis says the headroom is
   real. One environment variable gets the embedding half of this:
   `EMBEDDING_BACKEND=sentence-transformers`.
2. **Confidence-thresholded deployment** at ~0.6, per the table above. No modelling
   work at all, immediate operational value.
3. **Hierarchical classification** — first predict `General Support` vs the six
   distinct queues (an easy, high-accuracy decision), then a second model only for
   the four-way split inside it. Isolates the hard problem instead of letting it
   contaminate every prediction.
4. **Relabelling guidance** for the four overlapping queues. If humans cannot
   state the rule that separates Technical from Product Support, no model will
   infer it.
