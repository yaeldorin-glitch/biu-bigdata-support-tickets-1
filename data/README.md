# Dataset

## Source

**Customer IT Support — Ticket Dataset** (also published as *Multilingual
Customer Support Tickets*) by Tobias Bueck.

- Kaggle: <https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets>
- Hugging Face mirror: <https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets>
- Licence: **CC BY-NC 4.0** — attribution required, non-commercial use only.

The file used in this project is `aa_dataset-tickets-multi-lang-5-2-50-version.csv`
(28,587 rows). Other files in the same Kaggle dataset are larger and contain
more languages; the pipeline reads any of them, since the column set is shared.

## Why this dataset fits the brief

The brief requires *semi-structured or unstructured* data — "clean tabular data
alone is not sufficient". This dataset qualifies on two counts:

1. **Free text.** `subject`, `body` and `answer` are natural-language fields
   averaging 387 characters, in English and German. This is what the embeddings,
   semantic search and RAG layer operate on.
2. **A ragged, sparse tag structure.** `tag_1` … `tag_8` hold a variable-length
   list of 1,255 distinct labels flattened into fixed columns. `tag_5` is null in
   49% of rows and `tag_8` in 98%. Normalising this into a real array is part of
   the transformation, not a given.

## Not committed here

The full 26MB CSV is **not** in this repository. It is CC BY-NC licensed and
belongs to its author, and a 26MB binary would bloat the git history.

`sample/tickets_sample.csv` (300 rows) **is** committed so the project is
runnable immediately. It is stratified by `queue` × `language` so it preserves
the corpus shape — a uniform random sample would likely contain no
"General Inquiry" tickets at all, since they are 1.4% of the data.

## Getting the full dataset

```bash
# option 1: kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d tobiasbueck/multilingual-customer-support-tickets -p data/raw --unzip
mv data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv data/raw/tickets.csv

# option 2: download manually from the Kaggle page and save as:
#   data/raw/tickets.csv
```

Then regenerate the sample if you want:

```bash
make sample
```

## Schema

| column | type | notes |
|---|---|---|
| `subject` | text | **null in 3,838 rows (13.4%)** — handled, not an error |
| `body` | text | the main free-text field; never null |
| `answer` | text | agent reply; null in 7 rows |
| `type` | keyword | Incident, Request, Problem, Change |
| `queue` | keyword | 10 values — the routing label the AI predicts |
| `priority` | keyword | low, medium, high |
| `language` | keyword | `en` (16,338) / `de` (12,249) |
| `version` | int | 400, 52 or 51 — dataset build, not ticket version |
| `tag_1`…`tag_8` | keyword | ragged list, 1,255 distinct values |

## Observed distributions

```
type       Incident 11,466 | Request 8,187 | Problem 6,012 | Change 2,922
priority   medium 11,515 | high 11,178 | low 5,894
language   en 16,338 | de 12,249

queue      Technical Support               8,362   (29.3%)
           Product Support                 5,252   (18.4%)
           Customer Service                4,268   (14.9%)
           IT Support                      3,433   (12.0%)
           Billing and Payments            2,788    (9.8%)
           Returns and Exchanges           1,437    (5.0%)
           Service Outages and Maintenance 1,148    (4.0%)
           Sales and Pre-Sales               918    (3.2%)
           Human Resources                   576    (2.0%)
           General Inquiry                   405    (1.4%)
```

The 29.3% majority class is why the evaluation reports macro-F1 alongside
accuracy: a model that always answers "Technical Support" scores 29.3% accuracy
while being useless.

## Data quality notes we found

- **The `language` label is unreliable, and only in one direction.** Independent
  detection disagrees with the declared label on 11.7% of rows overall — but that
  headline hides the real shape. Rows labelled `de` are wrong **27.2%** of the
  time (n=12,210); rows labelled `en` are wrong **0.05%** of the time (n=16,234).
  More than a quarter of the "German" corpus is English text. Row 5000 is a clean
  example: labelled `de`, body reads "Dear Customer Service, I am reaching out…".
  The asymmetry points at an upstream generation or import defect rather than
  detector error, and anything keyed off this field is misfiring on those rows.
- **PII is already substituted.** The publisher replaced identifiers with
  placeholder tokens such as `<name>`, `<tel_num>` and `<acc_num>`. The pipeline
  still runs its own redaction pass over the text, because relying on an
  upstream promise is not a control.
- **Newlines are stored as literal backslash-n**, not as real newlines. Left
  alone, this puts a spurious `n` token in every text vector.

## Attribution

If you use this data, credit the original author per CC BY-NC 4.0:

> Tobias Bueck, *Customer IT Support — Ticket Dataset*, Kaggle / Hugging Face.
> Licensed CC BY-NC 4.0.
