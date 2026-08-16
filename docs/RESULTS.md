# Results

_Generated from `output/report.json`. Embedding backend: **offline:tfidf+svd** (dim 384). Corpus: **28,587** tickets._

## Corpus

| metric | value |
|---|---|
| tickets processed | 28,587 |
| rows rejected by validation | 0 |
| negative sentiment share | 42.9% |
| declared-vs-detected language mismatch | 11.7% |
| tickets with residual PII detected | 1,053 |

## Routing: predicting `queue` from free text

| method | accuracy | macro-F1 | n (held out) |
|---|---|---|---|
| `majority_class` | 29.2% | 0.045 | 7,145 |
| `keyword_rules` | 27.4% | 0.208 | 7,145 |
| `embedding_centroid` | 28.4% | 0.252 | 7,145 |
| `embedding_knn_k5` | 58.4% | 0.529 | 7,145 |
| `linear_svm_tfidf` | 60.6% | 0.604 | 7,145 |

## Retrieval: semantic vs keyword vs hybrid

| method | queue purity@k | precision@k | MRR | queries | k |
|---|---|---|---|---|---|
| `semantic` | 0.294 | 0.713 | 0.823 | 250 | 5 |
| `keyword` | 0.322 | 0.728 | 0.859 | 250 | 5 |
| `hybrid` | 0.294 | 0.745 | 0.844 | 250 | 5 |

_Relevance is a proxy: a result counts as relevant if it shares at least two tags with the query ticket. This rewards topical similarity, which is what routing and deduplication need. Read the numbers comparatively, not absolutely._


## Cross-lingual retrieval probe

| query | lang | semantic hits in other language | keyword hits |
|---|---|---|---|
| `Rechnung falsch berechnet` | de | 0 | 0 |
| `Server ist ausgefallen` | de | 1 | 0 |
| `cannot reset my password` | en | 0 | 0 |
| `data breach security incident` | en | 0 | 0 |

## Negative-sentiment rate by queue

| queue | negative rate | tickets |
|---|---|---|
| Technical Support | 53.8% | 8,362 |
| Service Outages and Maintenance | 53.5% | 1,148 |
| IT Support | 46.2% | 3,433 |
| Human Resources | 44.6% | 576 |
| Product Support | 43.6% | 5,252 |
| Returns and Exchanges | 36.2% | 1,437 |
| Billing and Payments | 33.1% | 2,788 |
| General Inquiry | 32.1% | 405 |
| Customer Service | 29.3% | 4,268 |
| Sales and Pre-Sales | 21.2% | 918 |

## High-priority share by queue

| queue | high-priority share | tickets |
|---|---|---|
| Service Outages and Maintenance | 71.0% | 1,148 |
| Technical Support | 58.6% | 8,362 |
| IT Support | 48.8% | 3,433 |
| Billing and Payments | 30.6% | 2,788 |
| Product Support | 29.6% | 5,252 |
| Returns and Exchanges | 21.4% | 1,437 |
| Customer Service | 18.9% | 4,268 |
| Sales and Pre-Sales | 17.5% | 918 |
| General Inquiry | 12.8% | 405 |
| Human Resources | 9.5% | 576 |

## Stage timings (single machine, offline runner)

| stage | seconds |
|---|---|
| load | 1.0 |
| enrich | 8.4 |
| embed | 205.5 |
| index | 16.6 |
| eval_routing | 107.9 |
| eval_retrieval | 59.5 |
| kpis | 137.8 |
