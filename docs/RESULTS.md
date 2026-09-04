# Results

_Generated from `output/report.json`. Embedding backend: **sentence-transformers:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (dim 384). Corpus: **28,587** tickets._

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
| `embedding_knn_k5` | 64.6% | 0.605 | 7,145 |
| `linear_svm_tfidf` | 64.8% | 0.645 | 7,145 |

## Retrieval: semantic vs keyword vs hybrid

| method | queue purity@k | precision@k | MRR | queries | k |
|---|---|---|---|---|---|
| `semantic` | 0.346 | 0.744 | 0.835 | 250 | 5 |
| `keyword` | 0.322 | 0.728 | 0.859 | 250 | 5 |
| `hybrid` | 0.352 | 0.758 | 0.876 | 250 | 5 |

_Relevance is a proxy: a result counts as relevant if it shares at least two tags with the query ticket. This rewards topical similarity, which is what routing and deduplication need. Read the numbers comparatively, not absolutely._


## Cross-lingual retrieval probe

| query | lang | semantic hits in other language | keyword hits |
|---|---|---|---|
| `Rechnung falsch berechnet` | de | 0 | 0 |
| `Server ist ausgefallen` | de | 1 | 0 |
| `cannot reset my password` | en | 0 | 0 |
| `data breach security incident` | en | 0 | 0 |

Read alone, this looks like the multilingual embedding buys nothing across languages. It does not: with 12,249 German tickets in the corpus, a German query's unrestricted top-5 is dominated by German tickets on volume alone, before cross-lingual ability ever gets a chance to show up. The candidate pool, not the model, is what this table measures.


### Cross-lingual retrieval, language-filtered

| method | queue purity@k | precision@k | MRR | queries | k |
|---|---|---|---|---|---|
| `semantic` | 0.313 | 0.685 | 0.787 | 250 | 5 |
| `keyword` | 0.266 | 0.660 | 0.768 | 250 | 5 |
| `hybrid` | 0.300 | 0.698 | 0.799 | 250 | 5 |

Same shared-tag proxy as the main retrieval table, but the candidate pool is restricted to the *other* language before ranking -- the fix `/search?q=...&language=en` gives a live user. Forced to match across the language boundary, semantic retrieval still beats keyword search, which is the fair test the unrestricted probe above could never pass.


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
| enrich | 9.9 |
| embed | 1535.0 |
| index | 28.1 |
| eval_routing | 241.6 |
| eval_retrieval | 23.1 |
| eval_cross_lingual_retrieval | 46.8 |
| kpis | 31.3 |
