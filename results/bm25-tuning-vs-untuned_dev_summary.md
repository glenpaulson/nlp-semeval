# BM25 tuning vs untuned, dev results: RETECO sub-track 2a

Generated 2026-11-26 20:17 UTC by `scripts/compare_runs.py`.

- Retrieval: starter-kit BM25 (Pyserini Lucene analyzer, gensim `LuceneBM25Model`), top 100 per turn; k1=0.9, b=0.4 except runs named `__bm25-k0.9-b0.6`, which use k1=0.9, b=0.6 (computed by `scripts/tune_bm25.py`, rankings identical to `scripts/eval_queries.py` at k1=0.9, b=0.4); nDCG@10 via pytrec_eval, averaged per domain, then over the 11 domains.
- Significance: two-sided paired randomization test vs `rw2+reason+hist` over all 858 dev turns, 10,000 sign-flip permutations (seed 2027); the statistic is the macro-averaged difference (each turn weighted 1 / (11 x turns in its domain)).
- BM25 k1/b chosen on train only (49-setting grid, scripts/tune_bm25.py); dev run once with k1=0.9, b=0.6.

## Per-domain nDCG@10

| domain | rw2+reason+hist | rw2+reason+hist__bm25-k0.9-b0.6 |
| --- | ---: | ---: |
| biology | 0.7248 | 0.6615 |
| drones | 0.2741 | 0.3007 |
| earth_science | 0.6990 | 0.6063 |
| economics | 0.4951 | 0.5020 |
| hardware | 0.2989 | 0.3550 |
| law | 0.3870 | 0.4083 |
| medicalsciences | 0.1875 | 0.2222 |
| politics | 0.3451 | 0.3686 |
| psychology | 0.6526 | 0.6304 |
| robotics | 0.6013 | 0.5769 |
| sustainable_living | 0.5914 | 0.5924 |
| **macro (11)** | **0.4779** | **0.4749** |

## Paired randomization test vs `rw2+reason+hist`

| run | train macro | dev macro | delta | p-value | turns better / same / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rw2+reason+hist__bm25-k0.9-b0.6` | 0.5061 | 0.4749 | -0.0029 | 0.4461 | 163 / 513 / 182 |

## By turn number: `rw2+reason+hist` vs `rw2+reason+hist__bm25-k0.9-b0.6` (turns pooled over all domains)

| turn | turns | `rw2+reason+hist` | `rw2+reason+hist__bm25-k0.9-b0.6` | delta | better / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 211 | 0.3903 | 0.3561 | -0.0343 | 32 / 58 |
| T2 | 211 | 0.5556 | 0.5464 | -0.0092 | 39 / 43 |
| T3 | 211 | 0.5787 | 0.5747 | -0.0040 | 49 / 38 |
| T4 | 121 | 0.6120 | 0.5979 | -0.0141 | 23 / 21 |
| T5+ | 104 | 0.6315 | 0.6053 | -0.0262 | 20 / 22 |

CSV files: `results/bm25-tuning-vs-untuned_dev_table.csv`, `results/bm25-tuning-vs-untuned_dev_significance.csv`, `results/bm25-tuning-vs-untuned_dev_by_turn.csv`; per-run files `results/<run>_dev.csv`.
