# BM25 tuning, dev results: RETECO sub-track 2a

Generated 2026-11-26 20:17 UTC by `scripts/compare_runs.py`.

- Retrieval: starter-kit BM25 (Pyserini Lucene analyzer, gensim `LuceneBM25Model`), top 100 per turn; k1=0.9, b=0.4 except runs named `__bm25-k0.9-b0.6`, which use k1=0.9, b=0.6 (computed by `scripts/tune_bm25.py`, rankings identical to `scripts/eval_queries.py` at k1=0.9, b=0.4); nDCG@10 via pytrec_eval, averaged per domain, then over the 11 domains.
- Significance: two-sided paired randomization test vs `baseline-turn-history` over all 858 dev turns, 10,000 sign-flip permutations (seed 2027); the statistic is the macro-averaged difference (each turn weighted 1 / (11 x turns in its domain)).
- BM25 k1/b chosen on train only (49-setting grid, scripts/tune_bm25.py); dev run once with k1=0.9, b=0.6.

## Per-domain nDCG@10

| domain | baseline-turn-history | baseline-turn-history__bm25-k0.9-b0.6 | rw2+reason+hist | rw2+reason+hist__bm25-k0.9-b0.6 |
| --- | ---: | ---: | ---: | ---: |
| biology | 0.6065 | 0.5353 | 0.7248 | 0.6615 |
| drones | 0.2954 | 0.3522 | 0.2741 | 0.3007 |
| earth_science | 0.6264 | 0.5489 | 0.6990 | 0.6063 |
| economics | 0.4740 | 0.4690 | 0.4951 | 0.5020 |
| hardware | 0.3378 | 0.3730 | 0.2989 | 0.3550 |
| law | 0.3437 | 0.3596 | 0.3870 | 0.4083 |
| medicalsciences | 0.1730 | 0.1998 | 0.1875 | 0.2222 |
| politics | 0.3219 | 0.3455 | 0.3451 | 0.3686 |
| psychology | 0.5726 | 0.5644 | 0.6526 | 0.6304 |
| robotics | 0.5123 | 0.4833 | 0.6013 | 0.5769 |
| sustainable_living | 0.5531 | 0.5077 | 0.5914 | 0.5924 |
| **macro (11)** | **0.4379** | **0.4308** | **0.4779** | **0.4749** |

## Paired randomization test vs `baseline-turn-history`

| run | train macro | dev macro | delta | p-value | turns better / same / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-turn-history__bm25-k0.9-b0.6` | 0.4574 | 0.4308 | -0.0071 | 0.0721 | 127 / 563 / 168 |
| `rw2+reason+hist` | 0.5007 | 0.4779 | +0.0400 | 0.0001 | 273 / 402 / 183 |
| `rw2+reason+hist__bm25-k0.9-b0.6` | 0.5061 | 0.4749 | +0.0371 | 0.0001 | 267 / 391 / 200 |

## By turn number: `baseline-turn-history` vs `rw2+reason+hist__bm25-k0.9-b0.6` (turns pooled over all domains)

| turn | turns | `baseline-turn-history` | `rw2+reason+hist__bm25-k0.9-b0.6` | delta | better / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 211 | 0.2670 | 0.3561 | +0.0891 | 89 / 45 |
| T2 | 211 | 0.5466 | 0.5464 | -0.0002 | 53 / 64 |
| T3 | 211 | 0.5568 | 0.5747 | +0.0179 | 59 / 48 |
| T4 | 121 | 0.5632 | 0.5979 | +0.0347 | 35 / 24 |
| T5+ | 104 | 0.5716 | 0.6053 | +0.0337 | 31 / 19 |

CSV files: `results/bm25-tuning_dev_table.csv`, `results/bm25-tuning_dev_significance.csv`, `results/bm25-tuning_dev_by_turn.csv`; per-run files `results/<run>_dev.csv`.
