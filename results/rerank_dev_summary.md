# Final dev results: RETECO sub-track 2a

Generated 2026-11-26 19:19 UTC by `scripts/compare_runs.py`.

- Retrieval: official starter-kit BM25 (Pyserini Lucene analyzer, gensim `LuceneBM25Model`, k1=0.9, b=0.4), top 100 per turn; nDCG@10 via pytrec_eval, averaged per domain, then over the 11 domains.
- Significance: two-sided paired randomization test vs `baseline-turn-history` over all 858 dev turns, 10,000 sign-flip permutations (seed 2027); the statistic is the macro-averaged difference (each turn weighted 1 / (11 x turns in its domain)).
- Reranker: Claude Haiku 5.5 scores the top 30 of rw2+reason+hist in one listwise call (prompts/rerank_v1.txt); final score = 0.3 x LLM/10 + 0.7 x normalized BM25 reciprocal rank. alpha=0.3 chosen on a 700-turn train sample (BM25 0.5112 to 0.5521); dev run once.

## Per-domain nDCG@10

| domain | baseline-turn-history | rw2+reason+hist | rw2+reason+hist+rerank-v1-a0.3 |
| --- | ---: | ---: | ---: |
| biology | 0.6065 | 0.7248 | 0.7294 |
| drones | 0.2954 | 0.2741 | 0.3205 |
| earth_science | 0.6264 | 0.6990 | 0.6581 |
| economics | 0.4740 | 0.4951 | 0.6006 |
| hardware | 0.3378 | 0.2989 | 0.3811 |
| law | 0.3437 | 0.3870 | 0.4364 |
| medicalsciences | 0.1730 | 0.1875 | 0.2620 |
| politics | 0.3219 | 0.3451 | 0.3757 |
| psychology | 0.5726 | 0.6526 | 0.6855 |
| robotics | 0.5123 | 0.6013 | 0.6195 |
| sustainable_living | 0.5531 | 0.5914 | 0.6698 |
| **macro (11)** | **0.4379** | **0.4779** | **0.5217** |

## Paired randomization test vs `baseline-turn-history`

| run | train macro | dev macro | delta | p-value | turns better / same / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rw2+reason+hist` | 0.5007 | 0.4779 | +0.0400 | 0.0001 | 273 / 402 / 183 |
| `rw2+reason+hist+rerank-v1-a0.3` | - | 0.5217 | +0.0838 | 0.0001 | 364 / 264 / 230 |

## By turn number: `baseline-turn-history` vs `rw2+reason+hist+rerank-v1-a0.3` (turns pooled over all domains)

| turn | turns | `baseline-turn-history` | `rw2+reason+hist+rerank-v1-a0.3` | delta | better / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 211 | 0.2670 | 0.5023 | +0.2354 | 117 / 33 |
| T2 | 211 | 0.5466 | 0.5917 | +0.0451 | 81 / 62 |
| T3 | 211 | 0.5568 | 0.5800 | +0.0232 | 79 / 76 |
| T4 | 121 | 0.5632 | 0.6127 | +0.0495 | 49 / 33 |
| T5+ | 104 | 0.5716 | 0.6231 | +0.0515 | 38 / 26 |

CSV files: `results/rerank_dev_table.csv`, `results/rerank_dev_significance.csv`, `results/rerank_dev_by_turn.csv`; per-run files `results/<run>_dev.csv`.
