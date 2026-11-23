# Final dev results: RETECO sub-track 2a

Generated 2026-11-23 20:29 UTC by `scripts/compare_runs.py`.

- Retrieval: official starter-kit BM25 (Pyserini Lucene analyzer, gensim `LuceneBM25Model`, k1=0.9, b=0.4), top 100 per turn; nDCG@10 via pytrec_eval, averaged per domain, then over the 11 domains.
- Significance: two-sided paired randomization test vs `baseline-turn-history` over all 858 dev turns, 10,000 sign-flip permutations (seed 2027); the statistic is the macro-averaged difference (each turn weighted 1 / (11 x turns in its domain)).
- Protocol: variants were designed and selected on train only; dev was run once with fixed prompts and settings. The by-turn variant (rw2+reason+hist) was fixed on train before the dev run (train macro 0.5007 vs 0.5005 for reason+hist).
- LLM: Claude Haiku 5.5 (anthropic/claude-haiku-5-5) via LiteLLM, reasoning_effort=low, max_tokens=1024; prompts in prompts/rewrite_v1.txt and prompts/reason_v1.txt.
- Queries follow the starter kit's +History layout (text, then 'Conversation History:' + history). rw2+hist = rewrite x2 + turn + history; reason+hist = turn + history + note; rw2+reason+hist = rewrite x2 + turn + history + note. Turn 1 keeps its original query (no rewrite); notes are generated for every turn and capped at 60 words.
- Dev generation: rewrite-v1 had 0 fallbacks; reason-v1 had 0 fallbacks and 195 of 858 notes cut at 60 words.
- Smoke tests, not in the reports below, cost a further $0.0037 (two rewrite-v1 dev smoke runs and one reason-v1 train smoke run).

## Per-domain nDCG@10

| domain | baseline-turn | baseline-turn-history | rewrite-v1 | rw2+hist | reason+hist | rw2+reason+hist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biology | 0.2034 | 0.6065 | 0.4946 | 0.6304 | 0.7245 | 0.7248 |
| drones | 0.1695 | 0.2954 | 0.2494 | 0.3165 | 0.2718 | 0.2741 |
| earth_science | 0.2195 | 0.6264 | 0.4525 | 0.6390 | 0.6885 | 0.6990 |
| economics | 0.1629 | 0.4740 | 0.3027 | 0.4694 | 0.4974 | 0.4951 |
| hardware | 0.1605 | 0.3378 | 0.2496 | 0.3268 | 0.3114 | 0.2989 |
| law | 0.1147 | 0.3437 | 0.2472 | 0.3614 | 0.3893 | 0.3870 |
| medicalsciences | 0.1168 | 0.1730 | 0.1805 | 0.1770 | 0.1836 | 0.1875 |
| politics | 0.1514 | 0.3219 | 0.3025 | 0.3290 | 0.3451 | 0.3451 |
| psychology | 0.2596 | 0.5726 | 0.4347 | 0.6049 | 0.6494 | 0.6526 |
| robotics | 0.1574 | 0.5123 | 0.3402 | 0.5176 | 0.6056 | 0.6013 |
| sustainable_living | 0.2938 | 0.5531 | 0.4213 | 0.5616 | 0.5879 | 0.5914 |
| **macro (11)** | **0.1827** | **0.4379** | **0.3341** | **0.4485** | **0.4777** | **0.4779** |

## Paired randomization test vs `baseline-turn-history`

| run | train macro | dev macro | delta | p-value | turns better / same / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-turn` | 0.1837 | 0.1827 | -0.2552 | 0.0001 | 53 / 360 / 445 |
| `rewrite-v1` | 0.3474 | 0.3341 | -0.1038 | 0.0001 | 136 / 397 / 325 |
| `rw2+hist` | 0.4638 | 0.4485 | +0.0106 | 0.0009 | 125 / 651 / 82 |
| `reason+hist` | 0.5005 | 0.4777 | +0.0398 | 0.0001 | 269 / 410 / 179 |
| `rw2+reason+hist` | 0.5007 | 0.4779 | +0.0400 | 0.0001 | 273 / 402 / 183 |

## By turn number: `baseline-turn-history` vs `rw2+reason+hist` (turns pooled over all domains)

| turn | turns | `baseline-turn-history` | `rw2+reason+hist` | delta | better / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 211 | 0.2670 | 0.3903 | +0.1234 | 99 / 41 |
| T2 | 211 | 0.5466 | 0.5556 | +0.0090 | 53 / 62 |
| T3 | 211 | 0.5568 | 0.5787 | +0.0219 | 56 / 49 |
| T4 | 121 | 0.5632 | 0.6120 | +0.0488 | 34 / 21 |
| T5+ | 104 | 0.5716 | 0.6315 | +0.0599 | 31 / 10 |

## LLM cost (Claude Haiku 5.5, $0.10 / $0.50 per million input / output tokens)

| report | variant | split | cost (USD) |
| --- | --- | --- | ---: |
| `rewrite-v1_dev_report.json` | rewrite-v1 | dev | 0.0519 |
| `rewrite-v1_train_report.json` | rewrite-v1 | train | 0.1339 |
| `reason-v1_train_report.json` | reason-v1 | train | 0.2620 |
| `reason-v1_dev_report.json` | reason-v1 | dev | 0.1058 |
| **total** |  |  | **0.5537** |

CSV files: `results/final_dev_table.csv`, `results/final_dev_significance.csv`, `results/final_dev_by_turn.csv`; per-run files `results/<run>_dev.csv`.
