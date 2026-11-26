# Final dev results: BM25 + dense fusion, RETECO sub-track 2a

Generated 2026-11-26 23:41 UTC by `scripts/compare_runs.py`.

- Retrieval: BM25 = official starter-kit BM25 (Pyserini Lucene analyzer, gensim `LuceneBM25Model`, k1=0.9, b=0.4); dense = BAAI/bge-small-en-v1.5 via sentence-transformers (fp32 on Apple MPS, 512-token cap, bge query prefix, cosine similarity, document vectors cached as fp16); fusion = weighted Reciprocal Rank Fusion of the two top-100 runs, score = w/(k + rank_BM25) + (1 - w)/(k + rank_dense), written with rank-derived scores (1000 - rank). Top 100 per turn; nDCG@10 via pytrec_eval, averaged per domain, then over the 11 domains.
- Significance: two-sided paired randomization test vs `baseline-turn-history` over all 858 dev turns, 10,000 sign-flip permutations (seed 2027); the statistic is the macro-averaged difference (each turn weighted 1 / (11 x turns in its domain)).
- Chosen fusion setting: k=1, w=0.6 (w = BM25 weight). Selected on train only from a 31-setting grid (k in {1, 3, 5, 10, 20, 30, 60, 100} x w in {0.3, 0.5, 0.6, 0.7, 0.8, 0.9}, not all pairs; results/fusion_grid_train.csv) with a fixed rule: take the highest train macro, unless another setting within 0.002 of it has a higher average over its adjacent k and w cells (a plateau, not a spike).
- Selection reason: decided by the highest-train-macro rule. After the scoring fix the chosen setting scores 0.5302 on train and the runner-up k=3, w=0.6 scores 0.5275 (both re-scored the same way), a margin of 0.0027, outside the 0.002 tolerance. The grid CSV was not re-run and holds pre-fix values (each within 0.0003). The chosen cell lies on the grid edge (smallest k, and smallest w of the extended grid).
- Inputs chosen on train: BM25 query rw2+reason+hist (train 0.5007) and dense variant rw2+reason+hist__dense-bge-small, the best of 4 dense query variants (train 0.4629). Dev was run once for this decision; the re-run after the scoring fix used the same setting and only changes how ties inside the fused files are written.
- Fused vs rw2+reason+hist (BM25 alone) on dev: delta +0.0256, paired randomization p = 0.0001, turns better/same/worse 234/430/194. Fused vs dense alone: delta +0.0824, p = 0.0001. With 10,000 permutations, 0.0001 is the smallest possible p-value.
- Ties: fused runs write rank-derived scores, so pytrec_eval scores exactly the fused ranking and agrees with the official rank-based scorer on every turn. Single runs keep their native scores; pytrec_eval treats exact and float32-level ties in them as ties and breaks them by doc ID, as the official metric does (at most 0.0009 macro vs rank order).
- Fusion makes no LLM calls; the LLM cost below is for generating the dev queries (rewrite-v1 and reason-v1).

## Per-domain nDCG@10

| domain | baseline-turn | baseline-turn-history | rw2+reason+hist | rw2+reason+hist__dense-bge-small | rw2+reason+hist__rrf-bm25-dense-k1-w0.6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 0.2034 | 0.6065 | 0.7248 | 0.5773 | 0.7337 |
| drones | 0.1695 | 0.2954 | 0.2741 | 0.2793 | 0.2910 |
| earth_science | 0.2195 | 0.6264 | 0.6990 | 0.5274 | 0.6773 |
| economics | 0.1629 | 0.4740 | 0.4951 | 0.4901 | 0.5325 |
| hardware | 0.1605 | 0.3378 | 0.2989 | 0.2956 | 0.3325 |
| law | 0.1147 | 0.3437 | 0.3870 | 0.3798 | 0.4507 |
| medicalsciences | 0.1168 | 0.1730 | 0.1875 | 0.3069 | 0.2633 |
| politics | 0.1514 | 0.3219 | 0.3451 | 0.4014 | 0.4077 |
| psychology | 0.2596 | 0.5726 | 0.6526 | 0.5029 | 0.6556 |
| robotics | 0.1574 | 0.5123 | 0.6013 | 0.3777 | 0.5865 |
| sustainable_living | 0.2938 | 0.5531 | 0.5914 | 0.4940 | 0.6081 |
| **macro (11)** | **0.1827** | **0.4379** | **0.4779** | **0.4211** | **0.5035** |

## Paired randomization test vs `baseline-turn-history`

| run | train macro | dev macro | delta | p-value | turns better / same / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-turn` | 0.1837 | 0.1827 | -0.2552 | 0.0001 | 53 / 360 / 445 |
| `rw2+reason+hist` | 0.5007 | 0.4779 | +0.0400 | 0.0001 | 273 / 402 / 183 |
| `rw2+reason+hist__dense-bge-small` | 0.4629 | 0.4211 | -0.0167 | 0.1856 | 280 / 276 / 302 |
| `rw2+reason+hist__rrf-bm25-dense-k1-w0.6` | 0.5302 | 0.5035 | +0.0657 | 0.0001 | 329 / 338 / 191 |

## By turn number: `baseline-turn-history` vs `rw2+reason+hist__rrf-bm25-dense-k1-w0.6` (turns pooled over all domains)

| turn | turns | `baseline-turn-history` | `rw2+reason+hist__rrf-bm25-dense-k1-w0.6` | delta | better / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 211 | 0.2670 | 0.4244 | +0.1575 | 103 / 42 |
| T2 | 211 | 0.5466 | 0.5715 | +0.0249 | 71 / 55 |
| T3 | 211 | 0.5568 | 0.5924 | +0.0356 | 76 / 54 |
| T4 | 121 | 0.5632 | 0.6305 | +0.0673 | 44 / 22 |
| T5+ | 104 | 0.5716 | 0.6315 | +0.0599 | 35 / 18 |

## LLM cost (Claude Haiku 5.5, $0.10 / $0.50 per million input / output tokens)

| report | variant | split | cost (USD) |
| --- | --- | --- | ---: |
| `rewrite-v1_dev_report.json` | rewrite-v1 | dev | 0.0519 |
| `reason-v1_dev_report.json` | reason-v1 | dev | 0.1058 |
| **total** |  |  | **0.1578** |

CSV files: `results/final_fusion_dev_table.csv`, `results/final_fusion_dev_significance.csv`, `results/final_fusion_dev_by_turn.csv`; per-run files `results/<run>_dev.csv`.

## How to regenerate

From a fresh clone, after the setup in `SETUP.md` (`uv sync`, the Track 2 data in `reteco_data/`, Java with `JAVA_HOME` and `JVM_PATH` in `.env`). No LLM calls or GPU are needed: the LLM outputs (`queries/rewrite-v1/`, `queries/reason-v1/`) and the dense run (`runs/rw2+reason+hist__dense-bge-small/`) are committed.

```bash
# 1. query variant (gitignored), built from the committed LLM outputs
uv run python scripts/build_queries.py --split train --variants rw2+reason+hist
uv run python scripts/build_queries.py --split dev --variants rw2+reason+hist
# 2. BM25 runs -> runs/rw2+reason+hist/{train,dev}/ (macro nDCG@10 0.5007 / 0.4779)
uv run python scripts/eval_queries.py queries/rw2+reason+hist/train
uv run python scripts/eval_queries.py queries/rw2+reason+hist/dev
# 3. fused runs from BM25 + the committed dense run (macro 0.5302 / 0.5035)
uv run python scripts/fuse_runs.py rw2+reason+hist rw2+reason+hist__dense-bge-small --split train --k 1 --w 0.6 --name rw2+reason+hist__rrf-bm25-dense-k1-w0.6
uv run python scripts/fuse_runs.py rw2+reason+hist rw2+reason+hist__dense-bge-small --split dev --k 1 --w 0.6 --name rw2+reason+hist__rrf-bm25-dense-k1-w0.6
# 4. top-5 passages for 2c -> runs/rw2+reason+hist__rrf-bm25-dense-k1-w0.6/dev/top5.jsonl (858 turns)
uv run python scripts/export_top5.py rw2+reason+hist__rrf-bm25-dense-k1-w0.6 --split dev
```

Tested 2026-11-26 in a fresh clone with these exact commands (about 2.5 minutes): the query files, BM25 runs, fused runs, `top5.jsonl` and result CSVs were byte-identical to the ones behind this summary. This section was added by hand; re-running `compare_runs.py --prefix final_fusion_dev` rewrites the file without it.
