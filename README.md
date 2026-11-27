# NLP SemEval-2027 — RETECO Sub-track 2a (Conversational Retrieval)

Team: Himanshu Jain, Rahul Hipparkar, Glen Vadakkoott

Task: SemEval-2027 Task 1, RETECO — https://datascienceuibk.github.io/RETECO/task.html
Sub-track: 2a, Conversational Retrieval (official metric: nDCG@10)

## Setup
Python 3.12, Java 21 (JDK). Data: `hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --include "track2_recor/*" --local-dir reteco_data`

## Layout
- `queries/`  — query variants (TSV: topic_id, query)
- `runs/`     — retrieval runs (TREC format)
- `results/`  — scores and ablation tables
- `src/`      — code

## Reproduce the reranker result (dev 0.5217) with no API key
The per-turn LLM scores are saved in `results/rerank_scores/`, so the reranked run rebuilds exactly and for free:
```
uv run python scripts/build_queries.py --split dev --variants rw2+reason+hist
uv run python scripts/eval_queries.py queries/rw2+reason+hist/dev            # BM25 top 100, dev 0.4779
uv run python scripts/run_rerank.py --base-run rw2+reason+hist --split dev --alphas 0.3 --reuse-scores   # dev 0.5217
```
Without `--reuse-scores` the script calls the LLM (needs `ANTHROPIC_API_KEY`, about $1.20 for dev) and rewrites the saved scores. Haiku is not deterministic, so a fresh call gives slightly different scores; use `--reuse-scores` for the reported numbers.
