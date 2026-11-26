# Pipeline validation: RETECO sub-track 2a

Read-only review of branch `rahul/dense-run` (commit 4a29762, includes the merged reranker from PR #7) plus the uncommitted dense + fusion work, 2026-11-26. No LLM calls were made. The reranker's dev run could not be checked end to end because its run files and LLM scores are only on Himanshu's machine; he is sharing them.

## Findings (most severe first)

| # | Severity | Finding | Action | Status |
|---|---|---|---|---|
| 1 | High | **Doc IDs leak gold labels.** In drones, hardware, law, medicalsciences and politics, 98-99% of doc IDs contain `neg` and no gold ID does; gold IDs are `<domain>_<conv_id>_doc_N`, and the conversation ID is part of the topic ID. An ID-only rule finds 100% of a turn's gold documents (39-55% precision). Our retrieval and reranking never use IDs (the reranker shows passages by number), so reported scores are clean. | **2c must keep doc IDs in files but never show them in LLM prompts**: `top5.jsonl` keeps `doc_id`, and the 2c generator must refer to passages by number. Add a check that no prompt contains a doc ID. Consider asking the organisers whether the hidden test uses the same naming. | Open |
| 2 | High | **No test-time path.** `SPLITS = ("train", "dev")` in `src/data.py`, and every retrieval script (`eval_queries`, `dense_retrieval`, `fuse_runs`, `run_rerank`) loads qrels and scores before finishing, so the hidden test (~200 turns, no qrels) cannot be run. Topic IDs repeat across domains, so a single submission file needs domain-aware IDs or one file per domain (official format not yet released). | Add a `test` split and a no-qrels mode that only writes runs. | Open |
| 3 | Medium | **Best result not reproducible from the repo.** The reranker's 0.5217 dev run depends on LLM scores that exist only in a local cache; Haiku runs at its default temperature, so a re-run (about $1.20) gives different scores. | Commit the per-turn LLM scores (JSON) or the run files. | Open (files requested) |
| 4 | Medium | **Dense truncation drops the note.** bge-small reads 512 tokens and the reason note comes last in `rw2+reason+hist`: 19% of dev queries exceed the limit, the note is cut entirely in 73 dev turns (9%) and partly in 90 more (train: 10% entirely). | For dense, put the note before the history or cut the oldest history first; re-tune on train only. | Open |
| 5 | Medium | **Reranker hides API failures.** `run_rerank.py` treats every exception (auth, exhausted retries) as "keep BM25 order", so a failed run looks valid; only `turns_without_usable_scores` in the report shows it. | Re-raise non-`LLMError` exceptions, as `run_rewrite.py` does. | Open |
| 6 | Medium | **Fused-run ties broken by doc ID.** pytrec_eval compares scores at about float32 precision, so the 1e-10 tie-break in `fuse_runs.py` was invisible and 22 of 858 fused dev turns were ordered by doc ID (which, per finding 1, is not neutral). | Fused runs now write rank-derived scores (1000 - rank); pytrec_eval agrees with the official rank-based scorer on every turn. | **Fixed**: train 0.5305 -> 0.5302, dev 0.5030 -> 0.5035; selection unchanged (margin 0.0027 > 0.002) |
| 7 | Low | BM25 tuning used k1=0.9, b=0.6 on dev, the train-best for turn+history; for rw2+reason+hist the train-best is k1=0.6, b=0.75 (0.5063 vs 0.5061), and the rule is not written down (dev -0.003, p=0.45, so keeping untuned BM25 stands). The dense and BM25 caches never invalidate (stale if data or `MAX_TOKENS` change), and fp16 dense vectors cause exact score ties (about 90 dev turns). The 220-turn reranker sample report shows $0 (re-run from cache), so its original cost is unrecorded; recorded reranker spend is at least $2.02. | Document the BM25 rule; key caches by model, settings and data; keep cost reports from first runs. | Open |

## Verified

- `check_data.py` passes; `Turn` carries only the six allowed fields; the rewrite, reason and rerank prompts use only the query and history (plus passage text for the reranker).
- `uv lock --check` passes; `requirements.txt` has the same 171 pinned versions as `uv.lock` (only marker spelling differs, from a different uv version).
- All 23 local train/dev run files: every turn present, 100 unique valid doc IDs, ranks 1-100, scores never increasing; every `results/<run>_<split>.csv` reproduces from its run files within 5e-7.
- `tune_bm25.py` matches `eval_queries.py` exactly at k1=0.9, b=0.4; the official baselines reproduce (dev 0.1827 / 0.4379, train 0.1837 / 0.4539).
- pytrec_eval and the official rank-based scorer differ by at most 0.0009 macro on single runs (tie handling only) and not at all on fused runs.
- Every decision was made on train and run on dev once: rewrite and reason prompts, query variants, dense variant, fusion setting, reranker alpha, BM25 k1/b.
- Repeated topic IDs across domains are handled per domain in every script.
- Note: `queries/*+hist/` is gitignored; only `queries/rewrite-v1/` and `queries/reason-v1/` are tracked, and the built variants are regenerated with `scripts/build_queries.py`.
