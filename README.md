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
