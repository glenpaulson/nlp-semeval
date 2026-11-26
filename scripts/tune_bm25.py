"""
Tune the BM25 parameters k1 and b.

The official baseline uses k1=0.9 and b=0.4. This script tries a grid of other
values and reports the macro nDCG@10 for each one.

The BM25 formula is the same one eval_queries.py uses (gensim's LuceneBM25Model),
but instead of transforming one document at a time we compute the weights for the
whole corpus at once with a sparse matrix. That makes each (k1, b) setting take
a few seconds instead of minutes. At k1=0.9, b=0.4 the rankings are identical
to eval_queries.py.

Rule: pick k1 and b on train, then run dev only once with the chosen values.

Examples:
    # grid search on train
    uv run python scripts/tune_bm25.py queries/rw2+reason+hist/train --baseline turn+history --split train

    # one run on dev with the chosen values (also saves the run files)
    uv run python scripts/tune_bm25.py queries/rw2+reason+hist/dev --k1 0.9 --b 0.6 --save-run
"""

import argparse
import csv
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# reuse the helpers from eval_queries.py (it also loads JAVA_HOME / JVM_PATH from .env)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_queries import BASELINE_NAMES, REPO_ROOT, TOP_K, baseline_queries, mean_ndcg10, read_queries, write_run

from src.data import DOMAINS, SPLITS, iter_corpus, load_qrels

K1_VALUES = [0.4, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0]
B_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]

# Analyzing the documents with Lucene is the slow part, so we save the result here.
# Delete this folder if the data ever changes.
CACHE_DIR = REPO_ROOT / "cache" / "bm25"


def make_count_matrix(token_lists, dictionary):
    """Turn a list of token lists into a sparse (rows x vocabulary) matrix of term counts."""
    indptr, indices, counts = [0], [], []
    for tokens in token_lists:
        bow = dictionary.doc2bow(tokens)  # sorted (term_id, count) pairs, unknown words are dropped
        for term_id, count in bow:
            indices.append(term_id)
            counts.append(count)
        indptr.append(len(indices))
    return indptr, indices, counts


def load_corpus(domain, analyzer):
    """Analyze a domain's documents once and cache the doc ids, dictionary and term counts."""
    from gensim.corpora import Dictionary

    cache_file = CACHE_DIR / f"{domain}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    doc_ids, tokens = [], []
    for doc_id, content in iter_corpus(domain):
        doc_ids.append(doc_id)
        tokens.append(analyzer.analyze(content))

    # Same dictionary as OfficialBM25 builds, so the term ids (and summation order) are the same
    dictionary = Dictionary(tokens)
    indptr, indices, counts = make_count_matrix(tokens, dictionary)
    corpus = {
        "doc_ids": doc_ids,
        "dictionary": dictionary,
        "indptr": np.array(indptr, dtype=np.int64),
        "indices": np.array(indices, dtype=np.int32),
        "counts": np.array(counts, dtype=np.int64),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(corpus, f)
    return corpus


def bm25_matrix(indptr, indices, counts, num_terms, idf, avgdl, k1, b):
    """
    Apply gensim's LuceneBM25Model weighting to every row of a count matrix:
        weight = idf * tf / (tf + k1 * (1 - b + b * length / avgdl))
    The starter kit applies this to the queries as well as the documents.
    """
    tf = counts.astype(np.float64)
    # length of each row = number of tokens in that document/query (running total trick)
    running_total = np.concatenate([[0], np.cumsum(counts)])
    row_lengths = running_total[indptr[1:]] - running_total[indptr[:-1]]
    length_per_entry = np.repeat(row_lengths.astype(np.float64), np.diff(indptr))

    weights = idf[indices] * (tf / (tf + k1 * (1 - b + b * length_per_entry / avgdl)))
    # gensim's SparseMatrixSimilarity stores everything as float32
    return sp.csr_matrix((weights.astype(np.float32), indices, indptr), shape=(len(indptr) - 1, num_terms))


def top_k_indices(scores, k):
    """Indices of the k best scores. Ties keep the document order, like Python's stable sort in the starter kit."""
    negative = -scores
    kth_value = np.partition(negative, k - 1)[k - 1]
    candidates = np.flatnonzero(negative <= kth_value)  # everything at least as good as the k-th score
    order = np.argsort(negative[candidates], kind="stable")
    return candidates[order][:k]


def search(doc_matrix, query_matrix, doc_ids, qids):
    """Score all queries against all documents and keep the top 100 for each query."""
    ranking = {}
    # do 64 queries at a time so the dense score matrix stays small enough for memory
    for start in range(0, len(qids), 64):
        q_chunk = query_matrix[start : start + 64].T.tocsc()
        scores = (doc_matrix * q_chunk).toarray().T  # shape: (queries, documents)
        for i, row in enumerate(scores):
            best = top_k_indices(row, TOP_K)
            ranking[qids[start + i]] = [(doc_ids[j], float(row[j])) for j in best]
    return ranking


def main():
    parser = argparse.ArgumentParser(description="Grid search over BM25 k1 and b.")
    parser.add_argument("queries", nargs="*", type=Path, help="query folders like queries/<variant>/<split>")
    parser.add_argument("--baseline", nargs="+", default=[], choices=sorted(BASELINE_NAMES))
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--k1", nargs="+", type=float, default=K1_VALUES)
    parser.add_argument("--b", nargs="+", type=float, default=B_VALUES)
    parser.add_argument("--save-run", action="store_true", help="save the run files (only with one k1 and one b)")
    parser.add_argument("--run-name", help="name for the saved run (only with one query source)")
    args = parser.parse_args()

    if not args.queries and not args.baseline:
        parser.error("give a queries folder and/or --baseline")
    if args.save_run and (len(args.k1) > 1 or len(args.b) > 1):
        parser.error("--save-run only works with one --k1 and one --b value")

    # each query source is (name, folder or None, baseline kind or None)
    sources = [(folder.parent.name, folder, None) for folder in args.queries]
    sources += [(BASELINE_NAMES[kind], None, kind) for kind in args.baseline]
    if args.run_name and len(sources) > 1:
        parser.error("--run-name only works with one query source")

    settings = [(k1, b) for k1 in args.k1 for b in args.b]
    qrels = load_qrels(args.split)

    from pyserini import analysis
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())

    # results[source_name][(k1, b)][domain] = nDCG@10
    results = {name: {s: {} for s in settings} for name, _, _ in sources}
    saved_rankings = {name: {} for name, _, _ in sources}

    for domain in DOMAINS:
        start_time = time.time()
        corpus = load_corpus(domain, analyzer)
        dictionary = corpus["dictionary"]
        num_docs, num_terms = len(corpus["doc_ids"]), len(dictionary)

        # these two only depend on the corpus, not on k1 or b
        avgdl = sum(dictionary.cfs.values()) / dictionary.num_docs
        # math.log (not np.log) so the numbers are exactly the same as gensim's
        idf = np.array([math.log(num_docs + 1.0) - math.log(dictionary.dfs[t] + 0.5) for t in range(num_terms)])

        # analyze the queries of every source once
        query_counts = {}
        for name, folder, kind in sources:
            if kind:
                queries = baseline_queries(args.split, domain, kind)
            else:
                queries = read_queries(folder / f"{domain}.tsv")
            qids = list(queries)
            tokens = [analyzer.analyze(queries[q]) for q in qids]
            indptr, indices, counts = make_count_matrix(tokens, dictionary)
            query_counts[name] = (qids, np.array(indptr), np.array(indices, dtype=np.int32), np.array(counts))

        for k1, b in settings:
            doc_matrix = bm25_matrix(corpus["indptr"], corpus["indices"], corpus["counts"],
                                     num_terms, idf, avgdl, k1, b)
            for name, (qids, indptr, indices, counts) in query_counts.items():
                query_matrix = bm25_matrix(indptr, indices, counts, num_terms, idf, avgdl, k1, b)
                ranking = search(doc_matrix, query_matrix, corpus["doc_ids"], qids)
                run = {qid: dict(docs) for qid, docs in ranking.items()}
                results[name][(k1, b)][domain] = mean_ndcg10(run, qrels[domain])
                if args.save_run:
                    saved_rankings[name][domain] = ranking

        print(f"{domain:<20} {num_docs:>7} docs  {len(settings)} settings  {time.time() - start_time:.1f}s")

    # report the results for each query source
    for name, _, _ in sources:
        macro = {s: np.mean(list(results[name][s].values())) for s in settings}
        ranked = sorted(settings, key=lambda s: macro[s], reverse=True)
        best = ranked[0]

        print(f"\n{name} on {args.split}:")
        print("   k1     b   macro")
        for k1, b in ranked[:10]:
            note = "  <- starter kit" if (k1, b) == (0.9, 0.4) else ""
            print(f"{k1:>5} {b:>5}  {macro[(k1, b)]:.4f}{note}")

        if (0.9, 0.4) in macro and len(settings) > 1:
            print(f"best is k1={best[0]}, b={best[1]}: {macro[best] - macro[(0.9, 0.4)]:+.4f} vs the starter kit")

        # full grid as a csv, one row per setting, best first
        if len(settings) > 1:
            grid_file = REPO_ROOT / "results" / f"bm25-grid_{name}_{args.split}.csv"
            with open(grid_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["k1", "b", *DOMAINS, "macro_avg (11 domains)"])
                for s in ranked:
                    writer.writerow([s[0], s[1], *[round(results[name][s][d], 6) for d in DOMAINS], round(macro[s], 6)])
            print(f"wrote {grid_file.relative_to(REPO_ROOT)}")

        # save the run in the same format as eval_queries.py, so compare_runs.py can read it
        if args.save_run:
            run_name = args.run_name or f"{name}__bm25-k{best[0]:g}-b{best[1]:g}"
            rows = []
            for domain in DOMAINS:
                write_run(REPO_ROOT / "runs" / run_name / args.split / f"{domain}.trec",
                          saved_rankings[name][domain], run_name)
                missing = len(set(qrels[domain]) - set(saved_rankings[name][domain]))
                rows.append({"domain": domain, "topics": len(qrels[domain]), "missing_topics": missing,
                             "ndcg_cut_10": round(results[name][best][domain], 6)})
            rows.append({"domain": "macro_avg (11 domains)", "topics": sum(r["topics"] for r in rows),
                         "missing_topics": sum(r["missing_topics"] for r in rows),
                         "ndcg_cut_10": round(macro[best], 6)})
            csv_file = REPO_ROOT / "results" / f"{run_name}_{args.split}.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote runs/{run_name}/{args.split}/ and {csv_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
