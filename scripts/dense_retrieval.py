"""
Dense retrieval with a sentence embedding model (bge-small-en-v1.5).

Every document and query is turned into a vector, and documents are ranked by
cosine similarity to the query. BM25 needs the same words in query and document,
a dense model can also match passages that say the same thing in other words.

The document vectors are computed once per domain and saved in cache/dense/,
so after the first run only the queries need to be encoded.

Output is the same as eval_queries.py: runs/<run>/<split>/<domain>.trec and
results/<run>_<split>.csv, with run name <query variant>__dense-bge-small.

Examples:
    uv run python scripts/dense_retrieval.py queries/rewrite-v1/train queries/rw2+reason+hist/train --baseline turn+history --split train
    uv run python scripts/dense_retrieval.py queries/rw2+reason+hist/dev --split dev
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# reuse the helpers from eval_queries.py (query reading, run writing, nDCG@10)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_queries import BASELINE_NAMES, REPO_ROOT, TOP_K, baseline_queries, mean_ndcg10, read_queries, write_run

from src.data import DOMAINS, SPLITS, iter_corpus, load_qrels

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RUN_SUFFIX = "dense-bge-small"
# the bge model card says to put this in front of queries (not documents) for retrieval
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 512  # longer texts get cut off, this is the model's limit
BATCH_SIZE = 64

CACHE_DIR = REPO_ROOT / "cache" / "dense" / "bge-small-en-v1.5"


def get_device():
    """Return the torch device to use: cuda, else Apple GPU (mps), else cpu."""
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    from sentence_transformers import SentenceTransformer

    device = get_device()
    print(f"loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_TOKENS
    # note: I tried model.half() (fp16) but on our Quadro T1000 it was 4x SLOWER than fp32
    # (16 vs 62 docs/s), this GPU has no tensor cores. So we keep the default fp32.
    return model


def encode(model, texts):
    # normalize so that the dot product is the cosine similarity
    return model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)


def load_doc_vectors(model, domain):
    """Encode all documents of a domain, or load them from the cache if we did it before."""
    vectors_file = CACHE_DIR / f"{domain}.npy"
    ids_file = CACHE_DIR / f"{domain}_ids.json"
    if vectors_file.exists() and ids_file.exists():
        return json.loads(ids_file.read_text(encoding="utf-8")), np.load(vectors_file)

    doc_ids, texts = [], []
    for doc_id, content in iter_corpus(domain):
        doc_ids.append(doc_id)
        texts.append(content)
    print(f"  encoding {len(texts)} documents of {domain} (only the first time)")
    vectors = encode(model, texts).astype(np.float16)  # float16 to keep the cache small

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(vectors_file, vectors)
    ids_file.write_text(json.dumps(doc_ids), encoding="utf-8")
    return doc_ids, vectors


def search(doc_vectors, query_vectors, doc_ids, qids):
    """Return {qid: [(doc_id, score), ...]} with the top 100 documents for each query."""
    device = get_device()
    docs = torch.from_numpy(doc_vectors).to(device).float()
    queries = torch.from_numpy(query_vectors).to(device).float()

    ranking = {}
    for start in range(0, len(qids), 256):
        scores = queries[start : start + 256] @ docs.T  # cosine similarities
        top_scores, top_idx = torch.topk(scores, k=min(TOP_K, len(doc_ids)), dim=1)
        for i, (row_scores, row_idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist())):
            ranking[qids[start + i]] = [(doc_ids[j], s) for j, s in zip(row_idx, row_scores)]
    return ranking


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval with bge-small-en-v1.5.")
    parser.add_argument("queries", nargs="*", type=Path, help="query folders like queries/<variant>/<split>")
    parser.add_argument("--baseline", nargs="+", default=[], choices=sorted(BASELINE_NAMES))
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS),
                        help="only for quick tests, the macro average needs all 11")
    args = parser.parse_args()
    if not args.queries and not args.baseline:
        parser.error("give a queries folder and/or --baseline")

    # each query source is (name, folder or None, baseline kind or None)
    sources = [(folder.parent.name, folder, None) for folder in args.queries]
    sources += [(BASELINE_NAMES[kind], None, kind) for kind in args.baseline]

    model = load_model()
    qrels = load_qrels(args.split, args.domains)
    scores = {name: {} for name, _, _ in sources}  # scores[source][domain] = nDCG@10

    for domain in args.domains:
        start_time = time.time()
        doc_ids, doc_vectors = load_doc_vectors(model, domain)

        for name, folder, kind in sources:
            if kind:
                queries = baseline_queries(args.split, domain, kind)
            else:
                queries = read_queries(folder / f"{domain}.tsv")
            qids = list(queries)
            query_vectors = encode(model, [QUERY_PREFIX + queries[q] for q in qids])

            ranking = search(doc_vectors, query_vectors, doc_ids, qids)
            run_name = f"{name}__{RUN_SUFFIX}"
            write_run(REPO_ROOT / "runs" / run_name / args.split / f"{domain}.trec", ranking, run_name)
            run = {qid: dict(docs) for qid, docs in ranking.items()}
            scores[name][domain] = mean_ndcg10(run, qrels[domain])

        summary = "  ".join(f"{name} {scores[name][domain]:.4f}" for name, _, _ in sources)
        print(f"{domain:<20} {summary}  ({time.time() - start_time:.0f}s)", flush=True)

    # one results csv per run, same columns as eval_queries.py so compare_runs.py can use it
    print()
    for name, _, _ in sources:
        run_name = f"{name}__{RUN_SUFFIX}"
        rows = []
        for domain in args.domains:
            rows.append({"domain": domain, "topics": len(qrels[domain]), "missing_topics": 0,
                         "ndcg_cut_10": round(scores[name][domain], 6)})
        macro = sum(scores[name].values()) / len(scores[name])
        rows.append({"domain": f"macro_avg ({len(args.domains)} domains)", "topics": sum(r["topics"] for r in rows),
                     "missing_topics": 0, "ndcg_cut_10": round(macro, 6)})
        csv_file = REPO_ROOT / "results" / f"{run_name}_{args.split}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{run_name:<45} macro nDCG@10 {macro:.4f}  -> {csv_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
