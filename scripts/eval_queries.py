"""Retrieve with the official RETECO starter-kit BM25 and score nDCG@10 per domain plus the macro average."""

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # JAVA_HOME / JVM_PATH from .env (shell values win), before pyserini starts the JVM

import pytrec_eval  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data import DOMAINS, SPLITS, iter_corpus, load_qrels, load_turns  # noqa: E402
from src.queries import history_query, read_tsv  # noqa: E402

TOP_K = 100
BASELINE_NAMES = {"turn": "baseline-turn", "turn+history": "baseline-turn-history"}


class OfficialBM25:
    """Starter-kit BM25 (official_baseline.py): Lucene analyzer + gensim LuceneBM25Model, k1=0.9, b=0.4."""

    def __init__(self, domain: str):
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity
        from pyserini import analysis

        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        self.doc_ids, corpus = [], []
        for doc_id, content in tqdm(iter_corpus(domain), desc=f"  index {domain}", unit="doc", leave=False):
            self.doc_ids.append(doc_id)
            corpus.append(self.analyzer.analyze(content))
        self.dictionary = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus))]
        self.index = SparseMatrixSimilarity(
            bm25_corpus,
            num_docs=len(corpus),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs, ordered exactly as the starter kit orders them."""
        sims = self.index[self.model[self.dictionary.doc2bow(self.analyzer.analyze(query))]].tolist()
        return sorted(zip(self.doc_ids, sims), key=lambda x: x[1], reverse=True)[:k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("queries", nargs="?", type=Path, help="folder queries/<variant>/<split> with <domain>.tsv files")
    source.add_argument("--baseline", choices=sorted(BASELINE_NAMES), help="queries built from load_turns")
    source.add_argument("--oracle", action="store_true", help="sanity check: gold docs ranked first (uses qrels)")
    parser.add_argument("--split", choices=SPLITS, help="default: the queries folder name, else dev")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, help="default: all 11 domains")
    parser.add_argument("--run-name", help="default: the variant folder name, baseline-<kind> or oracle")
    return parser.parse_args()


def resolve_source(args: argparse.Namespace) -> tuple[str, str]:
    """Return (run_name, split) for the chosen query source."""
    if args.queries is not None:
        folder = args.queries.resolve()
        if not folder.is_dir():
            sys.exit(f"queries folder not found: {folder}")
        split = args.split or (folder.name if folder.name in SPLITS else None)
        if split is None:
            sys.exit(f"cannot infer the split from {folder.name!r}; pass --split")
        name = args.run_name or folder.parent.name
    else:
        split = args.split or "dev"
        name = args.run_name or ("oracle" if args.oracle else BASELINE_NAMES[args.baseline])
    if not name or any(c.isspace() for c in name):
        sys.exit(f"run name must be non-empty with no whitespace, got {name!r}")
    return name, split


def baseline_queries(split: str, domain: str, kind: str) -> dict[str, str]:
    """Return {topic_id: query}: the current turn, or turn plus history joined as the starter kit does."""
    turns = load_turns(split, domain)
    if kind == "turn+history":
        return {t.topic_id: history_query(t, [t.query]) for t in turns}
    return {t.topic_id: t.query for t in turns}


def read_queries(path: Path) -> dict[str, str]:
    """Return {topic_id: query} from a topic_id<TAB>query file, exiting with a message if it is missing or malformed."""
    if not path.is_file():
        sys.exit(f"missing query file: {path}")
    try:
        return read_tsv(path)
    except ValueError as error:
        sys.exit(str(error))


def oracle_ranking(qrels: dict[str, dict[str, int]]) -> dict[str, list[tuple[str, float]]]:
    """Return a ranking with each topic's relevant docs first, for testing the scoring path."""
    ranking = {}
    for qid, docs in qrels.items():
        gold = sorted(d for d, rel in docs.items() if rel > 0)
        ranking[qid] = [(d, float(len(gold) - i)) for i, d in enumerate(gold)]
    return ranking


def write_run(path: Path, ranking: dict[str, list[tuple[str, float]]], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for qid, docs in ranking.items():
            for rank, (doc_id, score) in enumerate(docs, 1):
                fh.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score!r}\t{tag}\n")


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        qid, _q0, doc_id, _rank, score, _tag = line.split()
        run.setdefault(qid, {})[doc_id] = float(score)
    return run


def mean_ndcg10(run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]) -> float:
    """Return mean nDCG@10 over all qrels topics; a topic missing from the run scores 0."""
    per_topic = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    return sum(per_topic.get(q, {}).get("ndcg_cut_10", 0.0) for q in qrels) / len(qrels)


def main() -> int:
    args = parse_args()
    run_name, split = resolve_source(args)
    domains = args.domains or list(DOMAINS)
    qrels_by_domain = load_qrels(split, domains)
    rows = []
    for domain in domains:
        start = time.time()
        qrels = qrels_by_domain[domain]
        if args.oracle:
            ranking = oracle_ranking(qrels)
        else:
            if args.baseline:
                queries = baseline_queries(split, domain, args.baseline)
            else:
                queries = read_queries(args.queries / f"{domain}.tsv")
            known = {t.topic_id for t in load_turns(split, domain)}
            if unknown := sorted(set(queries) - known):
                sys.exit(f"{domain}: {len(unknown)} topic_ids not in {split} turns, e.g. {unknown[:3]}")
            bm25 = OfficialBM25(domain)
            ranking = {
                qid: bm25.search(q) for qid, q in tqdm(queries.items(), desc=f"  search {domain}", leave=False)
            }
        path = REPO_ROOT / "runs" / run_name / split / f"{domain}.trec"
        write_run(path, ranking, run_name)
        score = mean_ndcg10(read_run(path), qrels)
        missing = len(set(qrels) - set(ranking))
        rows.append({"domain": domain, "topics": len(qrels), "missing_topics": missing, "ndcg_cut_10": round(score, 6)})
        note = f"  ({missing} topics missing, scored 0)" if missing else ""
        print(f"{domain:<20} nDCG@10 {score:.4f}  topics {len(qrels):>4}  {time.time() - start:5.1f}s{note}", flush=True)

    macro = sum(r["ndcg_cut_10"] for r in rows) / len(rows)
    rows.append({
        "domain": f"macro_avg ({len(domains)} domains)",
        "topics": sum(r["topics"] for r in rows),
        "missing_topics": sum(r["missing_topics"] for r in rows),
        "ndcg_cut_10": round(macro, 6),
    })
    out = REPO_ROOT / "results" / f"{run_name}_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'MACRO AVG':<20} nDCG@10 {macro:.4f}  over {len(domains)} domains")
    print(f"wrote runs/{run_name}/{split}/ and {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
