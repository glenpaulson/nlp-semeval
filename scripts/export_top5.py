"""
Export the top 5 passages of a run, with their text, for the 2c answer generator.

Reads runs/<run>/<split>/<domain>.trec and writes runs/<run>/<split>/top5.jsonl
with one line per turn:
    {"domain": ..., "topic_id": ..., "passages": [{"rank": 1, "doc_id": ..., "score": ..., "text": ...}, ...]}

topic_id is only unique inside a domain, so every line also has the domain.

Example:
    uv run python scripts/export_top5.py rw2+reason+hist --split dev
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import DOMAINS, SPLITS, iter_corpus, load_turns

TOP_N = 5


def read_top_docs(trec_file, n):
    """Return {topic_id: [(rank, doc_id, score), ...]} with the first n docs of each topic."""
    top = {}
    with open(trec_file, encoding="utf-8") as f:
        for line in f:
            topic_id, _, doc_id, rank, score, _ = line.split()
            if int(rank) <= n:
                top.setdefault(topic_id, []).append((int(rank), doc_id, float(score)))
    for topic_id in top:
        top[topic_id].sort()  # by rank, just in case the file is not in order
    return top


def main():
    parser = argparse.ArgumentParser(description="Export the top-5 passages of a run as JSONL.")
    parser.add_argument("run", help="run name, i.e. a folder under runs/")
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "runs" / args.run / args.split
    out_file = run_dir / "top5.jsonl"

    n_turns = 0
    with open(out_file, "w", encoding="utf-8", newline="\n") as out:
        for domain in DOMAINS:
            top = read_top_docs(run_dir / f"{domain}.trec", TOP_N)

            # only keep the text of documents we actually need (the corpus is big)
            needed = {doc_id for docs in top.values() for _, doc_id, _ in docs}
            text = {doc_id: content for doc_id, content in iter_corpus(domain) if doc_id in needed}

            # write the turns in the same order as the benchmark file
            for turn in load_turns(args.split, domain):
                passages = []
                for rank, doc_id, score in top.get(turn.topic_id, []):
                    passages.append({"rank": rank, "doc_id": doc_id, "score": score, "text": text[doc_id]})
                if len(passages) < TOP_N:
                    print(f"warning: {domain} {turn.topic_id} has only {len(passages)} passages")
                line = {"domain": domain, "topic_id": turn.topic_id, "passages": passages}
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
                n_turns += 1
            print(f"{domain:<20} {len(top)} turns")

    print(f"wrote {n_turns} turns to {out_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
