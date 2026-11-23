"""Build query variants from stored LLM outputs and turns (no LLM calls) as queries/<variant>/<split>/<domain>.tsv."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import DOMAINS, SPLITS, Turn, load_turns, normalize_history  # noqa: E402
from src.queries import history_query, read_tsv, validate_tsv, write_tsv  # noqa: E402
from src.rewrite import single_line  # noqa: E402


def rewrite_plus_history(turn: Turn, rewrite: str, copies: int, note: str | None = None) -> str:
    """Rewrite (repeated) + turn + history (+ note); turn 1 has no rewrite or history, so it keeps the turn (+ note)."""
    after = [note] if note else []
    if not normalize_history(turn.conversation_history):
        return history_query(turn, [turn.query], after=after)
    return history_query(turn, [rewrite] * copies + [turn.query], after=after)


def turn_history_note(turn: Turn, note: str) -> str:
    """Turn + history + note, using the starter kit's +History layout with the note appended."""
    return history_query(turn, [turn.query], after=[note])


VARIANTS = {  # variant -> (source folders under queries/, builder taking the turn and one text per source)
    "rw+hist": (("rewrite-v1",), lambda turn, rw: rewrite_plus_history(turn, rw, 1)),
    "rw2+hist": (("rewrite-v1",), lambda turn, rw: rewrite_plus_history(turn, rw, 2)),
    "reason+hist": (("reason-v1",), turn_history_note),
    "rw2+reason+hist": (("rewrite-v1", "reason-v1"), lambda turn, rw, note: rewrite_plus_history(turn, rw, 2, note)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failed = False
    for variant in args.variants:
        sources, build = VARIANTS[variant]
        rows_total, words_total = 0, 0
        for domain in DOMAINS:
            turns = load_turns(args.split, domain)
            source_texts = []
            for source in sources:
                src_path = REPO_ROOT / "queries" / source / args.split / f"{domain}.tsv"
                if not src_path.is_file():
                    sys.exit(f"{variant}: missing source {src_path.relative_to(REPO_ROOT)}; run run_rewrite.py first")
                texts = read_tsv(src_path)
                if missing := [t.topic_id for t in turns if t.topic_id not in texts]:
                    sys.exit(f"{variant}/{domain}: {len(missing)} turns missing from {src_path}, e.g. {missing[:3]}")
                source_texts.append(texts)
            rows = [(t.topic_id, single_line(build(t, *(s[t.topic_id] for s in source_texts)))) for t in turns]
            out = REPO_ROOT / "queries" / variant / args.split / f"{domain}.tsv"
            write_tsv(out, rows)
            if problems := validate_tsv(out, turns):
                failed = True
                print(f"INVALID {variant}/{domain}: {'; '.join(problems)}")
            rows_total += len(rows)
            words_total += sum(len(q.split()) for _, q in rows)
        print(f"{variant:<16} from {' + '.join(sources):<22} {rows_total:>5} rows, mean {words_total / rows_total:6.1f} "
              f"words -> queries/{variant}/{args.split}/")
    print("validation:", "FAILED" if failed else "all variants and domains passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
