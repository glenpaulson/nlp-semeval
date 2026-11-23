"""Rewrite each turn into a standalone query and write queries/<variant>/<split>/<domain>.tsv plus a JSON report."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import DOMAINS, SPLITS, Turn, load_turns  # noqa: E402
from src.llm import LLMError, complete_many, resolve_model, usage_summary  # noqa: E402
from src.rewrite import build_messages, clean_query, single_line, template_path, unsupported_terms  # noqa: E402

PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per million input / output tokens
LLM_KWARGS = {"reasoning_effort": "low", "max_tokens": 1024}
TABLE_COLS = ["turns", "turn1_kept", "rewritten", "llm_calls", "cache_hits", "fallbacks", "flagged"]
SUM_KEYS = TABLE_COLS + ["input_tokens", "output_tokens"]


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return round(input_tokens * PRICE_IN / 1e6 + output_tokens * PRICE_OUT / 1e6, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="rewrite-v1")
    parser.add_argument("--split", default="dev", choices=SPLITS)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, help="default: all 11 domains")
    parser.add_argument("--limit", type=int, help="first N turns per domain; output goes to <split>-limit<N>/")
    return parser.parse_args()


def rewrite_domain(turns: list[Turn], variant: str) -> tuple[list[str], dict]:
    """Return one query per turn (in order) and the domain's stats, fallback reasons and flagged terms."""
    before = usage_summary()
    queries = [single_line(t.query) for t in turns]
    to_send = [(i, m) for i, t in enumerate(turns) if (m := build_messages(t, variant)) is not None]
    outputs = complete_many([m for _, m in to_send], return_exceptions=True, **LLM_KWARGS)
    fallbacks: dict[str, str] = {}
    flagged: dict[str, list[str]] = {}
    for (i, _), output in zip(to_send, outputs):
        topic = turns[i].topic_id
        if isinstance(output, LLMError):
            fallbacks[topic] = f"LLMError: {output}"
        elif isinstance(output, BaseException):
            raise output
        elif cleaned := clean_query(output):
            queries[i] = cleaned
            if terms := unsupported_terms(cleaned, turns[i]):
                flagged[topic] = terms
        else:
            fallbacks[topic] = f"empty after cleaning: {output!r}"
    after = usage_summary()
    stats = {
        "turns": len(turns),
        "turn1_kept": len(turns) - len(to_send),
        "rewritten": len(to_send) - len(fallbacks),
        "llm_calls": after["api_calls"] - before["api_calls"],
        "cache_hits": after["cache_hits"] - before["cache_hits"],
        "fallbacks": len(fallbacks),
        "flagged": len(flagged),
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "fallback_reasons": fallbacks,
        "flagged_terms": flagged,
    }
    return queries, stats


def write_tsv(path: Path, turns: list[Turn], queries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for turn, query in zip(turns, queries):
            fh.write(f"{turn.topic_id}\t{query}\n")


def validate(path: Path, turns: list[Turn]) -> list[str]:
    """Re-read a written TSV and return a list of problems (empty if valid)."""
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").split("\n")[:-1]]
    problems = []
    if bad := [n for n, row in enumerate(rows, 1) if len(row) != 2]:
        problems.append(f"{len(bad)} rows without exactly 2 fields (first: line {bad[0]})")
    ids = [row[0] for row in rows]
    if len(rows) != len(turns):
        problems.append(f"{len(rows)} rows, expected {len(turns)} turns")
    if len(set(ids)) != len(ids):
        problems.append(f"{len(ids) - len(set(ids))} duplicate topic_ids")
    if set(ids) != {t.topic_id for t in turns}:
        problems.append("topic_ids differ from load_turns")
    if empty := sum(1 for row in rows if len(row) < 2 or not row[1].strip()):
        problems.append(f"{empty} empty queries")
    return problems


def main() -> int:
    args = parse_args()
    if not template_path(args.variant).is_file():
        sys.exit(f"no template for variant {args.variant!r}: expected {template_path(args.variant)}")
    domains = args.domains or list(DOMAINS)
    split_dir = args.split if args.limit is None else f"{args.split}-limit{args.limit}"
    out_dir = REPO_ROOT / "queries" / args.variant / split_dir
    all_turns = load_turns(args.split, domains)

    per_domain: dict[str, dict] = {}
    for domain in domains:
        turns = [t for t in all_turns if t.domain == domain][: args.limit]
        queries, stats = rewrite_domain(turns, args.variant)
        path = out_dir / f"{domain}.tsv"
        write_tsv(path, turns, queries)
        problems = validate(path, turns)
        if stats["llm_calls"] + stats["cache_hits"] != stats["turns"] - stats["turn1_kept"]:
            problems.append("LLM calls + cache hits != turns sent to the LLM")
        stats.update(
            valid=not problems,
            problems=problems,
            flagged_rate=round(stats["flagged"] / stats["rewritten"], 4) if stats["rewritten"] else 0.0,
            cost_usd=cost_usd(stats["input_tokens"], stats["output_tokens"]),
        )
        per_domain[domain] = stats

    totals = {k: sum(s[k] for s in per_domain.values()) for k in SUM_KEYS}
    totals.update(
        valid=all(s["valid"] for s in per_domain.values()),
        flagged_rate=round(totals["flagged"] / totals["rewritten"], 4) if totals["rewritten"] else 0.0,
        cost_usd=cost_usd(totals["input_tokens"], totals["output_tokens"]),
    )
    report = {
        "variant": args.variant,
        "split": args.split,
        "limit": args.limit,
        "model": resolve_model(),
        "template": str(template_path(args.variant).relative_to(REPO_ROOT)),
        "llm_kwargs": LLM_KWARGS,
        "prices_usd_per_million_tokens": {"input": PRICE_IN, "output": PRICE_OUT},
        "note": "tokens and cost count API calls made in this run; cache hits cost nothing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domains": per_domain,
        "totals": totals,
    }
    report_path = REPO_ROOT / "results" / f"{args.variant}_{split_dir}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rule = "-" * (20 + 11 * len(TABLE_COLS) + 8)
    print(f"\n{'domain':<20}" + "".join(f"{c:>11}" for c in TABLE_COLS) + f"{'valid':>8}")
    print(rule)
    for domain, s in per_domain.items():
        print(f"{domain:<20}" + "".join(f"{s[c]:>11,}" for c in TABLE_COLS) + f"{'OK' if s['valid'] else 'FAIL':>8}")
    print(rule)
    print(f"{'TOTAL':<20}" + "".join(f"{totals[c]:>11,}" for c in TABLE_COLS) + f"{'OK' if totals['valid'] else 'FAIL':>8}")

    for domain, s in per_domain.items():
        for topic_id, reason in s["fallback_reasons"].items():
            print(f"fallback {domain}/{topic_id}: {reason}")
        if args.limit is not None:
            for topic_id, terms in s["flagged_terms"].items():
                print(f"flagged {domain}/{topic_id}: {', '.join(terms)}")
        for problem in s["problems"]:
            print(f"INVALID {domain}: {problem}")

    print(f"\nflagged rate: {totals['flagged']}/{totals['rewritten']} rewritten turns = {totals['flagged_rate']:.1%}")
    print("usage_summary():", usage_summary())
    print(f"estimated cost: ${totals['cost_usd']:.4f} (at ${PRICE_IN:.2f} / ${PRICE_OUT:.2f} per million input / output tokens)")
    print(f"wrote {len(per_domain)} files to {out_dir.relative_to(REPO_ROOT)}/ and {report_path.relative_to(REPO_ROOT)}")
    print("validation:", "all domains passed" if totals["valid"] else "FAILED (see INVALID lines)")
    return 0 if totals["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
