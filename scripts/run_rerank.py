"""LLM-rerank the top 30 of a BM25 run (one listwise call per turn), fuse with BM25 rank, and score nDCG@10.

Needs runs/<base-run>/<split>/<domain>.trec from scripts/eval_queries.py. Writes runs/<run>-a<alpha>/<split>/ and
results/<run>_<split>[_sample<N>].csv. On dev, pass exactly one --alphas value, chosen on train (dev is run once).
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_queries import mean_ndcg10, read_run, write_run  # noqa: E402

from src.data import DOMAINS, SPLITS, iter_corpus, load_qrels, load_turns  # noqa: E402
from src.llm import LLMError, complete_many, resolve_model, usage_summary  # noqa: E402
from src.rerank import TOP_K, build_messages, fuse, parse_scores  # noqa: E402

PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per million input / output tokens (Claude Haiku 5.5)
LLM_KWARGS = {"reasoning_effort": "low", "max_tokens": 1024}
SAMPLE_SEED = 2027


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, help="BM25 run name under runs/, e.g. rw2+reason+hist")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 0.9, 0.8, 0.7, 0.5],
                        help="LLM weight in the fusion; 1 = pure LLM order, 0 = BM25 order")
    parser.add_argument("--sample", type=int, help="score a random sample of this many turns (seed 2027), for cheap tuning")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, help="default: all 11")
    parser.add_argument("--variant", default="rerank-v1", help="prompt template prompts/<variant>.txt")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--run-name", help="default: <base-run>+<variant>")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--refresh", action="store_true",
                        help="ignore cached LLM replies and call the API again (overwrites those cache entries)")
    parser.add_argument("--reuse-scores", action="store_true",
                        help="read the saved per-turn LLM scores (results/rerank_scores/) instead of calling the LLM, "
                             "so a committed result can be rebuilt exactly with no API key or cost")
    return parser.parse_args()


def scores_path(run_name: str, split: str, domain: str) -> Path:
    return REPO_ROOT / "results" / "rerank_scores" / run_name / split / f"{domain}.json"


def save_scores(path: Path, meta: dict, scores: dict[str, dict[str, float]]) -> None:
    """Write {topic_id: {doc_id: llm_score}} for the reranked head of each turn, with the settings that produced it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "scores": scores}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        sys.exit(f"missing saved scores {path.relative_to(REPO_ROOT)}; run once without --reuse-scores")
    return json.loads(path.read_text(encoding="utf-8"))["scores"]


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return round(input_tokens * PRICE_IN / 1e6 + output_tokens * PRICE_OUT / 1e6, 6)


def main() -> int:
    args = parse_args()
    if not 0 <= min(args.alphas) <= max(args.alphas) <= 1:
        sys.exit("--alphas must be in [0, 1]")
    if args.split == "dev" and len(args.alphas) != 1:
        sys.exit("dev is run once: pass exactly one --alphas value, chosen on train")
    domains = args.domains or list(DOMAINS)
    run_name = args.run_name or f"{args.base_run}+{args.variant}"
    model = resolve_model()
    qrels_by_domain = load_qrels(args.split, domains)

    chosen: set[tuple[str, str]] | None = None
    if args.sample:
        everyone = [(t.domain, t.topic_id) for t in load_turns(args.split, domains)]
        chosen = set(random.Random(SAMPLE_SEED).sample(everyone, min(args.sample, len(everyone))))
    suffix = f"_sample{args.sample}" if args.sample else ""

    before = usage_summary()
    per_alpha: dict[float, dict[str, float]] = {a: {} for a in args.alphas}
    bm25_scores: dict[str, float] = {}
    topics_per_domain: dict[str, int] = {}
    unparsed = incomplete = 0
    for domain in domains:
        start = time.time()
        base_path = REPO_ROOT / "runs" / args.base_run / args.split / f"{domain}.trec"
        if not base_path.is_file():
            sys.exit(f"missing base run {base_path}; run scripts/eval_queries.py first")
        base = read_run(base_path)
        ranked = {q: sorted(d.items(), key=lambda x: x[1], reverse=True) for q, d in base.items()}
        turns = [t for t in load_turns(args.split, domain) if chosen is None or (domain, t.topic_id) in chosen]
        turns = [t for t in turns if t.topic_id in ranked]
        if not turns:  # a small --sample can leave a domain empty
            print(f"{domain:<20} no sampled turns, skipped")
            continue
        qrels = {q: r for q, r in qrels_by_domain[domain].items() if q in {t.topic_id for t in turns}}
        topics_per_domain[domain] = len(qrels)

        head = {t.topic_id: [d for d, _ in ranked[t.topic_id][: args.top_k]] for t in turns}
        scores = {}  # topic_id -> {1-based BM25 rank in the head: llm score}
        if args.reuse_scores:
            saved = load_scores(scores_path(run_name, args.split, domain))
            if missing := [t.topic_id for t in turns if t.topic_id not in saved]:
                sys.exit(f"{domain}: saved scores miss {len(missing)} turns, e.g. {missing[:3]}")
            for t in turns:
                scores[t.topic_id] = {r: saved[t.topic_id][d] for r, d in enumerate(head[t.topic_id], 1) if d in saved[t.topic_id]}
        else:
            need = {d for ds in head.values() for d in ds}
            content = {doc_id: text for doc_id, text in iter_corpus(domain) if doc_id in need}
            messages = [build_messages(t, [content[d] for d in head[t.topic_id]], args.variant) for t in turns]
            replies = complete_many(messages, concurrency=args.concurrency, return_exceptions=True, refresh=args.refresh, **LLM_KWARGS)
            for t, reply in zip(turns, replies):
                if isinstance(reply, LLMError):  # a refused or empty reply: keep BM25 order for this turn, but count it
                    print(f"  {t.topic_id}: LLM gave no usable reply ({reply}); keeping BM25 order")
                    scores[t.topic_id] = {}
                elif isinstance(reply, Exception):  # auth, exhausted retries, bugs: stop instead of hiding it in the score
                    raise reply
                else:
                    scores[t.topic_id] = parse_scores(reply, len(head[t.topic_id]))
            meta = {"base_run": args.base_run, "split": args.split, "domain": domain, "model": model,
                    "variant": args.variant, "top_k": args.top_k, "llm_kwargs": LLM_KWARGS}
            save_scores(
                scores_path(run_name, args.split, domain), meta,
                {q: {head[q][r - 1]: s for r, s in sc.items()} for q, sc in scores.items()},
            )
        for t in turns:
            n = len(head[t.topic_id])
            unparsed += not scores[t.topic_id]
            incomplete += bool(scores[t.topic_id]) and len(scores[t.topic_id]) < n

        bm25_scores[domain] = mean_ndcg10({t.topic_id: dict(ranked[t.topic_id]) for t in turns}, qrels)
        line = f"{domain:<20} BM25 {bm25_scores[domain]:.4f}"
        for alpha in args.alphas:
            fused = {t.topic_id: fuse(ranked[t.topic_id], scores[t.topic_id], alpha, args.top_k) for t in turns}
            path = REPO_ROOT / "runs" / f"{run_name}-a{alpha:g}" / args.split / f"{domain}.trec"
            write_run(path, fused, f"{run_name}-a{alpha:g}")
            per_alpha[alpha][domain] = mean_ndcg10(read_run(path), qrels)
            line += f"  a={alpha:g} {per_alpha[alpha][domain]:.4f}"
        print(f"{line}  turns {len(qrels):>4}  {time.time() - start:5.1f}s", flush=True)

    scored = [d for d in domains if d in bm25_scores]
    macro_bm25 = sum(bm25_scores.values()) / len(scored)
    rows = [{"setting": "bm25 (no rerank)", **{d: round(bm25_scores[d], 6) for d in scored}, "macro": round(macro_bm25, 6)}]
    for alpha in args.alphas:
        macro = sum(per_alpha[alpha].values()) / len(scored)
        rows.append({"setting": f"alpha={alpha:g}", **{d: round(per_alpha[alpha][d], 6) for d in scored}, "macro": round(macro, 6)})
        print(f"alpha={alpha:g}: macro nDCG@10 {macro:.4f} (BM25 {macro_bm25:.4f}, {macro - macro_bm25:+.4f})")

    used = {k: usage_summary()[k] - before[k] for k in before}
    cost = cost_usd(used["input_tokens"], used["output_tokens"])
    out = REPO_ROOT / "results" / f"{run_name}_{args.split}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "run": run_name, "base_run": args.base_run, "split": args.split, "model": model, "variant": args.variant,
        "top_k": args.top_k, "sample": args.sample, "sample_seed": SAMPLE_SEED if args.sample else None,
        "topics_per_domain": topics_per_domain, "llm_kwargs": LLM_KWARGS, "alphas": args.alphas,
        "turns_without_usable_scores": unparsed, "turns_with_partial_scores": incomplete,
        "usage": used, "cost_usd": cost,
        "prices_usd_per_million_tokens": {"input": PRICE_IN, "output": PRICE_OUT},
        "note": "tokens and cost count API calls made in this run; cache hits cost nothing",
    }
    if args.reuse_scores:  # a rebuild makes no API calls; keep the cost report from the run that did
        print("rebuilt from saved scores: the original LLM report was left unchanged")
    else:
        (REPO_ROOT / "results" / f"{run_name}_{args.split}{suffix}_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(f"unusable replies {unparsed}, partial {incomplete}; {used}; cost ${cost:.4f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
