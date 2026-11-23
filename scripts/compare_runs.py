"""Compare runs on one split: per-domain nDCG@10, paired randomization tests vs a baseline, and a by-turn breakdown."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pytrec_eval  # noqa: E402

from src.data import DOMAINS, SPLITS, load_qrels  # noqa: E402

BUCKETS = ["T1", "T2", "T3", "T4", "T5+"]
TIE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--runs", nargs="+", required=True, help="run names under runs/<name>/<split>/, in table order")
    parser.add_argument("--baseline", required=True, help="run every other run is tested against")
    parser.add_argument("--best", required=True, help="run shown against the baseline in the by-turn breakdown")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--prefix", help="output prefix under results/ (default: final_<split>)")
    parser.add_argument("--cost-report", nargs="*", default=[], help="run_rewrite.py JSON reports to sum for LLM cost")
    parser.add_argument("--note", action="append", default=[], help="extra bullet for the summary (repeatable)")
    return parser.parse_args()


def per_topic_ndcg(run: str, split: str, qrels: dict) -> dict[tuple[str, str], float]:
    """Return nDCG@10 per (domain, topic_id) from runs/<run>/<split>/<domain>.trec; topics missing from a run score 0."""
    scores = {}
    for domain, domain_qrels in qrels.items():
        path = REPO_ROOT / "runs" / run / split / f"{domain}.trec"
        if not path.is_file():
            sys.exit(f"missing run file {path.relative_to(REPO_ROOT)}; run eval_queries.py first")
        ranking: dict[str, dict[str, float]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            qid, _q0, doc_id, _rank, score, _tag = line.split()
            ranking.setdefault(qid, {})[doc_id] = float(score)
        evaluated = pytrec_eval.RelevanceEvaluator(domain_qrels, {"ndcg_cut.10"}).evaluate(ranking)
        scores.update({(domain, q): evaluated.get(q, {}).get("ndcg_cut_10", 0.0) for q in domain_qrels})
    return scores


def randomization_p(diffs: np.ndarray, weights: np.ndarray, permutations: int, rng: np.random.Generator) -> float:
    """Two-sided paired randomization p-value for the weighted sum of per-turn differences, by random sign flips."""
    weighted = weights * diffs
    observed = abs(weighted.sum())
    hits = 0
    for start in range(0, permutations, 1000):
        signs = rng.choice((-1.0, 1.0), size=(min(1000, permutations - start), len(diffs)))
        hits += int((np.abs(signs @ weighted) >= observed - 1e-12).sum())
    return (hits + 1) / (permutations + 1)


def turn_bucket(topic_id: str) -> str:
    turn = int(topic_id.rsplit("_turn_", 1)[1])
    return "T5+" if turn >= 5 else f"T{turn}"


def macro_from_csv(run: str, split: str) -> float | None:
    path = REPO_ROOT / "results" / f"{run}_{split}.csv"
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return float(rows[-1]["ndcg_cut_10"]) if rows and rows[-1]["domain"].startswith("macro_avg") else None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    """Return markdown table lines; align has one character per column, l or r."""
    rule = ["---" if a == "l" else "---:" for a in align]
    return ["| " + " | ".join(r) + " |" for r in [header, rule, *rows]]


def main() -> int:
    args = parse_args()
    prefix = args.prefix or f"final_{args.split}"
    runs = list(dict.fromkeys([*args.runs, args.baseline, args.best]))
    qrels = load_qrels(args.split)
    keys = [(d, q) for d in DOMAINS for q in qrels[d]]
    scores = {run: per_topic_ndcg(run, args.split, qrels) for run in runs}
    n_by_domain = {d: len(qrels[d]) for d in DOMAINS}
    weights = np.array([1.0 / (len(DOMAINS) * n_by_domain[d]) for d, _ in keys])

    per_domain = {r: {d: float(np.mean([scores[r][(d, q)] for q in qrels[d]])) for d in DOMAINS} for r in runs}
    macro = {r: float(np.mean(list(per_domain[r].values()))) for r in runs}
    for r in runs:
        stored = macro_from_csv(r, args.split)
        if stored is not None and abs(stored - macro[r]) > 1e-6:
            sys.exit(f"{r}: macro {macro[r]:.6f} from run files differs from results/{r}_{args.split}.csv ({stored:.6f})")

    rng = np.random.default_rng(args.seed)
    base = np.array([scores[args.baseline][k] for k in keys])
    significance = []
    for r in args.runs:
        if r == args.baseline:
            continue
        cur = np.array([scores[r][k] for k in keys])
        diffs = cur - base
        significance.append({
            "run": r,
            "train_macro": macro_from_csv(r, "train") if args.split != "train" else None,
            "macro": round(macro[r], 6),
            "delta_vs_baseline": round(macro[r] - macro[args.baseline], 6),
            "p_value": round(randomization_p(diffs, weights, args.permutations, rng), 6),
            "wins": int((diffs > TIE).sum()),
            "ties": int((np.abs(diffs) <= TIE).sum()),
            "losses": int((diffs < -TIE).sum()),
        })

    by_turn = []
    for b in BUCKETS:
        ks = [k for k in keys if turn_bucket(k[1]) == b]
        base_b = float(np.mean([scores[args.baseline][k] for k in ks]))
        best_b = float(np.mean([scores[args.best][k] for k in ks]))
        diffs_b = np.array([scores[args.best][k] - scores[args.baseline][k] for k in ks])
        by_turn.append({
            "turn": b, "turns": len(ks), args.baseline: round(base_b, 6), args.best: round(best_b, 6),
            "delta": round(best_b - base_b, 6), "wins": int((diffs_b > TIE).sum()), "losses": int((diffs_b < -TIE).sum()),
        })

    results = REPO_ROOT / "results"
    table_rows = [{"domain": d, **{r: round(per_domain[r][d], 6) for r in args.runs}} for d in DOMAINS]
    table_rows.append({"domain": f"macro_avg ({len(DOMAINS)} domains)", **{r: round(macro[r], 6) for r in args.runs}})
    write_csv(results / f"{prefix}_table.csv", table_rows)
    write_csv(results / f"{prefix}_significance.csv", significance)
    write_csv(results / f"{prefix}_by_turn.csv", by_turn)

    costs = []
    for report_path in args.cost_report:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        costs.append((Path(report_path).name, report["variant"], report["split"], report["totals"]["cost_usd"]))

    md = [f"# Final {args.split} results: RETECO sub-track 2a", ""]
    md.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/compare_runs.py`.")
    md.append("")
    md.append("- Retrieval: official starter-kit BM25 (Pyserini Lucene analyzer, gensim `LuceneBM25Model`, k1=0.9, b=0.4), "
              "top 100 per turn; nDCG@10 via pytrec_eval, averaged per domain, then over the 11 domains.")
    md.append(f"- Significance: two-sided paired randomization test vs `{args.baseline}` over all {len(keys)} "
              f"{args.split} turns, {args.permutations:,} sign-flip permutations (seed {args.seed}); the statistic is "
              f"the macro-averaged difference (each turn weighted 1 / (11 x turns in its domain)).")
    md += [f"- {note}" for note in args.note]
    md += ["", "## Per-domain nDCG@10", ""]
    md += md_table(["domain", *args.runs], [[d, *(f"{per_domain[r][d]:.4f}" for r in args.runs)] for d in DOMAINS]
                   + [["**macro (11)**", *(f"**{macro[r]:.4f}**" for r in args.runs)]], "l" + "r" * len(args.runs))
    md += ["", f"## Paired randomization test vs `{args.baseline}`", ""]
    md += md_table(
        ["run", "train macro", f"{args.split} macro", "delta", "p-value", "turns better / same / worse"],
        [[f"`{s['run']}`", f"{s['train_macro']:.4f}" if s["train_macro"] is not None else "-", f"{s['macro']:.4f}",
          f"{s['delta_vs_baseline']:+.4f}", f"{s['p_value']:.4f}", f"{s['wins']} / {s['ties']} / {s['losses']}"]
         for s in significance], "lrrrrr")
    md += ["", f"## By turn number: `{args.baseline}` vs `{args.best}` (turns pooled over all domains)", ""]
    md += md_table(
        ["turn", "turns", f"`{args.baseline}`", f"`{args.best}`", "delta", "better / worse"],
        [[t["turn"], str(t["turns"]), f"{t[args.baseline]:.4f}", f"{t[args.best]:.4f}", f"{t['delta']:+.4f}",
          f"{t['wins']} / {t['losses']}"] for t in by_turn], "lrrrrr")
    if costs:
        md += ["", "## LLM cost (Claude Haiku 5.5, $0.10 / $0.50 per million input / output tokens)", ""]
        md += md_table(["report", "variant", "split", "cost (USD)"],
                       [[f"`{n}`", v, s, f"{c:.4f}"] for n, v, s, c in costs]
                       + [["**total**", "", "", f"**{sum(c for *_, c in costs):.4f}**"]], "lllr")
    md += ["", f"CSV files: `results/{prefix}_table.csv`, `results/{prefix}_significance.csv`, "
           f"`results/{prefix}_by_turn.csv`; per-run files `results/<run>_{args.split}.csv`.", ""]
    (results / f"{prefix}_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
