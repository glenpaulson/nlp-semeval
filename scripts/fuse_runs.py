"""Weighted Reciprocal Rank Fusion of two runs: score = w/(k + rank_a) + (1-w)/(k + rank_b), top 100 per turn."""

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_queries import REPO_ROOT, TOP_K, mean_ndcg10, read_run, write_run  # noqa: E402

from src.data import DOMAINS, SPLITS, load_qrels  # noqa: E402

RANK_SCORE_BASE = 1000  # written score = 1000 - rank: distinct integers, so pytrec_eval never breaks ties by doc ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="first run (e.g. BM25), a folder under runs/; w is its weight")
    parser.add_argument("run_b", help="second run (e.g. dense), a folder under runs/; gets weight 1 - w")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--k", type=float, nargs="+", default=[60.0], help="RRF constant(s)")
    parser.add_argument("--w", type=float, nargs="+", default=[0.5], help="weight(s) of run_a, in [0, 1]")
    parser.add_argument("--name", help="fused run name (single k and w only); default rrf-k<k>-w<w>__<run_a>__<run_b>")
    parser.add_argument("--grid-out", type=Path, help="CSV for a grid search (default results/fusion_grid_<split>.csv)")
    parser.add_argument("--append", action="store_true", help="merge the grid into an existing grid CSV")
    parser.add_argument("--tolerance", type=float, default=0.002, help="plateau rule: max macro gap to the best setting")
    return parser.parse_args()


def select_setting(rows: list[dict], tolerance: float) -> tuple[float, float, str]:
    """Return (k, w, reason): the best macro, unless a setting within tolerance has a higher neighbour average."""
    ks, ws = sorted({r["k"] for r in rows}), sorted({r["w"] for r in rows})
    macro = {(r["k"], r["w"]): r["macro"] for r in rows}

    def neighbour_avg(k: float, w: float) -> float:
        i, j = ks.index(k), ws.index(w)
        cells = [(ks[i + s], w) for s in (-1, 1) if 0 <= i + s < len(ks)]
        cells += [(k, ws[j + s]) for s in (-1, 1) if 0 <= j + s < len(ws)]
        values = [macro[c] for c in cells if c in macro]
        return sum(values) / len(values) if values else float("-inf")

    best = max(macro, key=macro.get)
    near = [c for c in macro if c != best and macro[c] >= macro[best] - tolerance and neighbour_avg(*c) > neighbour_avg(*best)]
    if not near:
        return (*best, f"highest train macro ({macro[best]:.4f}); no setting within {tolerance} of it has a higher "
                       f"neighbour average than its {neighbour_avg(*best):.4f}")
    pick = max(near, key=lambda c: (neighbour_avg(*c), macro[c]))
    return (*pick, f"plateau rule: macro {macro[pick]:.4f} is within {tolerance} of the best (k={best[0]:g}, w={best[1]:g}, "
                   f"{macro[best]:.4f}) and its neighbour average {neighbour_avg(*pick):.4f} beats {neighbour_avg(*best):.4f}")


def read_ranked(path: Path) -> dict[str, list[str]]:
    """Return {topic_id: [doc_id, ...]} in rank order, using the rank column of a TREC run file."""
    if not path.is_file():
        sys.exit(f"missing run file {path.relative_to(REPO_ROOT)}")
    rows: dict[str, list[tuple[int, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        qid, _q0, doc_id, rank, _score, _tag = line.split()
        rows.setdefault(qid, []).append((int(rank), doc_id))
    return {qid: [d for _, d in sorted(docs)] for qid, docs in rows.items()}


def fuse(a: dict[str, list[str]], b: dict[str, list[str]], k: float, w: float) -> dict[str, list[tuple[str, float]]]:
    """Return the fused top-100 per topic with rank-derived scores; RRF ties go to the better best rank, then run_a's rank."""
    fused = {}
    for qid in sorted(set(a) | set(b)):
        rank_a = {d: r for r, d in enumerate(a.get(qid, []), 1)}
        rank_b = {d: r for r, d in enumerate(b.get(qid, []), 1)}
        scored = []
        for doc in rank_a.keys() | rank_b.keys():
            ra, rb = rank_a.get(doc), rank_b.get(doc)
            score = (w / (k + ra) if ra else 0.0) + ((1 - w) / (k + rb) if rb else 0.0)
            best = min(r for r in (ra, rb) if r)
            scored.append((-score, best, ra or float("inf"), doc))
        top = sorted(scored)[:TOP_K]
        fused[qid] = [(doc, float(RANK_SCORE_BASE - rank)) for rank, (*_key, doc) in enumerate(top, 1)]
    return fused


def write_results(path: Path, per_domain: dict[str, float], qrels: dict) -> None:
    rows = [{"domain": d, "topics": len(qrels[d]), "missing_topics": 0, "ndcg_cut_10": round(s, 6)} for d, s in per_domain.items()]
    macro = sum(per_domain.values()) / len(per_domain)
    rows.append({"domain": f"macro_avg ({len(per_domain)} domains)", "topics": sum(r["topics"] for r in rows),
                 "missing_topics": 0, "ndcg_cut_10": round(macro, 6)})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if any(not 0.0 <= w <= 1.0 for w in args.w) or any(k <= 0 for k in args.k):
        sys.exit("w must be in [0, 1] and k must be > 0")
    qrels = load_qrels(args.split)
    runs = {
        d: (read_ranked(REPO_ROOT / "runs" / args.run_a / args.split / f"{d}.trec"),
            read_ranked(REPO_ROOT / "runs" / args.run_b / args.split / f"{d}.trec"))
        for d in DOMAINS
    }
    combos = list(itertools.product(args.k, args.w))

    if len(combos) == 1:
        k, w = combos[0]
        name = args.name or f"rrf-k{k:g}-w{w:g}__{args.run_a}__{args.run_b}"
        per_domain = {}
        for d in DOMAINS:
            path = REPO_ROOT / "runs" / name / args.split / f"{d}.trec"
            write_run(path, fuse(*runs[d], k, w), name)
            per_domain[d] = mean_ndcg10(read_run(path), qrels[d])
            print(f"{d:<20} nDCG@10 {per_domain[d]:.4f}")
        out = REPO_ROOT / "results" / f"{name}_{args.split}.csv"
        write_results(out, per_domain, qrels)
        print(f"{'MACRO AVG':<20} nDCG@10 {sum(per_domain.values()) / len(DOMAINS):.4f}\nwrote runs/{name}/{args.split}/ and "
              f"{out.relative_to(REPO_ROOT)}")
        return 0

    rows = []
    for k, w in combos:
        per_domain = {}
        for d in DOMAINS:
            fused = fuse(*runs[d], k, w)
            per_domain[d] = mean_ndcg10({q: dict(docs) for q, docs in fused.items()}, qrels[d])
        rows.append({"k": k, "w": w, "macro": round(sum(per_domain.values()) / len(DOMAINS), 6),
                     **{d: round(s, 6) for d, s in per_domain.items()}})
    out = args.grid_out or REPO_ROOT / "results" / f"fusion_grid_{args.split}.csv"
    if args.append and out.is_file():
        old = {(r["k"], r["w"]): r for r in ({c: float(v) for c, v in row.items()} for row in csv.DictReader(out.open()))}
        for r in rows:
            if (r["k"], r["w"]) in old and abs(old[(r["k"], r["w"])]["macro"] - r["macro"]) > 1e-6:
                print(f"warning: k={r['k']:g}, w={r['w']:g} changed from {old[(r['k'], r['w'])]['macro']:.6f} to {r['macro']:.6f}")
            old[(r["k"], r["w"])] = r
        rows = list(old.values())
    rows.sort(key=lambda r: (r["k"], r["w"]))
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ks, ws = sorted({r["k"] for r in rows}), sorted({r["w"] for r in rows})
    macro = {(r["k"], r["w"]): r["macro"] for r in rows}
    print(f"macro nDCG@10, run_a = {args.run_a} (weight w), run_b = {args.run_b} (weight 1 - w)\n")
    print(f"{'k':>6}" + "".join(f"{'w=' + format(w, 'g'):>10}" for w in ws))
    for k in ks:
        print(f"{k:>6g}" + "".join(f"{macro[(k, w)]:>10.4f}" if (k, w) in macro else f"{'-':>10}" for w in ws))
    best = max(rows, key=lambda r: r["macro"])
    k, w, reason = select_setting(rows, args.tolerance)
    print(f"\nbest cell: k={best['k']:g}, w={best['w']:g}, macro {best['macro']:.4f}")
    print(f"selected:  k={k:g}, w={w:g}, macro {macro[(k, w)]:.4f} -- {reason}\nwrote {out} ({len(rows)} settings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
