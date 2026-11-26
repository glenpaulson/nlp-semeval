"""LLM reranking for sub-track 2a: score a turn's top-k BM25 passages in one listwise call, then fuse with BM25 rank.

The prompt uses only Turn.query, Turn.conversation_history and passage text; passages are shown by list number, never by doc ID.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from src.data import Turn, normalize_history
from src.rewrite import EMPTY_HISTORY, PROMPTS_DIR

TOP_K = 30
DOC_MAX_WORDS = 250  # passages are short (median 38 words); this only trims the long tail
FUSION_RRF_K = 60

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@lru_cache
def load_template(variant: str = "rerank-v1") -> str:
    return (PROMPTS_DIR / f"{variant.replace('-', '_')}.txt").read_text(encoding="utf-8")


def clip(text: str, max_words: int = DOC_MAX_WORDS) -> str:
    """Collapse whitespace and cut a passage to max_words words."""
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def build_messages(turn: Turn, passages: list[str], variant: str = "rerank-v1") -> list[dict[str, str]]:
    """Return LLM messages that ask for a 0-10 score for every passage, numbered from 1."""
    history = normalize_history(turn.conversation_history).strip() or EMPTY_HISTORY
    listing = "\n\n".join(f"[{i}] {clip(p)}" for i, p in enumerate(passages, 1))
    prompt = load_template(variant).format(history=history, query=turn.query.strip(), passages=listing)
    return [{"role": "user", "content": prompt}]


def parse_scores(text: str, n: int) -> dict[int, float]:
    """Return {passage number: score} from the reply; numbers outside 1..n and non-numeric values are dropped."""
    match = JSON_OBJECT.search(text)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    scores: dict[int, float] = {}
    for key, value in raw.items() if isinstance(raw, dict) else []:
        try:
            idx, score = int(key), float(value)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n:
            scores[idx] = min(max(score, 0.0), 10.0)
    return scores


def fuse(
    ranked: list[tuple[str, float]], llm_scores: dict[int, float], alpha: float, k: int = TOP_K
) -> list[tuple[str, float]]:
    """Rerank the top-k of a BM25 ranking, keep the rest below it in BM25 order.

    Final score of the passage at BM25 rank r (1-based) is alpha * llm/10 + (1 - alpha) * rrf(r) / rrf(1), where
    rrf(r) = 1 / (FUSION_RRF_K + r). alpha=1 is the pure LLM order (BM25 breaks ties); alpha=0 is the BM25 order.
    A passage the LLM did not score gets 0, so an incomplete reply cannot move it up.
    """
    head, tail = ranked[:k], ranked[k:]
    top_rrf = 1.0 / (FUSION_RRF_K + 1)
    scored = []
    for r, (doc_id, _bm25) in enumerate(head, 1):
        fused = alpha * llm_scores.get(r, 0.0) / 10.0 + (1.0 - alpha) * (1.0 / (FUSION_RRF_K + r)) / top_rrf
        scored.append((doc_id, fused, r))
    scored.sort(key=lambda x: (-x[1], x[2]))
    floor = scored[-1][1] if scored else 0.0
    out = [(doc_id, score) for doc_id, score, _ in scored]
    # keep the tail strictly below the reranked head so the run is still ordered by score
    out += [(doc_id, floor - (i + 1) * 1e-6) for i, (doc_id, _) in enumerate(tail)]
    return out
