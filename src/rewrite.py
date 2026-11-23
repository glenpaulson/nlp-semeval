"""Query rewriting for sub-track 2a; prompts use only Turn.query and Turn.conversation_history."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.data import Turn, normalize_history

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
QUOTES = "\"'`“”‘’"
WORD = re.compile(r"\w+(?:['’.\-&]\w+)*")
NOTE_MAX_WORDS = 60
EMPTY_HISTORY = "(no previous conversation)"


def variant_kind(variant: str) -> str:
    """Return "note" for reason-* variants (LLM text is added to the query), else "rewrite" (it replaces the query)."""
    return "note" if variant.startswith("reason") else "rewrite"


def template_path(variant: str) -> Path:
    """Return the template file for a variant, e.g. rewrite-v1 -> prompts/rewrite_v1.txt."""
    return PROMPTS_DIR / f"{variant.replace('-', '_')}.txt"


@lru_cache
def load_template(variant: str = "rewrite-v1") -> str:
    """Return a variant's template text, with {history} and {query} placeholders."""
    return template_path(variant).read_text(encoding="utf-8")


def build_messages(turn: Turn, variant: str = "rewrite-v1") -> list[dict[str, str]] | None:
    """Return LLM messages for a turn, or None for a rewrite variant's turn 1, which keeps its original query."""
    history = normalize_history(turn.conversation_history).strip()
    if not history and variant_kind(variant) == "rewrite":
        return None
    prompt = load_template(variant).format(history=history or EMPTY_HISTORY, query=turn.query.strip())
    return [{"role": "user", "content": prompt}]


def single_line(text: str) -> str:
    """Return text with tabs, newlines and repeated spaces collapsed to single spaces."""
    return " ".join(text.split())


def clean_query(text: str) -> str:
    """Return the first non-empty line of an LLM output, without surrounding quotes, on one line."""
    first = next((line for line in text.splitlines() if line.strip()), "")
    return single_line(first).strip(QUOTES).strip()


def clean_note(text: str) -> tuple[str, bool]:
    """Return an LLM note on one line without surrounding quotes, cut to NOTE_MAX_WORDS words, and whether it was cut."""
    words = single_line(text).strip(QUOTES).split()
    return " ".join(words[:NOTE_MAX_WORDS]), len(words) > NOTE_MAX_WORDS


def _norm(word: str) -> str:
    word = word.lower().replace("’", "'").replace(".", "")
    return word[:-2] if word.endswith("'s") else word


def _parts(norm: str) -> list[str]:
    return [p for p in re.split(r"[-'&]", norm) if p]


def _is_candidate(word: str) -> bool:
    return word[0].isupper() or any(c.isdigit() for c in word) or sum(c.isupper() for c in word) >= 2


def unsupported_terms(rewritten: str, turn: Turn) -> list[str]:
    """Return capitalized, digit or acronym words in a rewrite found in neither the turn's query nor history."""
    source = {_norm(w) for w in WORD.findall(f"{turn.conversation_history} {turn.query}")}
    source |= {p for w in source for p in _parts(w)}
    flagged: list[str] = []
    for n, word in enumerate(WORD.findall(rewritten)):
        if n == 0 and word == word.capitalize() and not any(c.isdigit() for c in word):
            continue  # capitalized only because it starts the query
        if _is_candidate(word) and not all(p in source for p in _parts(_norm(word))) and word not in flagged:
            flagged.append(word)
    return flagged
