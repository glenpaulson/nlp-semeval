"""RETECO Track 2 loader for sub-track 2a: Turns never carry gold fields; topic IDs repeat across domains."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "reteco_data" / "track2_recor"

DOMAINS: tuple[str, ...] = (
    "biology",
    "drones",
    "earth_science",
    "economics",
    "hardware",
    "law",
    "medicalsciences",
    "politics",
    "psychology",
    "robotics",
    "sustainable_living",
)

SPLITS: tuple[str, ...] = ("train", "dev")


@dataclass(frozen=True, slots=True)
class Turn:
    """One target turn: the only information a 2a system may use as input."""

    topic_id: str  # f"{conv_id}_turn_{turn_id}"; unique within a domain only
    domain: str
    conv_id: str
    turn_id: int
    query: str  # the current turn's question
    conversation_history: str  # earlier Q/A pairs; "No previous conversation." on turn 1


def topic_id(conv_id: str, turn_id: int) -> str:
    return f"{conv_id}_turn_{turn_id}"


NO_HISTORY = "No previous conversation."


def normalize_history(text: str) -> str:
    """Return "" for the turn-1 placeholder, else the text unchanged."""
    return "" if text.strip() == NO_HISTORY else text


def _resolve_domains(domains: str | Iterable[str] | None) -> tuple[str, ...]:
    if domains is None:
        return DOMAINS
    if isinstance(domains, str):
        domains = (domains,)
    domains = tuple(domains)
    unknown = sorted(set(domains) - set(DOMAINS))
    if unknown:
        raise ValueError(f"unknown domain(s) {unknown}; expected some of {list(DOMAINS)}")
    return domains


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def load_turns(split: str, domains: str | Iterable[str] | None = None) -> list[Turn]:
    """Return one Turn per target turn, in file order, for the given split and domains."""
    _check_split(split)
    turns: list[Turn] = []
    for domain in _resolve_domains(domains):
        path = DATA_DIR / domain / f"benchmark_{split}.json"
        with path.open(encoding="utf-8") as fh:
            conversations = json.load(fh)
        for conv in conversations:
            conv_id = str(conv["id"])
            for t in conv["turns"]:
                # Pick fields explicitly; never pass the raw record (**t) through.
                turns.append(
                    Turn(
                        topic_id=topic_id(conv_id, t["turn_id"]),
                        domain=domain,
                        conv_id=conv_id,
                        turn_id=int(t["turn_id"]),
                        query=t["query"],
                        conversation_history=t["conversation_history"],
                    )
                )
    return turns


def load_qrels(
    split: str, domains: str | Iterable[str] | None = None
) -> dict[str, dict[str, dict[str, int]]]:
    """Return {domain: {topic_id: {doc_id: relevance}}}; each inner dict is pytrec_eval qrels."""
    _check_split(split)
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for domain in _resolve_domains(domains):
        domain_qrels: dict[str, dict[str, int]] = {}
        path = DATA_DIR / domain / f"qrels_{split}.txt"
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"{path}:{lineno}: expected 4 fields, got {len(parts)}")
                qid, _iteration, doc_id, rel = parts
                domain_qrels.setdefault(qid, {})[doc_id] = int(rel)
        qrels[domain] = domain_qrels
    return qrels


def iter_corpus(domain: str) -> Iterator[tuple[str, str]]:
    """Yield (doc_id, content) from a domain's documents.jsonl, one line at a time."""
    (domain,) = _resolve_domains(domain)
    path = DATA_DIR / domain / "documents.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                doc = json.loads(line)
                yield doc["doc_id"], doc["content"]
