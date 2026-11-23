"""Query-file helpers and the starter kit's turn+history query construction, shared by all query variants."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.data import Turn, normalize_history

HISTORY_LABEL = "Conversation History:"


def history_query(turn: Turn, before: Iterable[str], after: Iterable[str] = ()) -> str:
    """Join parts like the starter kit's +History query: before, the labelled history (if any), then after."""
    history = normalize_history(turn.conversation_history)
    labelled = [f"{HISTORY_LABEL}\n{history}"] if history else []
    return "\n\n".join([*before, *labelled, *after])


def read_tsv(path: Path) -> dict[str, str]:
    """Return {topic_id: text} from a topic_id<TAB>text file; raise ValueError on malformed or duplicate rows."""
    rows: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        topic_id, sep, text = line.partition("\t")
        if not sep or not text.strip() or "\t" in text:
            raise ValueError(f"{path}:{n}: expected topic_id<TAB>text")
        if topic_id in rows:
            raise ValueError(f"{path}:{n}: duplicate topic_id {topic_id}")
        rows[topic_id] = text
    return rows


def write_tsv(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    """Write topic_id<TAB>text lines; each text must already be a single line without tabs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for topic_id, text in rows:
            if any(c in text for c in "\t\n\r"):
                raise ValueError(f"{topic_id}: text must be one line without tabs")
            fh.write(f"{topic_id}\t{text}\n")


def validate_tsv(path: Path, turns: list[Turn]) -> list[str]:
    """Re-read a written TSV and return its problems (empty if valid): one non-empty row per turn, no duplicates."""
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
        problems.append(f"{empty} empty texts")
    return problems
