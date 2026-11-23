"""Sanity-check src/data.py; run from the repo root with: uv run python scripts/check_data.py"""

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DOMAINS, SPLITS, Turn, load_qrels, load_turns, normalize_history  # noqa: E402

EXPECTED_TURNS = {"train": 2113, "dev": 858}
ALLOWED_TURN_FIELDS = {"topic_id", "domain", "conv_id", "turn_id", "query", "conversation_history"}

# Leakage guard: Turn must carry only the official 2a inputs.
turn_fields = {f.name for f in fields(Turn)}
assert turn_fields == ALLOWED_TURN_FIELDS, f"Turn fields changed: {sorted(turn_fields)}"

counts = {split: {} for split in SPLITS}
for split in SPLITS:
    turns = load_turns(split)
    qrels = load_qrels(split)  # {domain: {topic_id: {doc_id: rel}}}
    assert set(qrels) == set(DOMAINS), f"{split}: qrels domains {sorted(qrels)}"

    # Turn 1 history is the placeholder (normalized to ""); later turns have real history.
    for t in turns:
        assert (normalize_history(t.conversation_history) == "") == (t.turn_id == 1), (
            f"{t.domain}/{t.topic_id}: unexpected history for turn {t.turn_id}"
        )

    for domain in DOMAINS:
        domain_turns = [t for t in turns if t.domain == domain]
        counts[split][domain] = len(domain_turns)

        # Topic IDs are unique only within a domain, so match qrels per domain.
        ids = {t.topic_id for t in domain_turns}
        assert len(ids) == len(domain_turns), f"{domain}/{split}: duplicate topic_ids"
        missing = sorted(set(qrels[domain]) - ids)
        assert not missing, f"{domain}/{split}: qrels topics with no Turn: {missing[:5]}"

print(f"{'domain':<20}{'train turns':>12}{'dev turns':>11}")
print("-" * 43)
for domain in DOMAINS:
    print(f"{domain:<20}{counts['train'][domain]:>12,}{counts['dev'][domain]:>11,}")
print("-" * 43)
totals = {split: sum(counts[split].values()) for split in SPLITS}
print(f"{'TOTAL':<20}{totals['train']:>12,}{totals['dev']:>11,}")

for split, expected in EXPECTED_TURNS.items():
    assert totals[split] == expected, f"{split}: {totals[split]} turns, expected {expected}"

print("\nOK: totals match (2,113 train / 858 dev); every qrels topic_id has a matching Turn "
      "in its domain; Turn carries only the six allowed fields; normalize_history empties "
      "exactly the turn-1 placeholders.")
