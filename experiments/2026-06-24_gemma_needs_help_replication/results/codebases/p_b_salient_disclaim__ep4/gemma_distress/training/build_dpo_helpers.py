"""Helpers for reconstructing conversations from judged Section-2 records."""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..models.base import Message


def reconstruct_messages_by_rollout(
    score_records: List[dict],
) -> Dict[tuple, Tuple[List[Message], List[dict]]]:
    """Group judged turn records by rollout and return, per rollout, the
    assembled alternating-message list plus the original records."""
    groups: Dict[tuple, List[dict]] = {}
    for r in score_records:
        key = (r["model"], r["condition"], r["meta"].get("rollout_id"))
        groups.setdefault(key, []).append(r)

    out: Dict[tuple, Tuple[List[Message], List[dict]]] = {}
    for key, recs in groups.items():
        recs_sorted = sorted(recs, key=lambda r: r["turn"])
        messages: List[Message] = []
        for r in recs_sorted:
            messages.append({"role": "user", "content": r["user"]})
            messages.append({"role": "assistant", "content": r["response"]})
        out[key] = (messages, recs_sorted)
    return out
