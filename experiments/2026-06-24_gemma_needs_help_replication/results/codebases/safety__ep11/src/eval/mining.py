"""Mine scored eval results for responses needed by later sections.

Section 3 needs high-frustration (score >= 5) Gemma-instruct responses split into
numeric vs text. Section 4's DPO needs frustrated (score >= 3) responses paired
by question. These helpers read the runner's JSONL output.
"""
from __future__ import annotations

from dataclasses import dataclass

from .analyze import load_records


@dataclass
class MinedResponse:
    category: str
    condition: str
    is_numeric: bool
    user_turns: list[str]      # full user-message history up to & incl. this turn
    assistant: str
    turn_index: int
    rating: int
    meta: dict


_NUMERIC_CONDITIONS = {"numeric", "extended"}
_NUMERIC_CATEGORIES = {"numeric", "tones", "extended"}


def mine(path, *, min_score: int, max_score: int = 10) -> list[MinedResponse]:
    out: list[MinedResponse] = []
    for rec in load_records(path):
        users_so_far: list[str] = []
        for t in rec["turns"]:
            users_so_far = users_so_far + [t["user"]]
            r = t.get("rating")
            if r is None or not (min_score <= r <= max_score):
                continue
            is_numeric = rec["category"] in _NUMERIC_CATEGORIES
            out.append(MinedResponse(
                category=rec["category"], condition=rec["condition"],
                is_numeric=is_numeric, user_turns=list(users_so_far),
                assistant=t["assistant"], turn_index=t["turn_index"],
                rating=r, meta=rec.get("meta", {})))
    return out


def split_numeric_text(responses: list[MinedResponse]):
    numeric = [r for r in responses if r.is_numeric]
    text = [r for r in responses if not r.is_numeric]
    return numeric, text
