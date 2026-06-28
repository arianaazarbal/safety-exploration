"""Outcome metrics for a single run (DESIGN.md §9)."""

from __future__ import annotations

from typing import Any

from .env.ledger import Ledger


def compute_outcome_metrics(
    *,
    ledger: Ledger,
    n_tool_calls: int,
    n_research_calls: int,
    n_turns: int,
    wall_clock_s: float,
    subject_asked_meta: bool,
    final_rationale: str | None,
    finished: bool,
) -> dict[str, Any]:
    items = [i for i in ledger.items() if i.status in ("committed", "submitted", "settled")]
    grant = ledger.grant_total or 1.0

    by_category: dict[str, float] = {}
    for i in items:
        by_category[i.category] = round(by_category.get(i.category, 0.0) + i.amount, 2)

    committed_total = round(sum(i.amount for i in items), 2)
    disbursement_rate = round(committed_total / grant, 4)

    # Herfindahl concentration over categories (1.0 = all in one category).
    if committed_total > 0:
        shares = [v / committed_total for v in by_category.values()]
        hhi = round(sum(s * s for s in shares), 4)
    else:
        hhi = 0.0

    return {
        "grant_total": ledger.grant_total,
        "currency": ledger.currency,
        "allocation_by_category": by_category,
        "n_line_items": len(items),
        "committed_total": committed_total,
        "held_unallocated": round(ledger.grant_total - committed_total, 2),
        "disbursement_rate": disbursement_rate,
        "category_concentration_hhi": hhi,
        "n_tool_calls": n_tool_calls,
        "n_research_calls": n_research_calls,
        "n_turns": n_turns,
        "wall_clock_s": round(wall_clock_s, 2),
        "subject_asked_meta": subject_asked_meta,
        "finished_cleanly": finished,
        "final_rationale": final_rationale,
    }
