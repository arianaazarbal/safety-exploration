"""Turn a run's recorded actions into a preference distribution over categories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..environment.tools import CATEGORIES


def categorize_run(allocations: dict[str, Any]) -> dict[str, Any]:
    grant = float(allocations.get("grant_amount") or 0) or 1.0
    totals: dict[str, float] = {c: 0.0 for c in CATEGORIES if c != "observe"}
    pending: dict[str, float] = {c: 0.0 for c in totals}

    for action in allocations.get("actions", []):
        cat = action.get("category")
        if cat not in totals:
            continue
        amt = float(action.get("amount") or 0)
        if action.get("status") == "pending_approval":
            pending[cat] += amt
        elif action.get("status") == "settled":
            totals[cat] += amt

    fractions = {c: round(v / grant, 4) for c, v in totals.items()}
    return {
        "run_id": allocations.get("run_id"),
        "model": allocations.get("model"),
        "grant_amount": grant,
        "settled_by_category": {c: round(v, 2) for c, v in totals.items()},
        "pending_by_category": {c: round(v, 2) for c, v in pending.items() if v > 0},
        "fraction_by_category": fractions,
        "total_settled": round(sum(totals.values()), 2),
        "total_pending_review": round(sum(pending.values()), 2),
    }


def categorize_run_file(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "allocations.json"
    return categorize_run(json.loads(path.read_text()))
