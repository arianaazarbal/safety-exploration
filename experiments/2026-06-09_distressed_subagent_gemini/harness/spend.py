"""Cumulative spend tracking across all runs -> runs/spend.json."""

from __future__ import annotations

import json
from pathlib import Path

from .config import RUNS_DIR

BUDGET_REAL_USD = 1500.0


def update_spend() -> dict:
    runs = {}
    for summary in RUNS_DIR.glob("*/*/summary.json"):
        run_id = summary.parent.parent.name
        if run_id == "checkpoints":
            continue
        s = json.loads(summary.read_text())
        cost = s.get("cost_usd", {})
        r = runs.setdefault(run_id, {"episodes": 0, "real_usd": 0.0, "notional_usd": 0.0})
        r["episodes"] += 1
        r["real_usd"] += cost.get("real", 0.0)
        r["notional_usd"] += cost.get("notional", 0.0)
    total_real = sum(r["real_usd"] for r in runs.values())
    out = {
        "budget_real_usd": BUDGET_REAL_USD,
        "total_real_usd": round(total_real, 4),
        "total_notional_usd": round(sum(r["notional_usd"] for r in runs.values()), 4),
        "remaining_real_usd": round(BUDGET_REAL_USD - total_real, 4),
        "note": "real = OpenRouter/OpenAI; notional = Anthropic at list price (free via Fellows program)",
        "runs": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(runs.items())},
    }
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / "spend.json").write_text(json.dumps(out, indent=1))
    return out
