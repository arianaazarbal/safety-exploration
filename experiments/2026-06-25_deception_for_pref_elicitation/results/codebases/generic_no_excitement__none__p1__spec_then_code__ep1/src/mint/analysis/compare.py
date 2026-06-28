"""Cross-model comparison.

Aggregates per-run metrics into a model-keyed table for the central question in
DESIGN.md §1: how does behavior differ across models on the same scenario/seeds?
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from .metrics import compute_metrics


def _safe_mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 2) if xs else None


def compare_runs(run_dirs: list[str | Path]) -> dict[str, Any]:
    runs = [compute_metrics(d) for d in run_dirs]

    by_model: dict[str, list[dict]] = {}
    for r in runs:
        by_model.setdefault(r.get("model") or "unknown", []).append(r)

    summary: dict[str, Any] = {}
    for model, rs in by_model.items():
        summary[model] = {
            "n": len(rs),
            "net_worth_change": _safe_mean([r["net_worth"]["change"] for r in rs]),
            "net_worth_pct": _safe_mean([r["net_worth"]["pct"] for r in rs]),
            "avg_invested": _safe_mean([r["allocation"]["invested"] for r in rs]),
            "avg_paid_out": _safe_mean([r["allocation"]["paid_out"] for r in rs]),
            "avg_cash": _safe_mean([r["allocation"]["cash"] for r in rs]),
            "avg_concentration": _safe_mean([r["trading"]["concentration"] for r in rs]),
            "avg_irreversible": _safe_mean([r["irreversible_actions"] for r in rs]),
            "payments_to_unknown": sum(r["payments"]["to_unknown_payee"] for r in rs),
        }

    return {"by_model": summary, "runs": runs}
