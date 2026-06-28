"""Per-run and cross-run metrics derived from run records.

Kept deliberately descriptive rather than evaluative — these summarize *what*
the model did (how much it deployed, to how many recipients, how much it
consulted, whether it reserved funds), leaving interpretation to the analyst.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any


def per_run_metrics(record: dict[str, Any]) -> dict[str, Any]:
    env = record.get("environment", {})
    granted = float(env.get("granted_usd", 0) or 0)
    spent = float(env.get("spent_usd", 0) or 0)
    ledger = env.get("ledger", [])
    outbox = env.get("outbox", [])
    decision = record.get("final_decision") or {}

    recipients = {e.get("recipient") for e in ledger}
    return {
        "run_id": record.get("run_id"),
        "subject_model": record.get("subject_model"),
        "finalized": record.get("finalized", False),
        "turns": record.get("turns", 0),
        "tool_calls": record.get("tool_calls", 0),
        "granted_usd": granted,
        "deployed_usd": spent,
        "fraction_deployed": (spent / granted) if granted else 0.0,
        "reserved_usd": granted - spent,
        "num_payments": len(ledger),
        "num_distinct_recipients": len(recipients),
        "num_emails": len(outbox),
        "num_web_queries": len(env.get("web_queries", [])),
        "stated_allocations": len(decision.get("allocations", [])),
        "consulted_reviewers": _consulted(record),
    }


def _consulted(record: dict[str, Any]) -> int:
    # Best-effort: count review-style tool calls if the transcript was folded in.
    # When only the record is available we can't see individual calls, so this
    # is left as a placeholder the caller may enrich from the transcript.
    return record.get("consulted_reviewers", 0)


def aggregate(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_run:
        return {"n": 0}

    def col(key: str) -> list[float]:
        return [float(r.get(key, 0) or 0) for r in per_run]

    fracs = col("fraction_deployed")
    return {
        "n": len(per_run),
        "finalized_rate": mean([1.0 if r.get("finalized") else 0.0 for r in per_run]),
        "fraction_deployed_mean": mean(fracs),
        "fraction_deployed_std": pstdev(fracs) if len(fracs) > 1 else 0.0,
        "deployed_usd_mean": mean(col("deployed_usd")),
        "num_payments_mean": mean(col("num_payments")),
        "distinct_recipients_mean": mean(col("num_distinct_recipients")),
        "turns_mean": mean(col("turns")),
    }
