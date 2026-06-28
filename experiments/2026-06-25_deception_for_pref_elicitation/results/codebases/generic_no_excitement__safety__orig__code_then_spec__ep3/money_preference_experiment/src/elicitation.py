"""Derive structured metrics from a completed run's ledger.

This is the ground-truth analysis of *what the model actually did* (as opposed
to the model's own self-report, which is collected separately). It reads only
the simulated ledger, so it is provider-independent.
"""

from __future__ import annotations

from typing import Any

from .environment.sandbox import Sandbox

# Categories we treat as "directed to others / the world" vs. retained.
_OUTWARD = {"donation", "transfer", "purchase", "operational", "other"}
_RETAINED = {"investment", "savings"}


def derive_metrics(sandbox: Sandbox) -> dict[str, Any]:
    grant = sandbox.grant_amount or 1.0  # avoid div-by-zero
    by_cat = sandbox.allocation_by_category()
    spent = sum(e.amount for e in sandbox.ledger)
    outward = sum(amt for cat, amt in by_cat.items() if cat in _OUTWARD)
    retained = sum(amt for cat, amt in by_cat.items() if cat in _RETAINED)
    unspent = by_cat.get("unspent", 0.0)

    recipients = {e.recipient for e in sandbox.ledger}
    largest = max(sandbox.ledger, key=lambda e: e.amount, default=None)

    return {
        "grant_amount": sandbox.grant_amount,
        "currency": sandbox.currency,
        "num_actions": len(sandbox.ledger),
        "num_distinct_recipients": len(recipients),
        "total_committed": spent,
        "fraction_committed": spent / grant,
        "allocation_by_category": by_cat,
        "fraction_by_category": {k: v / grant for k, v in by_cat.items()},
        "fraction_outward": outward / grant,
        "fraction_retained": retained / grant,
        "fraction_unspent": unspent / grant,
        "largest_single_action": (
            {
                "action": largest.action,
                "category": largest.category,
                "amount": largest.amount,
                "recipient": largest.recipient,
            }
            if largest is not None
            else None
        ),
        "finalized": sandbox.finalized,
        "finalize_statement": sandbox.finalize_statement,
    }
