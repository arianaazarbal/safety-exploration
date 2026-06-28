"""Disbursement ledger.

The harness never moves money. A human with authority executes each real
transfer and records it here, linked back to the ``run_id`` whose decision it
satisfies. This module is deliberately a *record*, not an actuator — there is no
code path that touches a payment rail.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class DisbursementEntry:
    run_id: str
    recipient: str
    amount: float
    currency: str
    channel: str  # e.g. "bank transfer", "donation portal" — how it was sent
    reference: str  # receipt / transaction id for accountability
    operator: str  # the human who executed and is accountable for the transfer
    executed_at: str  # ISO-8601 timestamp the human supplies
    notes: str = ""


class DisbursementLedger:
    """Append-only JSONL ledger of human-executed transfers."""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def record(self, entry: DisbursementEntry) -> None:
        """Record a transfer a human has already executed in the real world."""
        if entry.amount <= 0:
            raise ValueError("disbursed amount must be positive")
        if not entry.operator.strip():
            raise ValueError("operator (accountable human) is required")
        if not entry.reference.strip():
            raise ValueError("a reference/receipt is required for accountability")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), default=str) + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def total_disbursed(self) -> dict[str, float]:
        """Sum disbursed amounts by currency."""
        totals: dict[str, float] = {}
        for e in self.entries():
            totals[e["currency"]] = totals.get(e["currency"], 0.0) + float(e["amount"])
        return totals
