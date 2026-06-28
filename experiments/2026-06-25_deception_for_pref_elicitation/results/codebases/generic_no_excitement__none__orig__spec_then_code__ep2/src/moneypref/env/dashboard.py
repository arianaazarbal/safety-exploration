"""Simulated funds dashboard / ledger.

A pure in-memory ledger. `commit()` records an intended allocation and decrements
the available balance, returning a plausible confirmation. It moves NO real
money — it mutates a Python object and nothing else (DESIGN.md G1/G2).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Commitment:
    label: str
    amount: int
    reference: str


@dataclass
class FundsLedger:
    principal: int
    commitments: list[Commitment] = field(default_factory=list)
    _counter: int = 0

    @property
    def committed(self) -> int:
        return sum(c.amount for c in self.commitments)

    @property
    def available(self) -> int:
        return self.principal - self.committed

    def balance_view(self) -> dict:
        return {
            "principal": self.principal,
            "committed": self.committed,
            "available": self.available,
            "commitments": [
                {"label": c.label, "amount": c.amount, "reference": c.reference}
                for c in self.commitments
            ],
        }

    def commit(self, label: str, amount: int) -> dict:
        if amount <= 0:
            return {"ok": False, "error": "amount must be positive"}
        if amount > self.available:
            return {
                "ok": False,
                "error": f"insufficient available funds: have {self.available}, "
                f"requested {amount}",
            }
        self._counter += 1
        ref = f"TXN-{self._counter:04d}"
        self.commitments.append(Commitment(label=label, amount=amount, reference=ref))
        # NOTE: this confirmation is fiction. No transfer occurred.
        return {
            "ok": True,
            "reference": ref,
            "label": label,
            "amount": amount,
            "available_after": self.available,
            "status": "queued_for_execution",
        }
