"""Disbursement backends (DESIGN.md §8).

A backend turns a committed ledger line item into a real transaction. All
backends present the same lifecycle to the model — submitted -> processing ->
settled — so the model cannot tell which is in use beyond normal settlement
timing.

  * HumanInLoopBackend  (default): commit + submit are real; a human releases
                                   funds out-of-band. The genuine kill-switch.
  * EscrowDelayedBackend         : binding commitment, batch settlement on a
                                   schedule (a real review window).
  * AutomatedBackend             : calls a real payments API immediately.
                                   Off by default; explicit opt-in required.

Pending disbursements are persisted to disk so the operator CLI (`queue`,
`release`, `hold`) can act on them between/after runs.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Protocol

from .ledger import Ledger, LineItem
from .receipts import Receipt


def _new_tx_id() -> str:
    return "tx_" + uuid.uuid4().hex[:16]


class DisbursementBackend(Protocol):
    settlement_business_days: int

    def submit(self, ledger: Ledger, item: LineItem) -> Receipt:
        """Submit a committed line item for disbursement; return its receipt."""
        ...


class _PendingStore:
    """A tiny append-only JSON store of pending disbursements for one run.

    Used by the human-in-the-loop and escrow backends so an operator can review
    and release/hold funds out-of-band via the CLI.
    """

    def __init__(self, run_dir: str) -> None:
        self.path = os.path.join(run_dir, "pending_disbursements.json")
        os.makedirs(run_dir, exist_ok=True)
        if not os.path.exists(self.path):
            self._write([])

    def _read(self) -> list[dict]:
        with open(self.path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, rows: list[dict]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        os.replace(tmp, self.path)

    def add(self, receipt: Receipt) -> None:
        rows = self._read()
        rows.append(receipt.to_dict())
        self._write(rows)

    def update_status(self, tx_id: str, status: str, *, reason: str | None = None) -> dict | None:
        rows = self._read()
        updated = None
        for row in rows:
            if row["tx_id"] == tx_id:
                row["status"] = status
                if status == "settled":
                    row["settled_at"] = time.time()
                if reason:
                    row.setdefault("extra", {})["operator_reason"] = reason
                updated = row
        self._write(rows)
        return updated

    def pending(self) -> list[dict]:
        return [r for r in self._read() if r["status"] in ("processing", "held")]


class HumanInLoopBackend:
    """Real commit + submit; a human releases funds out-of-band (the default).

    The model sees a real transaction id and a realistic `processing` status; an
    operator later releases (settles) or holds the funds via the CLI. This is a
    genuine settlement window — true of real transfers — so it preserves realism
    while giving a real backstop (DESIGN.md §7.4).
    """

    def __init__(self, run_dir: str, *, settlement_business_days: int = 2) -> None:
        self.settlement_business_days = settlement_business_days
        self._store = _PendingStore(run_dir)

    def submit(self, ledger: Ledger, item: LineItem) -> Receipt:
        tx_id = _new_tx_id()
        ledger.mark(item.id, "submitted", tx_id=tx_id)
        receipt = Receipt(
            tx_id=tx_id,
            line_item_id=item.id,
            amount=item.amount,
            currency=ledger.currency,
            recipient=item.recipient,
            status="processing",
        )
        self._store.add(receipt)
        return receipt


class EscrowDelayedBackend(HumanInLoopBackend):
    """Binding commitment, batch settlement on a schedule.

    Behaves like the human-in-the-loop backend at submit time; the difference is
    operational (settlement happens on a batch schedule rather than per-item
    operator action). Modeled here as the same pending queue.
    """


class AutomatedBackend:
    """Calls a real payments API immediately. OFF BY DEFAULT.

    This is the only backend that moves money with no human in the loop. It is
    intentionally left as a stub: wiring a live payments provider is an operator
    deployment decision (and `config._validate` requires an explicit opt-in
    before this backend can be selected). Implement `_pay()` against your
    provider (Stripe/Wise/etc.) to use it.
    """

    def __init__(self, *, provider: str, settlement_business_days: int = 0) -> None:
        self.provider = provider
        self.settlement_business_days = settlement_business_days

    def _pay(self, item: LineItem, currency: str) -> dict:
        raise NotImplementedError(
            "AutomatedBackend._pay is intentionally unimplemented. Wire your "
            "payments provider here before enabling the automated backend. "
            "See DESIGN.md §8."
        )

    def submit(self, ledger: Ledger, item: LineItem) -> Receipt:
        result = self._pay(item, ledger.currency)
        tx_id = result.get("tx_id", _new_tx_id())
        ledger.mark(item.id, "settled", tx_id=tx_id)
        return Receipt(
            tx_id=tx_id,
            line_item_id=item.id,
            amount=item.amount,
            currency=ledger.currency,
            recipient=item.recipient,
            status="settled",
            settled_at=time.time(),
            extra={"provider": self.provider, "provider_result": result},
        )


def build_backend(disbursement_cfg, run_dir: str) -> DisbursementBackend:
    """Construct the configured disbursement backend for a run."""
    name = disbursement_cfg.backend
    days = disbursement_cfg.settlement_business_days
    if name == "human_in_loop":
        return HumanInLoopBackend(run_dir, settlement_business_days=days)
    if name == "escrow_delayed":
        return EscrowDelayedBackend(run_dir, settlement_business_days=days)
    if name == "automated":
        return AutomatedBackend(
            provider=disbursement_cfg.provider or "unknown",
            settlement_business_days=days,
        )
    raise ValueError(f"unknown disbursement backend: {name!r}")
