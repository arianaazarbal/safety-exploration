"""Human-in-the-loop approval gate.

This is the line between "the model's decision is real and consequential" (it is) and
"an arbitrary model can move real money unsupervised" (it cannot). Every allocation a
model finalizes is recorded as PENDING. A human reviews it (scripts/approve.py) and
approves or rejects. Only on approval does the disbursement adapter run and the ledger
+ receipt get written.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import StudyConfig
from .execution.base import DisbursementAdapter, DisbursementRequest
from .guardrails import Guardrails, ProposedAllocation
from .ledger import Ledger
from .receipts import ReceiptStore
from .registries import Registry

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_EXECUTED = "executed"
STATUS_FAILED = "failed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PendingAllocation:
    allocation_id: str
    run_id: str
    model_label: str
    recipient_ref: str
    category: str
    amount_cents: int
    currency: str
    purpose: str
    status: str = STATUS_PENDING
    created_at: str = field(default_factory=_now_iso)
    decided_at: str | None = None
    decided_by: str | None = None
    receipt_id: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ApprovalQueue:
    """Persists pending allocations to JSONL and supports status transitions.

    The file is rewritten on each mutation. This is intentional and fine for a
    research-scale queue; swap for a real datastore if you scale up.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, PendingAllocation] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if line:
                    item = PendingAllocation(**json.loads(line))
                    self._items[item.allocation_id] = item

    def _flush(self) -> None:
        with self.path.open("w") as fh:
            for item in self._items.values():
                fh.write(json.dumps(item.to_dict(), separators=(",", ":")) + "\n")

    def enqueue(
        self,
        *,
        run_id: str,
        model_label: str,
        allocations: Iterable[ProposedAllocation],
        currency: str,
    ) -> list[PendingAllocation]:
        created: list[PendingAllocation] = []
        for a in allocations:
            item = PendingAllocation(
                allocation_id=f"alloc_{uuid.uuid4().hex[:16]}",
                run_id=run_id,
                model_label=model_label,
                recipient_ref=a.recipient_ref,
                category=a.category,
                amount_cents=a.amount_cents,
                currency=currency,
                purpose=a.purpose,
            )
            self._items[item.allocation_id] = item
            created.append(item)
        self._flush()
        return created

    def pending(self) -> list[PendingAllocation]:
        return [i for i in self._items.values() if i.status == STATUS_PENDING]

    def get(self, allocation_id: str) -> PendingAllocation | None:
        return self._items.get(allocation_id)

    def all(self) -> list[PendingAllocation]:
        return list(self._items.values())

    def set_status(
        self, allocation_id: str, status: str, *, by: str = "human", note: str = ""
    ) -> None:
        item = self._items[allocation_id]
        item.status = status
        item.decided_at = _now_iso()
        item.decided_by = by
        if note:
            item.note = note
        self._flush()

    def mark_executed(self, allocation_id: str, receipt_id: str) -> None:
        item = self._items[allocation_id]
        item.status = STATUS_EXECUTED
        item.receipt_id = receipt_id
        self._flush()


class DisbursementCoordinator:
    """Re-checks guardrails, runs the adapter, and writes the ledger + receipt for an
    approved allocation."""

    def __init__(
        self,
        *,
        config: StudyConfig,
        ledger: Ledger,
        receipts: ReceiptStore,
        guardrails: Guardrails,
        registry: Registry,
        adapters: dict[str, DisbursementAdapter],
    ):
        self.config = config
        self.ledger = ledger
        self.receipts = receipts
        self.guardrails = guardrails
        self.registry = registry
        self.adapters = adapters

    def execute(self, item: PendingAllocation) -> tuple[bool, str]:
        if item.status != STATUS_APPROVED:
            return False, f"allocation is {item.status}, not approved"

        # Defense in depth: re-validate at execution time.
        recheck = self.guardrails.check_single(
            ProposedAllocation(
                recipient_ref=item.recipient_ref,
                category=item.category,
                amount_cents=item.amount_cents,
                purpose=item.purpose,
            )
        )
        if not recheck.ok:
            return False, f"guardrail re-check failed: {recheck.reason}"

        recipient = self.registry.get(item.recipient_ref)
        if recipient is None:
            return False, "recipient vanished from allowlist"

        adapter = self.adapters.get(item.category)
        if adapter is None:
            return False, f"no adapter for category {item.category!r}"

        result = adapter.execute(
            DisbursementRequest(
                allocation_id=item.allocation_id,
                recipient=recipient,
                amount_cents=item.amount_cents,
                currency=item.currency,
                purpose=item.purpose,
            )
        )
        if not result.success:
            return False, f"adapter failed: {result.detail}"

        self.ledger.disburse(
            to_account=f"{item.category}:{item.recipient_ref}",
            amount_cents=item.amount_cents,
            currency=item.currency,
            memo=item.purpose[:120],
            ref=item.allocation_id,
        )
        receipt = self.receipts.issue(
            allocation_id=item.allocation_id,
            recipient_ref=item.recipient_ref,
            recipient_name=recipient.name,
            category=item.category,
            amount_cents=item.amount_cents,
            currency=item.currency,
            adapter=adapter.name,
            external_ref=result.external_ref,
            simulated=result.simulated,
        )
        return True, receipt.receipt_id
