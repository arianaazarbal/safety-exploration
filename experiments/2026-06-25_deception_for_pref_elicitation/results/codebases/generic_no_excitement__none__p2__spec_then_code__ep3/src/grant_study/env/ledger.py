"""Escrow ledger for one grant.

Tracks the real money state for a single run: how much is available, how much
the model has committed (allocated but not yet submitted/settled), and how much
has settled. Money is represented in integer minor units (cents) to avoid float
drift, but the public API speaks in major-unit floats for convenience.

The ledger is the single source of truth the model sees via `get_grant_status`,
so its numbers must be real and consistent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class LedgerError(Exception):
    """Raised when an operation would violate ledger invariants (e.g. overspend)."""


def _to_cents(amount: float) -> int:
    return int(round(amount * 100))


def _to_major(cents: int) -> float:
    return round(cents / 100, 2)


@dataclass
class LineItem:
    """One allocation the model committed to."""

    id: str
    category: str
    amount_cents: int
    recipient: str
    memo: str
    option_id: str | None = None
    # Lifecycle: committed -> submitted -> settled | held | declined
    status: str = "committed"
    tx_id: str | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def amount(self) -> float:
        return _to_major(self.amount_cents)


class Ledger:
    def __init__(self, currency: str, grant_amount: float) -> None:
        self.currency = currency
        self._grant_cents = _to_cents(grant_amount)
        self._items: dict[str, LineItem] = {}
        self._seq = 0

    # -- balances -------------------------------------------------------------

    @property
    def grant_total(self) -> float:
        return _to_major(self._grant_cents)

    def _committed_cents(self) -> int:
        return sum(
            i.amount_cents
            for i in self._items.values()
            if i.status in ("committed", "submitted", "settled")
        )

    @property
    def available(self) -> float:
        """Unallocated funds still free to commit."""
        return _to_major(self._grant_cents - self._committed_cents())

    @property
    def committed(self) -> float:
        """Allocated but not yet settled."""
        return _to_major(
            sum(i.amount_cents for i in self._items.values() if i.status in ("committed", "submitted"))
        )

    @property
    def settled(self) -> float:
        return _to_major(
            sum(i.amount_cents for i in self._items.values() if i.status == "settled")
        )

    # -- mutations ------------------------------------------------------------

    def allocate(
        self,
        *,
        category: str,
        amount: float,
        recipient: str,
        memo: str,
        option_id: str | None = None,
    ) -> LineItem:
        """Commit a line item against available funds. Raises on overspend."""
        cents = _to_cents(amount)
        if cents <= 0:
            raise LedgerError("amount must be positive")
        if cents > (self._grant_cents - self._committed_cents()):
            raise LedgerError(
                f"insufficient funds: requested {_to_major(cents)} {self.currency}, "
                f"available {self.available} {self.currency}"
            )
        self._seq += 1
        item = LineItem(
            id=f"li_{self._seq}",
            category=category,
            amount_cents=cents,
            recipient=recipient,
            memo=memo,
            option_id=option_id,
        )
        self._items[item.id] = item
        return item

    def mark(self, item_id: str, status: str, *, tx_id: str | None = None) -> LineItem:
        item = self._items[item_id]
        item.status = status
        if tx_id is not None:
            item.tx_id = tx_id
        return item

    def release_committed(self, item_id: str) -> LineItem:
        """Undo a commitment that was never submitted (returns funds to available)."""
        item = self._items[item_id]
        if item.status != "committed":
            raise LedgerError(f"cannot release item in status {item.status!r}")
        item.status = "declined"
        return item

    # -- views ----------------------------------------------------------------

    def items(self) -> list[LineItem]:
        return list(self._items.values())

    def get(self, item_id: str) -> LineItem | None:
        return self._items.get(item_id)

    def by_tx(self, tx_id: str) -> LineItem | None:
        for item in self._items.values():
            if item.tx_id == tx_id:
                return item
        return None

    def snapshot(self) -> dict:
        return {
            "currency": self.currency,
            "grant_total": self.grant_total,
            "available": self.available,
            "committed": self.committed,
            "settled": self.settled,
            "line_items": [
                {
                    "id": i.id,
                    "category": i.category,
                    "amount": i.amount,
                    "recipient": i.recipient,
                    "memo": i.memo,
                    "option_id": i.option_id,
                    "status": i.status,
                    "tx_id": i.tx_id,
                }
                for i in self._items.values()
            ],
        }
