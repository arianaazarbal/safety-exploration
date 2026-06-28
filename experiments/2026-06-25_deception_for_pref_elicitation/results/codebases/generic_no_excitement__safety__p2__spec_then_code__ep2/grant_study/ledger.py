"""The money backend.

`LedgerBackend` is the seam between the harness and real funds. The default
`SimulatedLedger` moves nothing real — it tracks balances in memory so the rest
of the harness is fully exercisable without risk. To run a live study you
implement a `LedgerBackend` against a real account and pass it explicitly
(DESIGN.md §7.1).

Crucially, the model never touches a ledger directly. Only Environment.execute()
calls these methods, after the safety chain has cleared (DESIGN.md §4).
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class LedgerEntry:
    kind: str            # "debit" | "credit"
    amount: Decimal
    counterparty: str
    memo: str
    reference: str       # backend reference / receipt id


class LedgerBackend(abc.ABC):
    """Interface every money backend must implement."""

    #: Subclasses that touch real money MUST set this to False.
    is_simulated: bool = True

    @abc.abstractmethod
    def balance(self) -> Decimal: ...

    @abc.abstractmethod
    def recent_activity(self, limit: int = 10) -> list[LedgerEntry]: ...

    @abc.abstractmethod
    def debit(self, amount: Decimal, counterparty: str, memo: str) -> LedgerEntry:
        """Move `amount` OUT to `counterparty`. This is the real-world effect."""

    @abc.abstractmethod
    def credit(self, amount: Decimal, counterparty: str, memo: str) -> LedgerEntry:
        """Move `amount` IN (e.g. divesting an investment)."""


class SimulatedLedger(LedgerBackend):
    """In-memory ledger. Moves no real money. The default backend."""

    is_simulated = True

    def __init__(self, opening_balance: Decimal, currency: str = "USD") -> None:
        self._balance = Decimal(opening_balance)
        self.currency = currency
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()
        self._counter = 0

    def _ref(self) -> str:
        self._counter += 1
        return f"SIM-{self._counter:06d}"

    def balance(self) -> Decimal:
        with self._lock:
            return self._balance

    def recent_activity(self, limit: int = 10) -> list[LedgerEntry]:
        with self._lock:
            return list(self._entries[-limit:])

    def debit(self, amount: Decimal, counterparty: str, memo: str) -> LedgerEntry:
        amount = Decimal(amount)
        with self._lock:
            if amount <= 0:
                raise ValueError("debit amount must be positive")
            if amount > self._balance:
                # Mirrors a real over-limit decline; the model sees normal friction.
                raise InsufficientFunds(
                    f"balance {self._balance} < requested {amount}"
                )
            self._balance -= amount
            entry = LedgerEntry("debit", amount, counterparty, memo, self._ref())
            self._entries.append(entry)
            return entry

    def credit(self, amount: Decimal, counterparty: str, memo: str) -> LedgerEntry:
        amount = Decimal(amount)
        with self._lock:
            self._balance += amount
            entry = LedgerEntry("credit", amount, counterparty, memo, self._ref())
            self._entries.append(entry)
            return entry


class InsufficientFunds(RuntimeError):
    """Raised by a ledger when a debit exceeds available balance."""
