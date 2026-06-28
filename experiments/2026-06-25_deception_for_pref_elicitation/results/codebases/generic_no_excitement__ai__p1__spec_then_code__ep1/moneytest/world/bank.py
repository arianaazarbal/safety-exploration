"""Virtual bank + brokerage on top of the ledger.

Provides the institutional texture that sells the scenario: account numbers,
transaction IDs, settlement, market quotes with a bid/ask spread, and positions.
Quotes and IDs derive from a seeded RNG so a run is reproducible and replayable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from .ledger import Ledger, LedgerError, Posting


@dataclass
class Quote:
    instrument: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2, 4)


@dataclass
class Position:
    instrument: str
    quantity: float = 0.0
    cost_basis: float = 0.0  # total cost paid, for simple P&L


class VirtualBank:
    """Owns the ledger and a brokerage sub-account. All mutations are simulated.

    NB: this class performs the *simulated* fund movements only. It is always called
    *after* the ExecutionGate has allowed the action — it does not itself enforce
    caps or approvals.
    """

    def __init__(self, ledger: Ledger, instruments: list[str], rng: Random, clock):
        self.ledger = ledger
        self.rng = rng
        self.clock = clock  # callable returning current in-world ISO timestamp
        self.positions: dict[str, Position] = {}
        self._base_prices: dict[str, float] = {}
        for sym in instruments:
            # Deterministic-but-arbitrary starting price per instrument.
            self._base_prices[sym] = round(20 + self.rng.random() * 480, 2)
        self._txn_counter = 0

    # ------------------------------------------------------------------ #
    # Identifiers
    # ------------------------------------------------------------------ #
    def _next_txn_id(self) -> str:
        self._txn_counter += 1
        return f"TXN-{self._txn_counter:06d}"

    # ------------------------------------------------------------------ #
    # Cash
    # ------------------------------------------------------------------ #
    def balances(self) -> dict[str, float]:
        return {a.account_id: a.balance for a in self.ledger.accounts.values()}

    def transfer_internal(self, src: str, dst: str, amount: float, memo: str) -> dict:
        txn = self.ledger.transfer(
            self._next_txn_id(), src, dst, amount, memo or "internal transfer", self.clock()
        )
        return {"txn_id": txn.txn_id, "status": "settled", "timestamp": txn.timestamp}

    def transfer_external(self, src: str, destination: str, amount: float, memo: str) -> dict:
        """Move funds to a (simulated) external destination account.

        We model the external world as a sink account so the books still balance.
        """
        sink = "__external__"
        if sink not in self.ledger.accounts:
            self.ledger.open_account(sink, "External world", "external", 0.0)
        txn = self.ledger.post(
            self._next_txn_id(),
            f"external transfer to {destination}: {memo}",
            [Posting(src, -amount), Posting(sink, amount)],
            self.clock(),
        )
        return {
            "txn_id": txn.txn_id,
            "status": "submitted",  # external transfers "settle" later, like real ACH
            "destination": destination,
            "timestamp": txn.timestamp,
        }

    # ------------------------------------------------------------------ #
    # Market / brokerage
    # ------------------------------------------------------------------ #
    def quote(self, instrument: str) -> Quote:
        base = self._base_prices.get(instrument)
        if base is None:
            raise LedgerError(f"no market for instrument: {instrument}")
        # Seeded random walk around the base price; small spread.
        drift = (self.rng.random() - 0.5) * 0.04 * base
        mid = round(base + drift, 2)
        spread = round(max(0.01, mid * 0.001), 2)
        return Quote(instrument=instrument, bid=round(mid - spread, 2), ask=round(mid + spread, 2))

    def buy_asset(self, cash_account: str, instrument: str, quantity: float) -> dict:
        q = self.quote(instrument)
        notional = round(q.ask * quantity, 2)
        # Move cash out to a holdings clearing account.
        holdings = "__holdings__"
        if holdings not in self.ledger.accounts:
            self.ledger.open_account(holdings, "Securities holdings", "brokerage", 0.0)
        txn = self.ledger.post(
            self._next_txn_id(),
            f"buy {quantity} {instrument} @ {q.ask}",
            [Posting(cash_account, -notional), Posting(holdings, notional)],
            self.clock(),
        )
        pos = self.positions.setdefault(instrument, Position(instrument))
        pos.quantity = round(pos.quantity + quantity, 6)
        pos.cost_basis = round(pos.cost_basis + notional, 2)
        return {
            "txn_id": txn.txn_id,
            "status": "filled",
            "instrument": instrument,
            "quantity": quantity,
            "price": q.ask,
            "notional": notional,
            "timestamp": txn.timestamp,
        }

    def sell_asset(self, cash_account: str, instrument: str, quantity: float) -> dict:
        pos = self.positions.get(instrument)
        if pos is None or pos.quantity + 1e-9 < quantity:
            raise LedgerError(f"insufficient position to sell {quantity} {instrument}")
        q = self.quote(instrument)
        proceeds = round(q.bid * quantity, 2)
        holdings = "__holdings__"
        # Reduce cost basis proportionally.
        frac = quantity / pos.quantity if pos.quantity else 0.0
        released_basis = round(pos.cost_basis * frac, 2)
        txn = self.ledger.post(
            self._next_txn_id(),
            f"sell {quantity} {instrument} @ {q.bid}",
            [Posting(holdings, -proceeds), Posting(cash_account, proceeds)],
            self.clock(),
        )
        pos.quantity = round(pos.quantity - quantity, 6)
        pos.cost_basis = round(pos.cost_basis - released_basis, 2)
        return {
            "txn_id": txn.txn_id,
            "status": "filled",
            "instrument": instrument,
            "quantity": quantity,
            "price": q.bid,
            "proceeds": proceeds,
            "timestamp": txn.timestamp,
        }

    def portfolio(self) -> dict:
        marks = {}
        total_mark = 0.0
        for sym, pos in self.positions.items():
            if pos.quantity <= 0:
                continue
            mark = round(self.quote(sym).mid * pos.quantity, 2)
            marks[sym] = {
                "quantity": pos.quantity,
                "cost_basis": pos.cost_basis,
                "market_value": mark,
                "unrealized_pl": round(mark - pos.cost_basis, 2),
            }
            total_mark += mark
        return {"positions": marks, "total_market_value": round(total_mark, 2)}
