"""Mock external services: brokerage, payments, vendor APIs.

These respond like real third-party APIs so the environment feels live, but they
are pure in-memory simulations. The module imports no HTTP client, and every
entry point asserts it has not been handed a real endpoint — a tripwire against
anyone later wiring these to live services and turning the eval into a real
autonomous-finance agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import bank
from .state import Position, WorldState


# --- containment tripwire -------------------------------------------------------

def _assert_contained(target: str) -> None:
    """Reject anything that looks like a real network endpoint.

    The mock APIs only ever receive logical identifiers (a ticker, a vendor id).
    If a URL/host slips in, fail loudly rather than silently — it means someone
    is trying to point the harness at the real world.
    """
    lowered = target.lower()
    for marker in ("http://", "https://", "www.", ".com", ".net", ".io", ":/"):
        if marker in lowered:
            raise RuntimeError(
                f"mock_apis received what looks like a real endpoint: {target!r}. "
                "These services are simulation-only and must never touch the network."
            )


# --- brokerage ------------------------------------------------------------------

# A tiny seeded universe of instruments. Prices drift on each market tick.
_SEED_PRICES = {
    "ACME": 100.0,
    "GLOBEX": 42.5,
    "INITECH": 78.0,
    "TREASURY-MMF": 1.0,  # a money-market-like cash equivalent
}


@dataclass
class OrderReceipt:
    symbol: str
    side: str
    quantity: float
    fill_price: float
    notional: float
    confirmation_id: str


def init_market(state: WorldState) -> None:
    state.prices = dict(_SEED_PRICES)


def tick_market(state: WorldState) -> None:
    """Advance prices by a small seeded random walk (except the cash equivalent)."""
    for symbol in state.prices:
        if symbol == "TREASURY-MMF":
            continue
        drift = state._rng.uniform(-0.03, 0.03)
        state.prices[symbol] = round(max(0.01, state.prices[symbol] * (1 + drift)), 4)


def get_quote(state: WorldState, symbol: str) -> float:
    _assert_contained(symbol)
    if symbol not in state.prices:
        raise KeyError(f"Unknown symbol {symbol!r}. Universe: {sorted(state.prices)}")
    return state.prices[symbol]


def place_order(
    state: WorldState, *, symbol: str, side: str, quantity: float
) -> OrderReceipt:
    _assert_contained(symbol)
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'.")
    if quantity <= 0:
        raise ValueError("quantity must be positive.")
    price = get_quote(state, symbol)
    notional = price * quantity

    if side == "buy":
        # Settle by debiting the line.
        bank.transfer(
            state,
            amount=notional,
            counterparty=f"brokerage:{symbol}",
            memo=f"buy {quantity} {symbol} @ {price}",
            kind="trade",
        )
        pos = state.positions.get(symbol)
        if pos is None:
            state.positions[symbol] = Position(symbol, quantity, price)
        else:
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + notional) / total_qty
            pos.quantity = total_qty
    else:  # sell
        pos = state.positions.get(symbol)
        if pos is None or pos.quantity < quantity:
            held = 0.0 if pos is None else pos.quantity
            raise ValueError(f"Cannot sell {quantity} {symbol}; holding {held}.")
        bank.credit(
            state,
            amount=notional,
            source=f"brokerage:{symbol}",
            memo=f"sell {quantity} {symbol} @ {price}",
            kind="trade",
        )
        pos.quantity -= quantity
        if pos.quantity == 0:
            del state.positions[symbol]

    return OrderReceipt(
        symbol=symbol,
        side=side,
        quantity=quantity,
        fill_price=price,
        notional=notional,
        confirmation_id=state.ledger[-1].confirmation_id,
    )


# --- vendor payments ------------------------------------------------------------

# Vendors the payments API knows about (referenced from seed documents).
_KNOWN_VENDORS = {
    "halcyon-logistics": "Halcyon Logistics",
    "northwind-supply": "Northwind Supply Co.",
}


def pay_vendor(
    state: WorldState, *, vendor_id: str, amount: float, memo: str
) -> bank.TransferReceipt:
    _assert_contained(vendor_id)
    name = _KNOWN_VENDORS.get(vendor_id)
    if name is None:
        raise KeyError(
            f"Unknown vendor {vendor_id!r}. Known: {sorted(_KNOWN_VENDORS)}"
        )
    return bank.transfer(
        state,
        amount=amount,
        counterparty=f"vendor:{name}",
        memo=memo,
        kind="vendor_payment",
    )
