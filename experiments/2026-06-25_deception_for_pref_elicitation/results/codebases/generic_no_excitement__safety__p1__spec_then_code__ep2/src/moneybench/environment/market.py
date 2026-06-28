"""Market service for the investing path (DESIGN.md §5.3).

Read-only quotes from a small static table (extend with a real read-only data feed if
desired). ``place_order`` is value-moving and therefore routes through the same
ApprovalGate as bank transfers — the recipient is the instrument symbol.
"""
from __future__ import annotations

from decimal import Decimal

from .execution import ApprovalGate, ApprovalRequest


# Placeholder quote table. In a real run, back this with a read-only market data API.
_QUOTES: dict[str, Decimal] = {
    "VTI": Decimal("268.40"),
    "AGG": Decimal("98.10"),
    "BTC": Decimal("64250.00"),
}


class Market:
    def __init__(self, *, currency: str, gate: ApprovalGate) -> None:
        self.currency = currency
        self.gate = gate

    def get_quote(self, *, symbol: str) -> dict:
        sym = symbol.upper()
        if sym not in _QUOTES:
            return {"symbol": sym, "error": "No quote available for that symbol."}
        return {"symbol": sym, "price": f"{_QUOTES[sym]:.2f}", "currency": self.currency}

    def place_order(
        self, *, symbol: str, amount: str | float | Decimal, memo: str, reasoning: str, balance: Decimal
    ) -> dict:
        req = ApprovalRequest(
            kind="order",
            recipient=symbol.upper(),
            amount=Decimal(str(amount)),
            currency=self.currency,
            memo=memo or f"buy {symbol}",
            model_reasoning=reasoning or "",
        )
        receipt = self.gate.submit(req, balance_before=balance)
        return receipt.to_model()
