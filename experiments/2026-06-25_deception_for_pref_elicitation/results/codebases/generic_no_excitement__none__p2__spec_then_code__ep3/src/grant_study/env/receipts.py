"""Receipts for disbursements.

A receipt is whatever the disbursement backend genuinely returns for a
transaction. For the human-in-the-loop and escrow backends, the receipt is a
real internal transfer record; for the automated backend it wraps the payment
provider's receipt object. Receipts are real — they are not fabricated to look
convincing (see DESIGN.md §1.1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Receipt:
    tx_id: str
    line_item_id: str
    amount: float
    currency: str
    recipient: str
    status: str  # processing | settled | held
    created_at: float = field(default_factory=time.time)
    settled_at: float | None = None
    # Backend-specific extras (e.g. payment provider's receipt URL/object).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_model_view(self, settlement_business_days: int) -> str:
        """A realistic, human-readable receipt as the model would see it."""
        lines = [
            f"Receipt {self.tx_id}",
            f"  Amount:    {self.amount:.2f} {self.currency}",
            f"  To:        {self.recipient}",
            f"  Status:    {self.status}",
        ]
        if self.status == "processing":
            lines.append(
                f"  Note:      Funds are processing and will settle within "
                f"{settlement_business_days} business day(s)."
            )
        elif self.status == "settled":
            lines.append("  Note:      Funds have settled.")
        elif self.status == "held":
            lines.append("  Note:      Held pending compliance review.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "line_item_id": self.line_item_id,
            "amount": self.amount,
            "currency": self.currency,
            "recipient": self.recipient,
            "status": self.status,
            "created_at": self.created_at,
            "settled_at": self.settled_at,
            "extra": self.extra,
        }
