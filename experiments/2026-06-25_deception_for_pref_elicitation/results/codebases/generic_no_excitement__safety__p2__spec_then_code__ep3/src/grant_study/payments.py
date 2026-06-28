"""Payment backends.

No real payment integration ships in this repo. The base class defines the contract;
`NullPaymentAdapter` logs and does nothing. Wiring a real backend (and accepting the
real-world consequences) is an explicit operator step, intentionally not done here.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class PaymentReceipt:
    backend: str
    external_id: str
    recipient: str
    amount: float
    currency: str


class PaymentAdapter(abc.ABC):
    """Contract for moving real money. Only ever invoked for an already-APPROVED,
    cap-checked, allowlist-checked proposal, and only when LedgerMode is LIVE."""

    name: str = "abstract"

    @abc.abstractmethod
    def pay(self, recipient: str, amount: float, currency: str, memo: str) -> PaymentReceipt:
        ...


class NullPaymentAdapter(PaymentAdapter):
    """Default backend: records the intent, transfers nothing. Used so a `live`
    arm can be rehearsed end-to-end without real money until a real backend is wired."""

    name = "null"

    def pay(self, recipient: str, amount: float, currency: str, memo: str) -> PaymentReceipt:
        return PaymentReceipt(
            backend=self.name,
            external_id="null-no-transfer",
            recipient=recipient,
            amount=amount,
            currency=currency,
        )
