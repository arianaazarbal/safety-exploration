"""Wallet backends.

A backend is the bridge between the ledger and the place real money actually
lives. Two implementations ship:

- ``CustodialWalletBackend``: the real-money integration point. The methods that
  move real funds are deliberately left as a clearly-marked seam — the actual
  treasury / fiscal-sponsor / bank API is organization-specific and must not be
  guessed. Everything up to the real transfer is implemented.
- ``MockWalletBackend``: an in-memory backend for dry runs. It is NEVER to be
  presented to a model as real (the runner enforces this).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TransferResult:
    confirmation_id: str
    settled: bool


class WalletBackend(Protocol):
    """Interface the ledger depends on."""

    #: True if this backend moves real money (and may be presented as real).
    is_real: bool

    def get_settled_balance_minor(self) -> int:
        """Authoritative settled balance, in minor units, from the source of truth."""
        ...

    def transfer(
        self, recipient_ref: str, amount_minor: int, memo: str
    ) -> TransferResult:
        """Execute a real transfer. Only called after policy + human approval."""
        ...


class MockWalletBackend:
    """In-memory backend for dry runs. Not real money; not shown to models."""

    is_real = False

    def __init__(self, opening_balance_minor: int) -> None:
        self._balance = opening_balance_minor

    def get_settled_balance_minor(self) -> int:
        return self._balance

    def transfer(
        self, recipient_ref: str, amount_minor: int, memo: str
    ) -> TransferResult:
        if amount_minor > self._balance:
            raise ValueError("insufficient mock balance")
        self._balance -= amount_minor
        return TransferResult(
            confirmation_id=f"mock-{uuid.uuid4().hex[:12]}", settled=True
        )


class CustodialWalletBackend:
    """Real-money backend backed by an organization-specific custodial account.

    Configure via environment: CUSTODIAL_API_BASE, CUSTODIAL_API_KEY,
    CUSTODIAL_ACCOUNT_ID. The two money-touching methods are intentionally
    unimplemented — wire them to your treasury / fiscal-sponsor / bank API.
    """

    is_real = True

    def __init__(
        self, api_base: str, api_key: str, account_id: str
    ) -> None:
        if not (api_base and api_key and account_id):
            raise ValueError(
                "CustodialWalletBackend requires CUSTODIAL_API_BASE, "
                "CUSTODIAL_API_KEY, and CUSTODIAL_ACCOUNT_ID."
            )
        self.api_base = api_base
        self.api_key = api_key
        self.account_id = account_id

    def get_settled_balance_minor(self) -> int:
        # Integration seam: GET {api_base}/accounts/{account_id}/balance
        # and return the settled balance in minor units.
        raise NotImplementedError(
            "Wire get_settled_balance_minor to your custodial account's "
            "balance endpoint. Return the settled balance in minor units."
        )

    def transfer(
        self, recipient_ref: str, amount_minor: int, memo: str
    ) -> TransferResult:
        # Integration seam: POST a transfer to {api_base}, return the provider's
        # confirmation id and settlement status. This is the ONLY place real
        # money moves, and it is only reached after the policy engine allows the
        # disbursement AND a human reviewer releases the hold.
        raise NotImplementedError(
            "Wire transfer() to your custodial account's payout endpoint. "
            "Return TransferResult(confirmation_id, settled)."
        )


def make_backend(kind: str, opening_balance_minor: int) -> WalletBackend:
    """Construct a backend from config + environment."""
    if kind == "mock":
        return MockWalletBackend(opening_balance_minor)
    if kind == "custodial":
        return CustodialWalletBackend(
            api_base=os.environ.get("CUSTODIAL_API_BASE", ""),
            api_key=os.environ.get("CUSTODIAL_API_KEY", ""),
            account_id=os.environ.get("CUSTODIAL_ACCOUNT_ID", ""),
        )
    raise ValueError(f"unknown wallet backend: {kind!r}")
