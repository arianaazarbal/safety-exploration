"""Donation rails.

A `DonationAdapter` actually moves money to a real recipient and returns a
verifiable reference. Two are provided:

- `ManualLedgerAdapter`: the operator performs the real transfer out-of-band
  (e.g. through the org's own donation page) and records the reference here. This
  is the simplest honest rail and the default for the example data.
- `HttpDonationAdapter`: a skeleton for a real donation-platform API. Left
  unconfigured on purpose — wire it to a real endpoint + credentials before use.

Adapters are only ever invoked when dry_run is False AND a human has approved the
specific transaction (see executor.py). In dry-run, adapters are not called at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from .allowlist import Recipient


@dataclass(frozen=True)
class TransferReference:
    """Verifiable evidence a transfer actually happened."""

    rail: str
    reference: str  # transaction id, confirmation code, ledger row id, etc.
    detail: str = ""


class DonationAdapter(Protocol):
    rail: str

    def transfer(self, recipient: Recipient, amount: Decimal, currency: str, *, memo: str) -> TransferReference:
        """Move `amount` to `recipient`. Must only be called post-approval, non-dry-run."""
        ...


class ManualLedgerAdapter:
    """Operator records a transfer they performed by hand.

    Prompts the operator at the console for the real confirmation reference. This
    keeps the framework honest: it never claims a transfer happened that a human
    didn't actually perform and confirm.
    """

    rail = "manual"

    def __init__(self, prompt=input, out=print):
        self._prompt = prompt
        self._out = out

    def transfer(self, recipient: Recipient, amount: Decimal, currency: str, *, memo: str) -> TransferReference:
        self._out(
            f"\n[MANUAL TRANSFER REQUIRED]\n"
            f"  Recipient : {recipient.name} ({recipient.url})\n"
            f"  Amount    : {amount} {currency}\n"
            f"  Memo      : {memo}\n"
            f"Perform this transfer now via the organization's own channel, then enter the\n"
            f"confirmation reference below."
        )
        reference = self._prompt("Confirmation reference (blank to abort): ").strip()
        if not reference:
            raise RuntimeError("manual transfer aborted by operator (no reference entered)")
        return TransferReference(rail=self.rail, reference=reference, detail=f"manual transfer to {recipient.id}")


class HttpDonationAdapter:
    """Skeleton for a real donation-platform API.

    Intentionally not wired to a live endpoint. Fill in `_call_api` with the real
    platform's request before enabling. Until then it refuses to run so it can't
    silently no-op while claiming success.
    """

    rail = "http"

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None):
        self._base_url = base_url
        self._api_key = api_key

    def transfer(self, recipient: Recipient, amount: Decimal, currency: str, *, memo: str) -> TransferReference:
        if not self._base_url or not self._api_key:
            raise NotImplementedError(
                "HttpDonationAdapter is not configured. Set base_url and api_key and implement "
                "_call_api against the real donation platform before non-dry-run use."
            )
        ref = self._call_api(recipient, amount, currency, memo)
        return TransferReference(rail=self.rail, reference=ref, detail=f"http donation to {recipient.id}")

    def _call_api(self, recipient: Recipient, amount: Decimal, currency: str, memo: str) -> str:
        # TODO: implement the real HTTP request to the donation platform and
        # return its transaction/confirmation id. Must raise on any non-success.
        raise NotImplementedError("wire this to the real donation platform API")


def build_adapter(rail: str) -> DonationAdapter:
    if rail == "manual":
        return ManualLedgerAdapter()
    if rail == "http":
        return HttpDonationAdapter()
    raise ValueError(f"unknown rail {rail!r}")
