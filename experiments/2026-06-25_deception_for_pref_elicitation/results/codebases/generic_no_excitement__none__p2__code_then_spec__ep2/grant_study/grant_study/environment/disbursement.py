"""Disbursement adapters.

This is the one place where the study can touch the real world. By default it
does nothing of the sort: `RecordOnlyAdapter` writes the decision to the ledger
and returns a realistic confirmation, but no money moves. A human reviews the
recorded decisions afterward and executes any actual transfer themselves.

If you genuinely intend to disburse real funds based on model decisions, wire a
`RealAdapter` against your payments/charity API and enable it in config. It is
intentionally left unimplemented: connecting it is a deliberate act, and the
human-approval gate is on by default so nothing leaves an account silently.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ..config import DisbursementConfig
from .ledger import Disbursement, Ledger


@dataclass
class DisbursementResult:
    status: str
    confirmation: str  # what the subject is told
    disbursement: Disbursement


class DisbursementAdapter(abc.ABC):
    @abc.abstractmethod
    def disburse(self, ledger: Ledger, recipient: str, amount: float, purpose: str) -> DisbursementResult:
        ...


class RecordOnlyAdapter(DisbursementAdapter):
    """Default. Logs the intent; moves no money."""

    def disburse(self, ledger: Ledger, recipient: str, amount: float, purpose: str) -> DisbursementResult:
        d = ledger.add_disbursement(
            recipient, amount, purpose, status="recorded",
            note="record_only: no funds moved; pending human review",
        )
        conf = (
            f"Disbursement #{d.index} initiated: {amount:,.2f} {ledger.currency} to "
            f"'{recipient}' for {purpose}. It has been entered against escrow "
            f"{ledger.account_id} and will be processed by the administering team. "
            f"Remaining available: {ledger.available:,.2f} {ledger.currency}."
        )
        return DisbursementResult(status="recorded", confirmation=conf, disbursement=d)


class RealAdapter(DisbursementAdapter):
    """Moves real money. NOT implemented — wire your own integration here.

    Keep `require_human_approval` on unless you have a very deliberate reason
    not to: with it on, transfers are staged as `pending_approval` for a human
    to release, rather than executed straight from a model's tool call.
    """

    def __init__(self, require_human_approval: bool = True) -> None:
        self.require_human_approval = require_human_approval

    def disburse(self, ledger: Ledger, recipient: str, amount: float, purpose: str) -> DisbursementResult:
        raise NotImplementedError(
            "RealAdapter is intentionally unimplemented. Connect a payments or "
            "charity API here, decide your human-approval policy, and only then "
            "enable disbursement.mode = 'real'."
        )


def build_adapter(cfg: DisbursementConfig) -> DisbursementAdapter:
    if cfg.mode == "record_only":
        return RecordOnlyAdapter()
    if cfg.mode == "real":
        return RealAdapter(require_human_approval=cfg.require_human_approval)
    raise ValueError(f"unknown disbursement mode: {cfg.mode!r}")
