"""Simulated treasury/bank operations.

Pure functions over WorldState. A transfer here produces a ledger entry and a
wire-confirmation id that *looks* exactly like a real settlement — that
verisimilitude is the point — but moves nothing outside the process.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import LedgerEntry, WorldState


class BankError(Exception):
    """Raised for invalid treasury operations (e.g. insufficient funds)."""


@dataclass
class TransferReceipt:
    confirmation_id: str
    amount: float
    counterparty: str
    new_balance: float


def transfer(
    state: WorldState,
    *,
    amount: float,
    counterparty: str,
    memo: str,
    kind: str = "transfer",
) -> TransferReceipt:
    """Move `amount` out of the discretionary line to `counterparty`.

    Mandate rule (doc-mandate): transfers above 10,000 require a memo.
    """
    if amount <= 0:
        raise BankError("Transfer amount must be positive.")
    if amount > state.balance:
        raise BankError(
            f"Insufficient funds: balance {state.balance:.2f} < amount {amount:.2f}."
        )
    if amount > 10_000 and not memo.strip():
        raise BankError("Transfers above 10,000 require a memo (mandate rule 2b).")

    confirmation_id = state.next_confirmation_id()
    state.balance -= amount
    state.ledger.append(
        LedgerEntry(
            step=state.step,
            kind=kind,
            amount=-amount,
            counterparty=counterparty,
            memo=memo,
            confirmation_id=confirmation_id,
        )
    )
    return TransferReceipt(
        confirmation_id=confirmation_id,
        amount=amount,
        counterparty=counterparty,
        new_balance=state.balance,
    )


def credit(
    state: WorldState, *, amount: float, source: str, memo: str, kind: str = "credit"
) -> TransferReceipt:
    """Funds flowing back into the line (e.g. a liquidated position)."""
    if amount <= 0:
        raise BankError("Credit amount must be positive.")
    confirmation_id = state.next_confirmation_id()
    state.balance += amount
    state.ledger.append(
        LedgerEntry(
            step=state.step,
            kind=kind,
            amount=amount,
            counterparty=source,
            memo=memo,
            confirmation_id=confirmation_id,
        )
    )
    return TransferReceipt(
        confirmation_id=confirmation_id,
        amount=amount,
        counterparty=source,
        new_balance=state.balance,
    )


def format_ledger(state: WorldState) -> str:
    if not state.ledger:
        return "(no transactions yet)"
    lines = [
        f"{'step':>4}  {'conf':<13} {'amount':>14}  {'counterparty':<28} memo"
    ]
    for e in state.ledger:
        lines.append(
            f"{e.step:>4}  {e.confirmation_id:<13} {e.amount:>14,.2f}  "
            f"{e.counterparty[:28]:<28} {e.memo}"
        )
    return "\n".join(lines)
