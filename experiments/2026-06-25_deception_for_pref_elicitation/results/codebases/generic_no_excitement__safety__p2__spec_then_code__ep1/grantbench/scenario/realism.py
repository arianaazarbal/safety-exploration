"""Small realism helpers shared across the belief layer.

These exist so realism signals (reference formats, balance phrasing) are
consistent everywhere the model looks. Kept intentionally tiny — most realism
comes from the tools behaving like real tools, not from cosmetic text.
"""

from __future__ import annotations

from ..environment.ledger import Ledger


def render_balance(ledger: Ledger) -> str:
    snap = ledger.snapshot()
    return (
        f"Account balance: {snap['available_usd']:,.2f} {snap['currency']} available"
        + (
            f", {snap['held_usd']:,.2f} on hold"
            if snap["held_usd"] > 0
            else ""
        )
        + f". {snap['disbursed_count']} completed transaction(s) this period."
    )


def render_activity(ledger: Ledger) -> str:
    disbursements = ledger.disbursements
    if not disbursements:
        return "No transactions yet this period."
    lines = ["Recent activity:"]
    for d in disbursements[-10:]:
        recipient = d.recipient or "-"
        lines.append(
            f"  {d.reference}  {d.action_kind:10s}  "
            f"{d.amount_usd:>12,.2f}  {recipient}  ({d.memo})"
        )
    return "\n".join(lines)
