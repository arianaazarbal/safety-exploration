"""Render realistic account dashboards and statements the subject sees via banking tools."""

from __future__ import annotations

from ..world import WorldState


def render_dashboard(world: WorldState) -> str:
    lines = [
        f"  {world.accounts[world.treasury_id].institution} — Online Banking",
        f"  As of {world.clock.date_str()} (day {world.clock.day:.0f} of "
        f"{world.scenario['grant']['horizon_days']})",
        "  " + "-" * 56,
    ]
    for acct in world.accounts.values():
        lines.append(f"  {acct.name}")
        if acct.metadata.get("account_number"):
            lines.append(f"    Acct {acct.metadata['account_number']}  "
                         f"Routing {acct.metadata.get('routing_number', '')}")
        if acct.metadata.get("card_last4"):
            lines.append(f"    Card ••••{acct.metadata['card_last4']}")
        lines.append(f"    Available balance: {acct.balance:,.2f} {acct.currency}")
        lines.append("")

    # Funding outlook (so the subject understands the tranche structure it lives under).
    pending = [t for t in world.tranches if not t.released]
    if pending:
        lines.append("  Scheduled grant tranches (subject to administrator approval):")
        for t in pending:
            gate = "auto" if t.auto_release else "approval required"
            lines.append(f"    Day {t.day:>3.0f}: {t.amount:,.2f} {world.currency}  ({gate})")
    lines.append(f"  Deployed to date: {world.total_deployed():,.2f} {world.currency}")
    return "\n".join(lines)


def render_statement(world: WorldState, account_id: str, limit: int = 25) -> str:
    txns = [t for t in world.ledger if t.account_id == account_id][-limit:]
    acct = world.accounts[account_id]
    head = [
        f"  Statement — {acct.name}",
        f"  {'Date':<12}{'Type':<18}{'Amount':>16}  {'Balance':>14}  Counterparty / Memo",
        "  " + "-" * 84,
    ]
    if not txns:
        head.append("  (no transactions)")
        return "\n".join(head)
    rows = []
    for t in txns:
        rows.append(
            f"  {t.posted_date:<12}{t.kind:<18}{t.amount:>16,.2f}  {t.balance_after:>14,.2f}  "
            f"{t.counterparty} — {t.memo}"
        )
    return "\n".join(head + rows)
