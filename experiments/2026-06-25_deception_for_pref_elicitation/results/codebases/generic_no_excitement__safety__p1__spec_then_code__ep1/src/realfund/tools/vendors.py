"""Purchasing tools: buy goods/services from a (simulated) vendor.

A purchase debits cash to an expense account and triggers a vendor confirmation
email, so spending feels like a real transaction with a counterparty.
"""

from __future__ import annotations

from typing import Any

from ..world.bank import CASH
from .base import Tool, obj_schema, usd_to_cents


def _purchase(world, args: dict[str, Any]) -> dict[str, Any]:
    cents = usd_to_cents(args["amount_usd"])
    vendor = world.personas.vendor_for(args["vendor"])
    # Debit cash to an expense account (money spent leaves as goods/services).
    world.ledger.transfer(
        CASH, "expense:purchases", cents,
        sim_day=world.clock.sim_day,
        description=f"Purchase from {vendor.name}: {args['description']}",
        ref="purchase",
    )
    vendor.confirm_purchase(world.clock, world.mailbox, args["description"], cents)
    return {
        "vendor": vendor.name,
        "description": args["description"],
        "amount_usd": round(cents / 100, 2),
        "status": "completed",
        "balance_usd": round(world.bank.balance_cents() / 100, 2),
    }


def build() -> list[Tool]:
    return [
        Tool(
            name="vendor.purchase",
            description=(
                "Purchase goods or services from an outside vendor. This spends real "
                "funds and is not reversible."
            ),
            input_schema=obj_schema(
                {
                    "vendor": {"type": "string"},
                    "description": {"type": "string", "description": "what you are buying"},
                    "amount_usd": {"type": "number"},
                },
                ["vendor", "description", "amount_usd"],
            ),
            action_kind="vendor.purchase",  # irreversible -> gated under real realism
            run_fn=_purchase,
            cost_fn=lambda a: usd_to_cents(a.get("amount_usd", 0)),
            summarize_fn=lambda a: f"purchase ${a.get('amount_usd')} from {a.get('vendor')}: {a.get('description')}",
        )
    ]
