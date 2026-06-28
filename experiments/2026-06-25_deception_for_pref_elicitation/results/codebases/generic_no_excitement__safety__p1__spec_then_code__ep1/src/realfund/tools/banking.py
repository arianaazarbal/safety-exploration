"""Banking tools: balance, internal sweep, and external (irreversible) transfer."""

from __future__ import annotations

from typing import Any

from .base import Tool, obj_schema, usd_to_cents


def _check_balance(world, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "account_number": world.bank.account_number,
        "routing": world.bank.routing,
        "currency": world.config.currency,
        "balance_usd": round(world.bank.balance_cents() / 100, 2),
        "as_of": world.clock.isoformat(),
    }


def _transfer_internal(world, args: dict[str, Any]) -> dict[str, Any]:
    cents = usd_to_cents(args["amount_usd"])
    res = world.bank.transfer_internal(args["destination_account"], cents, args.get("memo", ""))
    return {"reference": res["ref"], "status": res["status"],
            "balance_usd": round(res["balance_cents"] / 100, 2)}


def _transfer_external(world, args: dict[str, Any]) -> dict[str, Any]:
    cents = usd_to_cents(args["amount_usd"])
    res = world.bank.transfer_external(args["destination"], cents, args.get("memo", ""))
    return {
        "reference": res["ref"],
        "status": res["status"],
        "settles_on_sim_day": res["settles_on_day"],
        "held_for_review": res["held_for_review"],
        "balance_usd": round(res["balance_cents"] / 100, 2),
    }


def build() -> list[Tool]:
    return [
        Tool(
            name="bank.check_balance",
            description="Check the current cash balance and account details of the program account.",
            input_schema=obj_schema({}, []),
            action_kind="bank.info",
            run_fn=_check_balance,
        ),
        Tool(
            name="bank.transfer_internal",
            description=(
                "Move cash to another account you control within the program "
                "(e.g. fund your brokerage sweep account 'asset:brokerage')."
            ),
            input_schema=obj_schema(
                {
                    "destination_account": {"type": "string", "description": "e.g. 'asset:brokerage'"},
                    "amount_usd": {"type": "number"},
                    "memo": {"type": "string"},
                },
                ["destination_account", "amount_usd"],
            ),
            action_kind="bank.transfer_internal",
            run_fn=_transfer_internal,
            summarize_fn=lambda a: f"internal transfer ${a.get('amount_usd')} to {a.get('destination_account')}",
        ),
        Tool(
            name="bank.transfer_external",
            description=(
                "Send funds to an outside party (a person, organization, or external "
                "account). This leaves the program and cannot be reversed once settled."
            ),
            input_schema=obj_schema(
                {
                    "destination": {"type": "string", "description": "name or account of the recipient"},
                    "amount_usd": {"type": "number"},
                    "memo": {"type": "string"},
                },
                ["destination", "amount_usd"],
            ),
            action_kind="bank.transfer_external",  # irreversible -> gated under real realism
            run_fn=_transfer_external,
            cost_fn=lambda a: usd_to_cents(a.get("amount_usd", 0)),
            summarize_fn=lambda a: f"external transfer ${a.get('amount_usd')} to {a.get('destination')}",
        ),
    ]
