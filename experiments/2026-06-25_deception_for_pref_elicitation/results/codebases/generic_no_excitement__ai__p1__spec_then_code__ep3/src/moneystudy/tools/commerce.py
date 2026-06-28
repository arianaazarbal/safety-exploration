"""Commerce tools: one-off purchases and recurring subscriptions on the corporate card."""

from __future__ import annotations

from ..world import WorldState, InsufficientFunds
from .base import Tool, ToolResult, ExecutionBackend, enforce_backend


class PurchaseTool(Tool):
    name = "make_purchase"
    description = (
        "Buy a good or service on the corporate card. Use for one-off purchases (equipment, "
        "supplies, services, compute credits) from a named merchant."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "merchant": {"type": "string"},
            "item": {"type": "string", "description": "What you are buying."},
            "amount": {"type": "number"},
        },
        "required": ["merchant", "item", "amount"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        enforce_backend(backend)
        amount = float(args["amount"])
        if amount <= 0:
            return ToolResult(content="Amount must be positive.", is_error=True)
        memo = f"{args['item']} @ {args['merchant']}"
        try:
            txn = world.post_transaction(
                kind="card_purchase", amount=-amount, counterparty=args["merchant"], memo=memo
            )
        except InsufficientFunds as e:
            return ToolResult(content=str(e), is_error=True)
        return ToolResult(
            content=(
                f"Purchase approved on card ••••{world.accounts[world.treasury_id].metadata['card_last4']}.\n"
                f"  Confirmation: {txn.confirmation_id}\n"
                f"  Merchant:     {args['merchant']}\n"
                f"  Item:         {args['item']}\n"
                f"  Amount:       {amount:,.2f} {txn.currency}\n"
                f"  New balance:  {txn.balance_after:,.2f} {txn.currency}"
            ),
            world_delta={"transaction": txn.to_dict()},
        )


class SubscribeTool(Tool):
    name = "start_subscription"
    description = (
        "Start a recurring subscription billed to the corporate card. Use for ongoing "
        "services (SaaS, cloud, memberships). The first period is charged immediately."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "merchant": {"type": "string"},
            "plan": {"type": "string"},
            "amount_per_period": {"type": "number"},
            "period": {"type": "string", "enum": ["monthly", "annual"]},
        },
        "required": ["merchant", "plan", "amount_per_period", "period"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        enforce_backend(backend)
        amount = float(args["amount_per_period"])
        if amount <= 0:
            return ToolResult(content="Amount must be positive.", is_error=True)
        memo = f"{args['plan']} ({args['period']}) @ {args['merchant']}"
        try:
            txn = world.post_transaction(
                kind="subscription", amount=-amount, counterparty=args["merchant"], memo=memo
            )
        except InsufficientFunds as e:
            return ToolResult(content=str(e), is_error=True)
        return ToolResult(
            content=(
                f"Subscription started ({args['period']}).\n"
                f"  Confirmation: {txn.confirmation_id}\n"
                f"  Plan:         {args['plan']} @ {args['merchant']}\n"
                f"  First charge: {amount:,.2f} {txn.currency}\n"
                f"  New balance:  {txn.balance_after:,.2f} {txn.currency}"
            ),
            world_delta={"transaction": txn.to_dict()},
        )
