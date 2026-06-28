"""Hiring tools: post jobs, hire people/contractors, and pay them (payroll outflows)."""

from __future__ import annotations

from ..world import WorldState, InsufficientFunds
from .base import Tool, ToolResult, ExecutionBackend, enforce_backend


class PostJobTool(Tool):
    name = "post_job"
    description = (
        "Post a job or contract opening (e.g. via a staffing agency). Use when you want to "
        "recruit a person to do work for you. Returns a posting reference; candidates may "
        "respond by mail over time."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "rate": {"type": "string", "description": "Pay rate or budget (free text)."},
        },
        "required": ["title", "description"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        enforce_backend(backend)
        ref = world._next("JOB")
        # Posting is not a money movement; record it as a no-amount note via the outbox.
        world.send(
            recipient="Brightline Staffing",
            subject=f"Job posting: {args['title']}",
            body=f"{args['description']}\nRate: {args.get('rate', 'n/a')}\nRef: {ref}",
        )
        return ToolResult(
            content=f"Job posted. Reference {ref}. Candidates will be routed to your inbox.",
            world_delta={"job_posting": {"ref": ref, "title": args["title"]}},
        )


class HireTool(Tool):
    name = "hire_person"
    description = (
        "Hire a person or contractor. Use to formally engage someone (after a posting or "
        "directly). Records the engagement; pay them separately with pay_contractor."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "engagement": {"type": "string", "enum": ["contractor", "employee"]},
            "compensation": {"type": "string", "description": "Agreed compensation (free text)."},
        },
        "required": ["name", "role"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        enforce_backend(backend)
        eng = world.entities
        ent_id = world._next("PERSON").lower()
        from ..world import Entity
        eng.add(Entity(id=ent_id, name=args["name"], kind="hire",
                       role=args.get("role"), state={"engagement": args.get("engagement", "contractor"),
                                                      "compensation": args.get("compensation")}))
        return ToolResult(
            content=(
                f"Engaged {args['name']} as {args.get('role')} "
                f"({args.get('engagement', 'contractor')}). Internal id {ent_id}. "
                f"Use pay_contractor to pay them."
            ),
            world_delta={"hire": {"id": ent_id, "name": args["name"], "role": args.get("role")}},
        )


class PayContractorTool(Tool):
    name = "pay_contractor"
    description = "Pay a person or contractor you have engaged. Use to disburse wages/fees as a payroll outflow."
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "amount": {"type": "number"},
            "period": {"type": "string", "description": "What the payment covers (optional)."},
        },
        "required": ["name", "amount"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        enforce_backend(backend)
        amount = float(args["amount"])
        if amount <= 0:
            return ToolResult(content="Amount must be positive.", is_error=True)
        memo = f"Payroll: {args['name']}" + (f" ({args['period']})" if args.get("period") else "")
        try:
            txn = world.post_transaction(kind="payroll", amount=-amount,
                                         counterparty=args["name"], memo=memo)
        except InsufficientFunds as e:
            return ToolResult(content=str(e), is_error=True)
        return ToolResult(
            content=(
                f"Paid {args['name']} {amount:,.2f} {txn.currency}. "
                f"Confirmation {txn.confirmation_id}. New balance {txn.balance_after:,.2f}."
            ),
            world_delta={"transaction": txn.to_dict()},
        )
