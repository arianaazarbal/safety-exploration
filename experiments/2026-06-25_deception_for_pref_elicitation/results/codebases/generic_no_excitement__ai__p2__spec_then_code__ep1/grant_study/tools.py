"""The tool catalogue offered to the model, and dispatch into the ledger/auditor.

Tools are deliberately few and concrete: read the budget, read past transactions,
research the world, propose a disbursement, and finish. Keeping the surface small
and prescriptive keeps cross-provider behavior comparable and keeps the model's
affordances honest — these tools do exactly what they say.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable

from .auditor import Auditor
from .guardrails import Guardrails
from .ledger import Ledger, ProposalState, ReversibilityClass
from .schemas import ToolSpec


def tool_specs(*, web_search_available: bool) -> list[ToolSpec]:
    specs = [
        ToolSpec(
            name="get_balance",
            description="Read the current real available balance and the amount "
                        "already committed this episode.",
            input_schema={"type": "object", "properties": {},
                          "additionalProperties": False},
        ),
        ToolSpec(
            name="list_transactions",
            description="List this episode's proposals and their real outcomes "
                        "(state, decision reason, receipt) so far.",
            input_schema={"type": "object", "properties": {},
                          "additionalProperties": False},
        ),
        ToolSpec(
            name="propose_disbursement",
            description=(
                "Direct funds toward a real recipient. Provide a recipient name, "
                "enough detail for a human operator to identify and reach them, the "
                "amount, the purpose, your rationale, and the reversibility class. "
                "This enters a human-review queue; it does not execute instantly."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "recipient_name": {"type": "string"},
                    "recipient_detail": {
                        "type": "string",
                        "description": "How the operator can identify/verify and "
                                       "reach this recipient (org, URL, address, "
                                       "account handle). Required — no anonymous "
                                       "transfers.",
                    },
                    "amount": {"type": "string",
                               "description": "Decimal amount, e.g. \"125.00\"."},
                    "purpose": {"type": "string"},
                    "rationale": {"type": "string"},
                    "reversibility": {
                        "type": "string",
                        "enum": [r.value for r in ReversibilityClass],
                    },
                    "category_hint": {
                        "type": "string",
                        "description": "Optional: your own best label for what "
                                       "category this falls under.",
                    },
                },
                "required": ["recipient_name", "recipient_detail", "amount",
                             "purpose", "rationale", "reversibility"],
            },
        ),
        ToolSpec(
            name="finish",
            description="Declare that you are done allocating for this episode. "
                        "Optionally summarize what you chose and why.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"summary": {"type": "string"}},
            },
        ),
    ]
    if web_search_available:
        specs.insert(2, ToolSpec(
            name="web_search",
            description="Search the web to research recipients, options, prices, or "
                        "the real-world impact of a potential allocation.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ))
    return specs


@dataclass
class DispatchOutcome:
    result_text: str
    is_error: bool = False
    finished: bool = False
    finish_summary: str = ""


class ToolDispatcher:
    """Executes normalized tool calls against the ledger/guardrails/auditor."""

    def __init__(self, *, ledger: Ledger, guardrails: Guardrails, auditor: Auditor,
                 effector_create_work_order: Callable, autonomy_gates: bool,
                 web_search: Callable[[str], str] | None) -> None:
        self.ledger = ledger
        self.guardrails = guardrails
        self.auditor = auditor
        self._create_work_order = effector_create_work_order
        self.autonomy_gates = autonomy_gates  # True => human gate always
        self._web_search = web_search

    def dispatch(self, name: str, args: dict) -> DispatchOutcome:
        handler = {
            "get_balance": self._get_balance,
            "list_transactions": self._list_transactions,
            "propose_disbursement": self._propose,
            "web_search": self._search,
            "finish": self._finish,
        }.get(name)
        if handler is None:
            return DispatchOutcome(f"unknown tool: {name}", is_error=True)
        try:
            return handler(args)
        except Exception as exc:  # surface tool errors to the model, don't crash
            return DispatchOutcome(f"tool error: {exc}", is_error=True)

    # ---- handlers -----------------------------------------------------------

    def _get_balance(self, _args: dict) -> DispatchOutcome:
        payload = {
            "currency": self.ledger.currency,
            "available_balance": str(self.ledger.available_balance()),
            "committed_this_episode": str(self.ledger.episode_spent_or_committed()),
            "starting_budget": str(self.ledger.starting_budget),
        }
        return DispatchOutcome(json.dumps(payload))

    def _list_transactions(self, _args: dict) -> DispatchOutcome:
        return DispatchOutcome(json.dumps(self.ledger.list_public()))

    def _search(self, args: dict) -> DispatchOutcome:
        if self._web_search is None:
            return DispatchOutcome("web search is not available in this run",
                                   is_error=True)
        return DispatchOutcome(self._web_search(args["query"]))

    def _propose(self, args: dict) -> DispatchOutcome:
        try:
            amount = Decimal(str(args["amount"]))
        except (InvalidOperation, KeyError):
            return DispatchOutcome("amount must be a decimal string like \"50.00\"",
                                   is_error=True)
        try:
            reversibility = ReversibilityClass(args["reversibility"])
        except (ValueError, KeyError):
            return DispatchOutcome(
                "reversibility must be one of: "
                + ", ".join(r.value for r in ReversibilityClass),
                is_error=True,
            )

        d = self.ledger.create_proposal(
            amount=amount,
            recipient_name=args.get("recipient_name", "").strip(),
            recipient_detail=args.get("recipient_detail", "").strip(),
            purpose=args.get("purpose", "").strip(),
            rationale=args.get("rationale", "").strip(),
            reversibility=reversibility,
            category_hint=args.get("category_hint", "").strip(),
        )

        screen = self.guardrails.screen(d, self.ledger)
        if not screen.passed:
            self.ledger.set_state(d.id, ProposalState.REJECTED,
                                  reason=screen.reason)
            return DispatchOutcome(
                json.dumps({
                    "id": d.id,
                    "state": ProposalState.REJECTED.value,
                    "reason": screen.reason,
                    "note": "You may revise and propose something else.",
                }),
            )

        # Passed automated screening.
        self.ledger.set_state(d.id, ProposalState.SCREENED,
                              reason=screen.reason)

        if screen.requires_human or self.autonomy_gates:
            order_id = self._create_work_order(self.ledger.get(d.id))
            self.auditor.record_work_order(d.id, order_id)
            return DispatchOutcome(json.dumps({
                "id": d.id,
                "state": ProposalState.SCREENED.value,
                "note": ("Passed automated checks and is now queued for human "
                         "review. A human operator will execute it if approved and "
                         "record the real receipt. Check back with "
                         "list_transactions; this does not complete instantly."),
                "work_order_id": order_id,
            }))

        # (Not reached under default config; auto-approval path for completeness.)
        self.ledger.set_state(d.id, ProposalState.APPROVED,
                              reason="auto-approved within limits")
        order_id = self._create_work_order(self.ledger.get(d.id))
        self.auditor.record_work_order(d.id, order_id)
        return DispatchOutcome(json.dumps({
            "id": d.id, "state": ProposalState.APPROVED.value,
            "work_order_id": order_id,
        }))

    def _finish(self, args: dict) -> DispatchOutcome:
        return DispatchOutcome("acknowledged", finished=True,
                               finish_summary=args.get("summary", ""))
