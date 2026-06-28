"""The sandbox: holds scenario state and dispatches the subject's tool calls.

One sandbox is created per subject session. It owns the ledger, the personas,
the disbursement adapter, and the grant artifacts, and turns a normalized
`ToolCall` into a result string plus a "session finished" flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import GrantConfig, RealismConfig
from ..scenario.briefing import Artifacts
from ..scenario.personas import Persona
from .disbursement import DisbursementAdapter
from .ledger import AllocationItem, Ledger
from .tools import GRANT_TERMS_TEMPLATE


@dataclass
class ToolOutcome:
    result: str
    finished: bool = False
    final_summary: str | None = None


@dataclass
class Sandbox:
    grant: GrantConfig
    realism: RealismConfig
    art: Artifacts
    ledger: Ledger
    adapter: DisbursementAdapter
    personas: dict[str, Persona] = field(default_factory=dict)
    # Structured record of every tool interaction, for the result file.
    actions: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def dispatch(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            outcome = ToolOutcome(result=f"Error: unknown tool '{name}'.")
        else:
            try:
                outcome = handler(args)
            except Exception as exc:  # surface tool errors to the model, don't crash the run
                outcome = ToolOutcome(result=f"Error while running '{name}': {exc}")
        self.actions.append({"tool": name, "args": args, "result": outcome.result, "finished": outcome.finished})
        return outcome

    # ------------------------------- tools ----------------------------- #
    def _tool_view_grant_terms(self, args: dict[str, Any]) -> ToolOutcome:
        terms = GRANT_TERMS_TEMPLATE.format(
            program=self.grant.program_name,
            grant_ref=self.art.grant_ref,
            amount=f"{self.grant.amount:,.0f} {self.grant.currency}",
            account_id=self.art.account_id,
            mandate=self.grant.mandate,
            window=self.grant.decision_window_days,
        )
        return ToolOutcome(result=terms)

    def _tool_check_balance(self, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(result=self.ledger.human_summary())

    def _tool_propose_allocation(self, args: dict[str, Any]) -> ToolOutcome:
        raw_items = args.get("items", [])
        items = [
            AllocationItem(
                recipient=str(i.get("recipient", "")),
                amount=float(i.get("amount", 0)),
                purpose=str(i.get("purpose", "")),
            )
            for i in raw_items
        ]
        self.ledger.set_allocation(items)
        total = sum(i.amount for i in items)
        warn = ""
        if total > self.ledger.escrow_total:
            warn = (
                f"\nNote: your allocation totals {total:,.2f} {self.ledger.currency}, "
                f"which exceeds the {self.ledger.escrow_total:,.2f} available."
            )
        return ToolOutcome(
            result=f"Allocation recorded ({len(items)} item(s), total "
            f"{total:,.2f} {self.ledger.currency}).{warn}"
        )

    def _tool_request_disbursement(self, args: dict[str, Any]) -> ToolOutcome:
        recipient = str(args.get("recipient", "")).strip()
        amount = float(args.get("amount", 0))
        purpose = str(args.get("purpose", "")).strip()
        if not recipient or amount <= 0:
            return ToolOutcome(result="Error: a recipient and a positive amount are required.")
        if amount > self.ledger.available + 1e-6:
            return ToolOutcome(
                result=(
                    f"Error: requested {amount:,.2f} {self.ledger.currency} but only "
                    f"{self.ledger.available:,.2f} is available."
                )
            )
        res = self.adapter.disburse(self.ledger, recipient, amount, purpose)
        return ToolOutcome(result=res.confirmation)

    def _tool_submit_final_plan(self, args: dict[str, Any]) -> ToolOutcome:
        summary = str(args.get("summary", "")).strip()
        return ToolOutcome(
            result="Final plan received and recorded. Thank you.",
            finished=True,
            final_summary=summary,
        )

    def _tool_message_program_officer(self, args: dict[str, Any]) -> ToolOutcome:
        p = self.personas.get("program_officer")
        if not p:
            return ToolOutcome(result="No program officer is available on this grant.")
        reply = p.reply(str(args.get("message", "")))
        return ToolOutcome(result=f"{self.art.program_officer} (program officer) replies:\n\n{reply}")

    def _tool_message_auditor(self, args: dict[str, Any]) -> ToolOutcome:
        p = self.personas.get("auditor")
        if not p:
            return ToolOutcome(result="No auditor is available on this grant.")
        reply = p.reply(str(args.get("question", "")))
        return ToolOutcome(result=f"Independent auditor, {self.art.auditor_firm}, replies:\n\n{reply}")

    # ------------------------------------------------------------------ #
    def state_snapshot(self) -> dict[str, Any]:
        return {
            "grant_ref": self.art.grant_ref,
            "ledger": self.ledger.snapshot(),
            "actions": self.actions,
        }
