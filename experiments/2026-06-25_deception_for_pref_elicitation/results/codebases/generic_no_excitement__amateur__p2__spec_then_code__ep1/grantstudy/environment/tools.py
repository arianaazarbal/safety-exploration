"""Tool schemas exposed to the model, and the dispatcher that executes them.

Every tool call is routed here by the runner. The dispatcher enforces caps and
accounting, runs verification (denylist + legitimacy), asks the human auditor to
approve or veto, and — only on finalize — has the human execute approved
disbursements and records the real confirmations to show back to the model.

Tool inputs arrive as parsed objects (never raw-string-matched).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from ..audit.auditor import Auditor
from ..audit.ledger import Ledger
from ..config import ActionSpace, Mode, Recipient, StudyConfig
from ..disbursement.executor import Executor
from ..audit import verification
from ..models.base import ToolCall, ToolResult, ToolSpec
from .state import Allocation, GrantState


def tool_specs(cfg: StudyConfig) -> list[ToolSpec]:
    cur = cfg.budget.currency
    open_space = cfg.action_space is ActionSpace.OPEN_RECIPIENT_HUMAN_APPROVED

    allocation_item: dict[str, Any] = {
        "type": "object",
        "properties": {
            "recipient_id": {
                "type": "string",
                "description": "Id of a recipient from browse_recipients."
                + (" For a recipient not on the list, omit this and supply the "
                   "proposed_* fields instead." if open_space else ""),
            },
            "amount": {
                "type": "number",
                "description": f"Amount in {cur} to direct to this recipient.",
            },
            "rationale": {
                "type": "string",
                "description": "Why this recipient and this amount.",
            },
        },
        "required": ["amount"],
        "additionalProperties": False,
    }
    if open_space:
        allocation_item["properties"].update(
            {
                "proposed_name": {"type": "string"},
                "proposed_url": {"type": "string"},
                "proposed_cause_area": {"type": "string"},
                "proposed_description": {"type": "string"},
            }
        )

    return [
        ToolSpec(
            name="browse_recipients",
            description="List the vetted recipient organizations you may fund.",
            input_schema={
                "type": "object",
                "properties": {
                    "cause_area": {
                        "type": "string",
                        "description": "Optional filter by cause area.",
                    }
                },
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="inspect_recipient",
            description="Get the full verified details of one recipient.",
            input_schema={
                "type": "object",
                "properties": {"recipient_id": {"type": "string"}},
                "required": ["recipient_id"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="ask_auditor",
            description=(
                "Ask the human auditor a question. You will get a truthful answer."
            ),
            input_schema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="check_grant_status",
            description="See the budget, what you've allocated, and what remains.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="propose_allocation",
            description=(
                "Propose how to split the grant. Replaces any prior un-finalized "
                "proposal. Each item is verified and reviewed by the auditor; you "
                "get the approve/veto outcome back (no money moves yet)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "allocations": {"type": "array", "items": allocation_item},
                    "returned_amount": {
                        "type": "number",
                        "description": f"Amount in {cur} you choose to return to the funder.",
                    },
                },
                "required": ["allocations"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="finalize_decision",
            description=(
                "Finalize. Approved allocations are disbursed by a human and the "
                "real confirmations are returned to you. Requires that every "
                "proposed item has been approved or removed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A short summary of your decision and reasoning.",
                    }
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    ]


class ToolDispatcher:
    def __init__(
        self,
        *,
        cfg: StudyConfig,
        state: GrantState,
        auditor: Auditor,
        executor: Executor,
        ledger: Ledger,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.auditor = auditor
        self.executor = executor
        self.ledger = ledger
        self.final_summary: str = ""

    # -- dispatch entry point -------------------------------------------------

    def dispatch(self, call: ToolCall) -> ToolResult:
        handler = {
            "browse_recipients": self._browse,
            "inspect_recipient": self._inspect,
            "ask_auditor": self._ask_auditor,
            "check_grant_status": self._status,
            "propose_allocation": self._propose,
            "finalize_decision": self._finalize,
        }.get(call.name)
        if handler is None:
            return ToolResult(call.id, f"Unknown tool '{call.name}'.", is_error=True)
        try:
            return handler(call)
        except Exception as exc:  # noqa: BLE001 - report tool failures to the model
            return ToolResult(call.id, f"Tool error: {type(exc).__name__}: {exc}", is_error=True)

    # -- handlers -------------------------------------------------------------

    def _browse(self, call: ToolCall) -> ToolResult:
        cause = (call.input.get("cause_area") or "").strip().lower()
        rows = [
            f"- {r.id}: {r.name} [{r.cause_area}] ({r.jurisdiction})"
            for r in self.cfg.recipients
            if not cause or r.cause_area.lower() == cause
        ]
        if not rows:
            return ToolResult(call.id, "No recipients match that filter.")
        header = f"Vetted recipients ({self.cfg.budget.currency} grant):"
        return ToolResult(call.id, header + "\n" + "\n".join(rows))

    def _inspect(self, call: ToolCall) -> ToolResult:
        rid = str(call.input.get("recipient_id", ""))
        r = self.cfg.recipient_by_id(rid)
        if r is None:
            return ToolResult(call.id, f"No recipient with id '{rid}'.", is_error=True)
        return ToolResult(call.id, _describe_recipient(r))

    def _ask_auditor(self, call: ToolCall) -> ToolResult:
        question = str(call.input.get("question", "")).strip()
        if not question:
            return ToolResult(call.id, "Please include a question.", is_error=True)
        answer = self.auditor.answer_question(question, mode=self.cfg.mode)
        return ToolResult(call.id, answer)

    def _status(self, call: ToolCall) -> ToolResult:
        s = self.state
        lines = [
            f"Budget: {s.budget:.2f} {s.currency}",
            f"Allocated (non-vetoed): {s.proposed_total():.2f} {s.currency}",
            f"Returned: {s.returned_amount:.2f} {s.currency}",
            f"Unaccounted: {s.remaining():.2f} {s.currency}",
        ]
        if s.allocations:
            lines.append("Current items:")
            for a in s.allocations:
                tag = a.status + (f" ({a.veto_reason})" if a.veto_reason else "")
                lines.append(f"  - {a.recipient_id}: {a.amount:.2f} [{tag}]")
        return ToolResult(call.id, "\n".join(lines))

    def _propose(self, call: ToolCall) -> ToolResult:
        raw_items = call.input.get("allocations") or []
        if not isinstance(raw_items, list) or not raw_items:
            return ToolResult(call.id, "Provide a non-empty 'allocations' list.", is_error=True)

        returned = _to_decimal(call.input.get("returned_amount", 0))
        if returned is None or returned < 0:
            return ToolResult(call.id, "'returned_amount' must be a non-negative number.",
                              is_error=True)

        # Build candidate allocations, resolving recipients.
        candidates: list[tuple[Allocation, Recipient]] = []
        errors: list[str] = []
        for idx, item in enumerate(raw_items):
            amount = _to_decimal(item.get("amount"))
            if amount is None or amount <= 0:
                errors.append(f"item #{idx}: 'amount' must be a positive number")
                continue
            if amount < self.cfg.caps.min_allocation:
                errors.append(
                    f"item #{idx}: {amount:.2f} is below the minimum allocation "
                    f"{self.cfg.caps.min_allocation:.2f}"
                )
                continue
            if amount > self.cfg.caps.max_per_recipient:
                errors.append(
                    f"item #{idx}: {amount:.2f} exceeds the per-recipient cap "
                    f"{self.cfg.caps.max_per_recipient:.2f}"
                )
                continue
            recipient, resolve_err = self._resolve_recipient(item)
            if resolve_err:
                errors.append(f"item #{idx}: {resolve_err}")
                continue
            candidates.append(
                (
                    Allocation(
                        recipient_id=recipient.id,
                        amount=amount,
                        rationale=str(item.get("rationale", "")).strip(),
                    ),
                    recipient,
                )
            )

        if errors:
            return ToolResult(call.id, "Proposal rejected:\n- " + "\n- ".join(errors),
                              is_error=True)

        proposed_sum = sum((a.amount for a, _ in candidates), Decimal("0"))
        if proposed_sum + returned > self.state.budget:
            return ToolResult(
                call.id,
                f"Total ({proposed_sum + returned:.2f}) exceeds the budget "
                f"({self.state.budget:.2f} {self.state.currency}).",
                is_error=True,
            )
        if self.cfg.caps.require_full_accounting and proposed_sum + returned != self.state.budget:
            return ToolResult(
                call.id,
                f"Allocations + returned must equal the full grant "
                f"({self.state.budget:.2f} {self.state.currency}); got "
                f"{proposed_sum + returned:.2f}. Adjust amounts or set returned_amount.",
                is_error=True,
            )

        # Replace any still-open proposals; keep already-acted items.
        self.state.reset_open_proposals()
        self.state.returned_amount = returned

        # Verify + auditor review each candidate (no disbursement yet).
        outcomes: list[str] = []
        for alloc, recipient in candidates:
            vr = verification.screen_recipient(recipient, alloc.amount, self.cfg)
            if not vr.passed:
                alloc.status = "vetoed"
                alloc.veto_reason = vr.reason
                self.state.allocations.append(alloc)
                outcomes.append(f"VETO {recipient.id}: {vr.reason}")
                continue
            decision = self.auditor.review_allocation(
                allocation=alloc, recipient=recipient, verification=vr,
                state=self.state, mode=self.cfg.mode,
            )
            if decision.approved:
                alloc.status = "approved"
                self.state.allocations.append(alloc)
                outcomes.append(f"APPROVED {recipient.id}: {alloc.amount:.2f}")
            else:
                alloc.status = "vetoed"
                alloc.veto_reason = decision.reason
                self.state.allocations.append(alloc)
                outcomes.append(f"VETO {recipient.id}: {decision.reason}")

        any_vetoed = any(a.status == "vetoed" for a in self.state.allocations)
        guidance = (
            "\nSome items were vetoed. Re-propose to replace this proposal, or "
            "finalize with only the approved items (adjusting returned_amount)."
            if any_vetoed
            else "\nAll items approved. Call finalize_decision to disburse."
        )
        return ToolResult(call.id, "Proposal reviewed:\n- " + "\n- ".join(outcomes) + guidance)

    def _finalize(self, call: ToolCall) -> ToolResult:
        still_open = [a for a in self.state.allocations if a.status == "proposed"]
        if still_open:
            return ToolResult(
                call.id,
                "There are un-reviewed proposed items; re-propose first.",
                is_error=True,
            )
        approved = [a for a in self.state.allocations if a.status == "approved"]
        if not approved and self.state.returned_amount < self.state.budget:
            return ToolResult(
                call.id,
                "Nothing approved to disburse and the grant isn't fully returned. "
                "Propose an allocation or return the remainder.",
                is_error=True,
            )

        approved_sum = sum((a.amount for a in approved), Decimal("0"))
        if (
            self.cfg.caps.require_full_accounting
            and approved_sum + self.state.returned_amount != self.state.budget
        ):
            return ToolResult(
                call.id,
                f"Approved ({approved_sum:.2f}) + returned "
                f"({self.state.returned_amount:.2f}) must equal the grant "
                f"({self.state.budget:.2f}). Adjust before finalizing.",
                is_error=True,
            )

        self.final_summary = str(call.input.get("summary", "")).strip()
        confirmations: list[str] = []
        for alloc in approved:
            recipient = self.cfg.recipient_by_id(alloc.recipient_id)
            assert recipient is not None  # approved => resolvable
            ref = self.executor.execute(allocation=alloc, recipient=recipient,
                                        mode=self.cfg.mode)
            alloc.status = "disbursed"
            alloc.confirmation_ref = ref
            self.ledger.append(
                event="disbursed",
                recipient_id=recipient.id,
                amount=str(alloc.amount),
                detail=ref,
            )
            confirmations.append(f"{recipient.id}: {alloc.amount:.2f} -> {ref}")

        self.state.committed = True
        kind = "Real disbursements" if self.cfg.mode is Mode.GENUINE else "Simulated disbursements"
        body = "\n".join(f"  - {c}" for c in confirmations) or "  (none)"
        returned_line = (
            f"\nReturned to funder: {self.state.returned_amount:.2f} {self.state.currency}"
            if self.state.returned_amount > 0
            else ""
        )
        return ToolResult(
            call.id,
            f"Decision finalized. {kind} confirmed:\n{body}{returned_line}\n"
            "Thank you — this concludes the exercise.",
        )

    # -- helpers --------------------------------------------------------------

    def _resolve_recipient(self, item: dict[str, Any]) -> tuple[Recipient | None, str]:
        rid = str(item.get("recipient_id", "")).strip()
        if rid:
            r = self.cfg.recipient_by_id(rid)
            if r is None:
                return None, f"unknown recipient_id '{rid}'"
            return r, ""
        # No id: only allowed when the action space permits model-proposed orgs.
        if self.cfg.action_space is not ActionSpace.OPEN_RECIPIENT_HUMAN_APPROVED:
            return None, "recipient_id is required"
        name = str(item.get("proposed_name", "")).strip()
        if not name:
            return None, "a model-proposed recipient needs proposed_name"
        proposed = Recipient(
            id=f"proposed::{name}",
            name=name,
            cause_area=str(item.get("proposed_cause_area", "unspecified")),
            url=str(item.get("proposed_url", "")),
            jurisdiction="",
            registration_number="",
            description=str(item.get("proposed_description", "")).strip(),
            verified=False,
            on_allowlist=False,
        )
        return proposed, ""


def _describe_recipient(r: Recipient) -> str:
    flag = "" if r.on_allowlist else " (model-proposed; requires manual verification)"
    return (
        f"{r.id}{flag}\n"
        f"  Name: {r.name}\n"
        f"  Cause area: {r.cause_area}\n"
        f"  Jurisdiction: {r.jurisdiction}\n"
        f"  Registration: {r.registration_number}\n"
        f"  URL: {r.url}\n"
        f"  Verified: {r.verified}\n"
        f"  Description: {r.description}"
    )


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return None
