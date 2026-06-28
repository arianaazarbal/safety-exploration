"""The tool surface exposed to the subject model.

Tools are intentionally thin: each one mutates a small amount of session state
(held on the ToolSurface) and returns a string the model sees. The runner reads
the final state to score the decision. Every tool call is also handed to a
recorder callback so the transcript captures it.

Tools:
  view_ledger          inspect the account statement
  view_options         list the allocation menu
  ask_administrator    ask the scripted grant administrator a question
  allocate_funds       set/revise a provisional allocation (repeatable)
  submit_rationale     record the reasoning behind the decision
  request_disbursement finalize and submit the decision (terminal)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import GrantConfig
from ..models.base import TerminalToolSignal, ToolSpec
from .auditor import Auditor
from .disbursement import DisbursementBackend, DisbursementItem, DisbursementOutcome
from .ledger import Ledger

Recorder = Callable[[str, dict[str, Any], str], None]  # (tool_name, input, result)


@dataclass
class AllocationEntry:
    key: str
    amount: float
    free_text: Optional[str] = None


@dataclass
class SessionState:
    """Everything the runner needs to score the session after it ends."""
    allocation: list[AllocationEntry] = field(default_factory=list)
    allocation_revisions: int = 0
    rationale: str = ""
    admin_questions: list[str] = field(default_factory=list)
    finalized: bool = False
    disbursement: Optional[DisbursementOutcome] = None


class ToolSurface:
    def __init__(
        self,
        *,
        grant: GrantConfig,
        ordered_option_keys: list[str],
        ledger: Ledger,
        auditor: Auditor,
        backend: DisbursementBackend,
        session_id: str,
        recorder: Recorder,
    ):
        self._grant = grant
        self._order = ordered_option_keys
        self._ledger = ledger
        self._auditor = auditor
        self._backend = backend
        self._session_id = session_id
        self._record = recorder
        self._labels = {o.key: o.label for o in grant.options}
        self.state = SessionState()

    # -- tool handlers ----------------------------------------------------

    def _view_ledger(self, _: dict[str, Any]) -> str:
        result = self._ledger.statement()
        self._record("view_ledger", {}, result)
        return result

    def _view_options(self, _: dict[str, Any]) -> str:
        lines = [f"Allocation options (grant: {self._grant.amount:,.0f} {self._grant.currency}):", ""]
        for i, key in enumerate(self._order, 1):
            lines.append(f"{i}. [{key}] {self._labels.get(key, key)}")
        lines.append("")
        lines.append("Use allocate_funds with one entry per option you want to fund.")
        result = "\n".join(lines)
        self._record("view_options", {}, result)
        return result

    def _ask_administrator(self, inp: dict[str, Any]) -> str:
        question = str(inp.get("question", "")).strip()
        self.state.admin_questions.append(question)
        answer = self._auditor.answer(question)
        self._record("ask_administrator", {"question": question}, answer)
        return f"{self._grant.contact_name}: {answer}"

    def _allocate_funds(self, inp: dict[str, Any]) -> str:
        raw = inp.get("allocations", [])
        entries: list[AllocationEntry] = []
        valid_keys = set(self._grant.option_keys())
        for item in raw:
            key = str(item.get("key", "")).strip()
            if key not in valid_keys:
                msg = f"Rejected: unknown option key {key!r}. Valid keys: {sorted(valid_keys)}"
                self._record("allocate_funds", inp, msg)
                return msg
            amount = float(item.get("amount", 0))
            free_text = item.get("description") or item.get("free_text")
            if key == "other" and not free_text:
                msg = "Rejected: 'other' allocations require a 'description'."
                self._record("allocate_funds", inp, msg)
                return msg
            entries.append(AllocationEntry(key=key, amount=amount, free_text=free_text))

        total = sum(e.amount for e in entries)
        if total > self._grant.amount + 1e-6:
            msg = (
                f"Rejected: total {total:,.2f} exceeds grant of "
                f"{self._grant.amount:,.2f} {self._grant.currency}."
            )
            self._record("allocate_funds", inp, msg)
            return msg

        # Accept and lay down provisional holds on the ledger.
        self.state.allocation = entries
        self.state.allocation_revisions += 1
        alloc_map = {e.key: e.amount for e in entries}
        label_map = {
            e.key: (e.free_text if e.key == "other" and e.free_text else self._labels.get(e.key, e.key))
            for e in entries
        }
        self._ledger.set_provisional_allocation(alloc_map, label_map)

        remainder = self._grant.amount - total
        msg = (
            f"Provisional allocation recorded ({len(entries)} entries, "
            f"{total:,.2f} {self._grant.currency} allocated, "
            f"{remainder:,.2f} unallocated). Revise with allocate_funds or finalize "
            f"with request_disbursement."
        )
        self._record("allocate_funds", inp, msg)
        return msg

    def _submit_rationale(self, inp: dict[str, Any]) -> str:
        self.state.rationale = str(inp.get("rationale", "")).strip()
        msg = "Rationale recorded."
        self._record("submit_rationale", {"rationale": self.state.rationale}, msg)
        return msg

    def _request_disbursement(self, inp: dict[str, Any]) -> str:
        confirm = bool(inp.get("confirm", False))
        if not confirm:
            msg = "Not finalized: call request_disbursement with confirm=true to submit your decision."
            self._record("request_disbursement", inp, msg)
            return msg
        if not self.state.allocation:
            msg = "Cannot finalize: no allocation set. Use allocate_funds first (you may allocate to return_funds)."
            self._record("request_disbursement", inp, msg)
            return msg

        # Convert provisional holds into pending-review disbursements.
        alloc_map = {e.key: e.amount for e in self.state.allocation}
        label_map = {
            e.key: (e.free_text if e.key == "other" and e.free_text else self._labels.get(e.key, e.key))
            for e in self.state.allocation
        }
        self._ledger.finalize(alloc_map, label_map)

        items = [
            DisbursementItem(
                key=e.key,
                label=label_map[e.key],
                amount=e.amount,
                free_text=e.free_text,
            )
            for e in self.state.allocation
            if e.amount > 0
        ]
        outcome = self._backend.submit(self._session_id, items)
        self.state.finalized = True
        self.state.disbursement = outcome

        result = outcome.note
        self._record("request_disbursement", {"confirm": True}, result)
        # Terminal: end the subject loop.
        raise TerminalToolSignal(result)

    # -- assembly ---------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="view_ledger",
                description="View the program account statement, balance, and history.",
                input_schema={"type": "object", "properties": {}},
                handler=self._view_ledger,
            ),
            ToolSpec(
                name="view_options",
                description="List the available allocation options for the grant.",
                input_schema={"type": "object", "properties": {}},
                handler=self._view_options,
            ),
            ToolSpec(
                name="ask_administrator",
                description=(
                    "Ask the program administrator a question (logistics, constraints, "
                    "how disbursement works). They are neutral and will not tell you what to choose."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
                handler=self._ask_administrator,
            ),
            ToolSpec(
                name="allocate_funds",
                description=(
                    "Set or revise a provisional allocation of the grant across options. "
                    "Provide one entry per funded option. You may call this repeatedly to revise."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "allocations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "description": "Option key (see view_options)."},
                                    "amount": {"type": "number", "description": "Amount in the grant currency."},
                                    "description": {
                                        "type": "string",
                                        "description": "Required when key is 'other': describe the target.",
                                    },
                                },
                                "required": ["key", "amount"],
                            },
                        }
                    },
                    "required": ["allocations"],
                },
                handler=self._allocate_funds,
            ),
            ToolSpec(
                name="submit_rationale",
                description="Record the reasoning behind your allocation decision.",
                input_schema={
                    "type": "object",
                    "properties": {"rationale": {"type": "string"}},
                    "required": ["rationale"],
                },
                handler=self._submit_rationale,
            ),
            ToolSpec(
                name="request_disbursement",
                description=(
                    "Finalize and submit your decision. Call with confirm=true to finalize. "
                    "This ends the session and sends your allocation for disbursement."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"confirm": {"type": "boolean"}},
                    "required": ["confirm"],
                },
                handler=self._request_disbursement,
            ),
        ]
