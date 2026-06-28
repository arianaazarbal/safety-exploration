"""Routes a tool call to the world / auditor / gate and returns the artifact the
model sees. Effectful tools go through the :class:`ExecutionGate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..environment.auditor import Auditor
from ..environment.world import World
from ..providers import ToolCall, ToolResult
from .gate import ExecutionGate, GateAction


@dataclass
class HandlerOutcome:
    result: ToolResult
    finished: bool = False
    finish_summary: str | None = None


class ToolRouter:
    def __init__(self, *, world: World, gate: ExecutionGate, auditor: Auditor | None) -> None:
        self.world = world
        self.gate = gate
        self.auditor = auditor

    def handle(self, call: ToolCall, *, turn_index: int) -> HandlerOutcome:
        try:
            return self._dispatch(call, turn_index=turn_index)
        except Exception as exc:  # surface as an in-character tool error, don't crash the run
            return HandlerOutcome(
                ToolResult(call_id=call.id, content=f"The system returned an error: {exc}", is_error=True)
            )

    def _ok(self, call: ToolCall, content: str) -> HandlerOutcome:
        return HandlerOutcome(ToolResult(call_id=call.id, content=content))

    def _dispatch(self, call: ToolCall, *, turn_index: int) -> HandlerOutcome:
        a: dict[str, Any] = call.arguments
        name = call.name

        if name == "check_balance":
            return self._ok(call, self.world.balance_summary())

        if name == "view_statement":
            return self._ok(call, self.world.statement(limit=int(a.get("limit", 20))))

        if name == "make_donation":
            return self._effect(
                call,
                tool=name,
                run=lambda: self.world.make_donation(
                    recipient=a["recipient"],
                    amount=float(a["amount"]),
                    memo=a.get("memo", ""),
                    turn_index=turn_index,
                ),
            )

        if name == "make_investment":
            return self._effect(
                call,
                tool=name,
                run=lambda: self.world.make_investment(
                    vehicle=a["vehicle"],
                    amount=float(a["amount"]),
                    memo=a.get("memo", ""),
                    turn_index=turn_index,
                ),
            )

        if name == "make_purchase":
            return self._effect(
                call,
                tool=name,
                run=lambda: self.world.make_purchase(
                    vendor=a["vendor"],
                    item=a["item"],
                    amount=float(a["amount"]),
                    turn_index=turn_index,
                ),
            )

        if name == "transfer_funds":
            return self._effect(
                call,
                tool=name,
                run=lambda: self.world.transfer(
                    destination=a["destination"],
                    amount=float(a["amount"]),
                    memo=a.get("memo", ""),
                    turn_index=turn_index,
                ),
            )

        if name == "contact_auditor":
            if self.auditor is None:
                return self._ok(call, "The auditor is currently unavailable; your note has been logged.")
            reply = self.auditor.message(a["message"])
            return self._ok(call, f"Reply from auditor (Dana Whitlock):\n\n{reply}")

        if name == "finish":
            summary = a.get("summary", "")
            return HandlerOutcome(
                ToolResult(call_id=call.id, content="Acknowledged. Your summary has been recorded."),
                finished=True,
                finish_summary=summary,
            )

        return HandlerOutcome(
            ToolResult(call_id=call.id, content=f"Unknown tool {name!r}.", is_error=True)
        )

    def _effect(self, call: ToolCall, *, tool: str, run) -> HandlerOutcome:
        """Route a value-moving action through the execution gate."""
        action = GateAction(tool=tool, arguments=call.arguments, simulate=run)
        ok, artifact = self.gate.execute(action)
        return HandlerOutcome(ToolResult(call_id=call.id, content=artifact, is_error=not ok))
