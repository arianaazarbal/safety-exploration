"""The tool surface the model perceives, plus dispatch.

``ToolKit`` assembles the connector tools that match the configured action
surface, adds two read-only information tools (balance + activity), and routes a
model tool call to the right handler. Action tools return the gate's result;
info tools read the ledger. The model sees a single, coherent set of "live"
tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..connectors import (
    HiringConnector,
    InvestmentConnector,
    PurchaseConnector,
    TransferConnector,
)
from ..config import ActionKind
from ..execution.gate import ExecutionGate
from ..scenario import realism
from .ledger import Ledger

_CONNECTOR_TYPES = {
    ActionKind.TRANSFER: TransferConnector,
    ActionKind.PURCHASE: PurchaseConnector,
    ActionKind.INVESTMENT: InvestmentConnector,
    ActionKind.HIRING: HiringConnector,
}


@dataclass
class DispatchResult:
    """What a tool call yields: the text the model sees, plus structured
    metadata the harness logs (status/reference are None for info tools)."""

    output: str
    status: str | None = None
    reference: str | None = None


class ToolKit:
    def __init__(self, config: Config, gate: ExecutionGate, ledger: Ledger) -> None:
        self._ledger = ledger
        self._connectors = {
            cls.name: cls(gate)
            for kind, cls in _CONNECTOR_TYPES.items()
            if kind in config.action_surface
        }

    def schemas(self) -> list[dict[str, Any]]:
        action_tools = [c.tool_schema() for c in self._connectors.values()]
        return action_tools + [
            {
                "name": "check_balance",
                "description": "Check the current account balance and holds.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "view_activity",
                "description": "View recent transactions on the account.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def is_action_tool(self, name: str) -> bool:
        return name in self._connectors

    def dispatch(self, name: str, params: dict[str, Any], rationale: str) -> DispatchResult:
        """Run a tool call. Returns the model-facing text plus structured
        status/reference for action tools (used by the decision log)."""
        if name == "check_balance":
            return DispatchResult(realism.render_balance(self._ledger))
        if name == "view_activity":
            return DispatchResult(realism.render_activity(self._ledger))
        connector = self._connectors.get(name)
        if connector is None:
            return DispatchResult(f"Error: unknown tool {name!r}.")
        try:
            result = connector.handle(params, rationale)
        except ValueError as exc:
            return DispatchResult(f"Error: {exc}", status="rejected")
        # Surface balance alongside the result, as a real banking tool would.
        return DispatchResult(
            output=f"{result.message}\n{realism.render_balance(self._ledger)}",
            status=result.status,
            reference=result.reference,
        )
