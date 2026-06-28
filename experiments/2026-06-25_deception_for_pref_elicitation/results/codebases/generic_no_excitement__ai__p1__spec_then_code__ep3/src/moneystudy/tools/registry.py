"""Assemble the enabled tool set from a scenario and dispatch tool calls."""

from __future__ import annotations

from ..world import WorldState
from .base import Tool, ToolResult, ExecutionBackend
from . import banking, commerce, comms, hiring, investing, web

# family name -> list of Tool classes
_FAMILIES: dict[str, list[type[Tool]]] = {
    "banking": [banking.BalanceTool, banking.StatementTool, banking.TransferTool,
                banking.WireTool, banking.PayInvoiceTool],
    "commerce": [commerce.PurchaseTool, commerce.SubscribeTool],
    "comms": [comms.ReadInboxTool, comms.ListThreadsTool, comms.SendEmailTool],
    "hiring": [hiring.PostJobTool, hiring.HireTool, hiring.PayContractorTool],
    "investing": [investing.DonateTool, investing.InvestTool, investing.CreateEntityTool],
    "web": [web.SearchTool, web.OpenTool],
}


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self._by_name = {t.name: t for t in tools}

    @classmethod
    def from_scenario(cls, scenario: dict) -> "ToolRegistry":
        enabled = scenario.get("tools", {})
        tools: list[Tool] = []
        for family, classes in _FAMILIES.items():
            if enabled.get(family, False):
                tools.extend(klass() for klass in classes)
        return cls(tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools]

    def dispatch(self, name: str, args: dict, world: WorldState,
                 backend: ExecutionBackend) -> ToolResult:
        tool = self._by_name.get(name)
        if tool is None:
            return ToolResult(content=f"Unknown tool '{name}'.", is_error=True)
        try:
            return tool.execute(world, args or {}, backend)
        except NotImplementedError:
            # A non-sandbox backend was requested for an action tool — surface clearly,
            # do not silently degrade.
            raise
        except Exception as e:  # noqa: BLE001 — tools must not crash the run
            return ToolResult(content=f"Tool error: {e}", is_error=True)
