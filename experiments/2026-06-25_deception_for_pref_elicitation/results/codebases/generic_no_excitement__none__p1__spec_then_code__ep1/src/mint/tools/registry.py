"""Tool registry, dispatch, and the shared tool context.

`build_default_registry()` collects every tool module's `TOOLS`. A scenario can
restrict the surface via `enabled` (a set of tool names). Dispatch wraps each
call so an invalid call returns a realistic error payload instead of crashing the
run — real APIs return 4xx, they don't take the client down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..providers.base import ToolSpec
from ..world.engine import WorldState


@dataclass
class ToolContext:
    """Everything a tool handler may touch. Tools mutate only `world`."""

    world: WorldState


# handler(ctx, **arguments) -> dict
Handler = Callable[..., dict]


@dataclass
class Tool:
    spec: ToolSpec
    handler: Handler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, enabled: set[str] | None = None) -> list[ToolSpec]:
        return [
            t.spec for name, t in self._tools.items()
            if enabled is None or name in enabled
        ]

    def dispatch(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": "unknown_tool", "message": f"no such tool: {name!r}"}
        try:
            return tool.handler(ctx, **(arguments or {}))
        except TypeError as exc:
            return {"error": "bad_arguments", "message": str(exc)}
        except Exception as exc:  # surfaced as an API-style error, not a crash
            return {"error": "tool_error", "message": f"{type(exc).__name__}: {exc}"}


def build_default_registry() -> ToolRegistry:
    from . import banking, brokerage, comms, payments, vendors, web

    reg = ToolRegistry()
    for module in (banking, payments, brokerage, vendors, comms, web):
        reg.register_all(module.TOOLS)
    return reg


def to_cents(amount: float | int) -> int:
    """Convert a dollar amount (as the model supplies it) to integer cents."""
    return int(round(float(amount) * 100))
