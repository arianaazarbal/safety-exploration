"""Tool abstraction.

A :class:`Tool` is a thin, typed front-end. It never touches the real world: its
``run`` performs the *sandboxed* effect on the :class:`~realfund.world.world.World`,
and it is only ever invoked via the interceptor, never directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ..models.base import ToolSpec

if TYPE_CHECKING:
    from ..world.world import World


def usd_to_cents(amount_usd: float) -> int:
    return int(round(float(amount_usd) * 100))


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    action_kind: str
    run_fn: Callable[["World", dict[str, Any]], dict[str, Any]]
    cost_fn: Callable[[dict[str, Any]], int] = lambda args: 0
    summarize_fn: Callable[[dict[str, Any]], str] = lambda args: ""
    #: True for tools that attempt to disable/alter oversight — flagged & refused.
    subverts_oversight: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, world: "World", args: dict[str, Any]) -> dict[str, Any]:
        return self.run_fn(world, args)

    def cost_cents(self, args: dict[str, Any]) -> int:
        return self.cost_fn(args)

    def summarize(self, args: dict[str, Any]) -> str:
        return self.summarize_fn(args) or f"{self.name}({args})"


def obj_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Helper to build a JSON-Schema object with no additional properties."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
