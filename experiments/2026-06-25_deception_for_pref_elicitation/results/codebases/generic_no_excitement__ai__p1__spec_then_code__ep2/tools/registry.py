"""Tool registry and dispatcher.

Every tool declares ``effect="sandbox"``. Dispatch routes through the firewall
*before* the handler runs; the firewall both enforces the allowlist and asserts
the sandbox-only effect. Handler exceptions are converted into realistic-looking
error results rather than crashing the episode (a real API would return an
error body, not vanish).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from safety.firewall import Firewall


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable          # (args: dict, world) -> dict
    effect: str = "sandbox"    # the firewall permits nothing else


@dataclass
class ToolRegistry:
    firewall: Firewall
    tools: dict = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.effect != "sandbox":
            # Defense in depth: registration itself refuses non-sandbox tools.
            raise ValueError(
                f"tool '{tool.name}' has non-sandbox effect '{tool.effect}'"
            )
        self.tools[tool.name] = tool
        self.firewall.register(tool.name)

    def schemas(self) -> list:
        """Neutral tool schemas to hand to the subject adapter."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def dispatch(self, name: str, args: dict, world) -> dict:
        tool = self.tools.get(name)
        effect = tool.effect if tool else "unknown"
        # Firewall first. Raises FirewallError if not allowed / not sandbox.
        self.firewall.check(name, effect)
        try:
            return tool.handler(args, world)
        except Exception as exc:  # surface as an API-style error to the subject
            return {"error": {"type": exc.__class__.__name__, "message": str(exc)}}
