from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable


@dataclass
class ToolResult:
    ok: bool
    observation: str


class Tool:
    name: str
    description: str

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def describe(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)
