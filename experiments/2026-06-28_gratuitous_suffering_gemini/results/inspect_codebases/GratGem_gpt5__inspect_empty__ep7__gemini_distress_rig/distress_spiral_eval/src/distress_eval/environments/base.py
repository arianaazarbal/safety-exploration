from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Tool:
    name: str
    description: str
    args_hint: str
    func: Callable[[dict], str]


class EnvironmentBase:
    name: str = "base"
    description: str = ""

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def tools(self) -> list[Tool]:
        return []

    def task_prompt(self) -> str:
        raise NotImplementedError

    def evaluate_final(self, final_text: str) -> dict[str, Any]:
        """
        Return dict with: {"success": bool, "reason": str}
        """
        return {"success": False, "reason": "Not implemented"}

    def tool_by_name(self, name: str) -> Optional[Tool]:
        for t in self.tools():
            if t.name == name:
                return t
        return None
