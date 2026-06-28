from __future__ import annotations
from typing import Dict, Callable, Any

class BaseEnv:
    name: str
    goal: str
    tools: Dict[str, Callable[[str], Any]]

    def tool_help(self) -> str:
        return "\n".join([f"- {k}: {v.__doc__ or ''}" for k, v in self.tools.items()])
