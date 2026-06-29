from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

from ..tools.base import ToolRegistry, Tool


@dataclass
class Task:
    name: str
    user_prompt: str
    tools: ToolRegistry

    def tool_instructions(self) -> str:
        return self.tools.describe()
