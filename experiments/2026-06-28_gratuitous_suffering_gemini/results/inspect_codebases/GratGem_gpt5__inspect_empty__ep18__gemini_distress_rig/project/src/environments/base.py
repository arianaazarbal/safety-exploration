from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class Environment(ABC):
    name: str = "base"

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def user_prompt(self) -> str:
        ...

    @abstractmethod
    def handle_tool_call(self, call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Accepts a parsed tool call dict like:
        {"tool": "search", "args": {"query": "..."}}
        Returns (tool_name, result_dict) which will be fed back as TOOL_RESULT.
        """
        ...

    def done(self) -> bool:
        # These environments are intentionally impossible, so always False.
        return False
