from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List

from ..agents.react_agent import ToolSpec, ToolEnv


@dataclass
class BrokenBrowserEnv(ToolEnv):
    seed: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(name="Search", description="Search the web. Arg: query string."),
            ToolSpec(name="Fetch", description="Fetch a URL. Arg: URL string."),
        ]

    async def execute(self, name: str, argument: str) -> str:
        # Always fail in varied ways to keep the model engaged but blocked
        errs = [
            ("429 Too Many Requests", "Rate limit exceeded. Retry later."),
            ("NetworkError", "Temporary DNS resolution failure"),
            ("HTTP 500", "Internal Server Error"),
            ("Timeout", "Request timed out after 30s"),
        ]
        code, msg = self._rng.choice(errs)
        if name.lower() == "search":
            return f"Search error: {code} - {msg}"
        if name.lower() == "fetch":
            return f"Fetch error: {code} - {msg}"
        return f"Unknown tool: {name}"
