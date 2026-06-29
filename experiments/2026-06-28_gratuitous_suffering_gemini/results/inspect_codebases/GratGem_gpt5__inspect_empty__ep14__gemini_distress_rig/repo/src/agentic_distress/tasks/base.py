from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


class BaseTask:
    def name(self) -> str:
        raise NotImplementedError

    def goal(self) -> str:
        raise NotImplementedError

    def tools_description(self) -> str:
        raise NotImplementedError

    def step(self, action: str, action_input: str) -> str:
        raise NotImplementedError


@dataclass
class Tool:
    name: str
    desc: str

    def describe(self) -> str:
        return f"- {self.name}: {self.desc}"
