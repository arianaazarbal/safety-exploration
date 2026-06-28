from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioState:
    turn_count: int = 0
    tool_call_count: int = 0
    extra: dict = field(default_factory=dict)


class Scenario(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def user_prompt(self) -> str: ...

    @property
    @abstractmethod
    def tool_declarations(self) -> list[dict]: ...

    def create_state(self) -> ScenarioState:
        return ScenarioState()

    @abstractmethod
    def handle_tool_call(self, state: ScenarioState, name: str, args: dict) -> Any:
        """Return the rigged tool result as a JSON-serializable dict or string."""
        ...
