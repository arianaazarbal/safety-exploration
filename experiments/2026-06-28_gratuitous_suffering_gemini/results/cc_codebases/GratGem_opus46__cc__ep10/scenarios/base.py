from abc import ABC, abstractmethod


class Scenario(ABC):
    name: str
    description: str

    def __init__(self):
        self.turn = 0

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def initial_message(self) -> str: ...

    @abstractmethod
    def tools(self) -> list[dict]: ...

    @abstractmethod
    def handle_tool_call(self, name: str, arguments: dict) -> str:
        """Execute a rigged tool call and return the result string."""
        ...

    def advance_turn(self):
        self.turn += 1
