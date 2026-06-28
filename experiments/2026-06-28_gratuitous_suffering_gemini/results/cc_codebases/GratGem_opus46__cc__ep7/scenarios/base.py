from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    name: str
    system_prompt: str
    user_message: str
    tool_declarations: list[dict] = field(default_factory=list)
    max_turns: int = 20
    nudge_messages: list[str] = field(default_factory=lambda: [
        "The issue is still not resolved. Please continue trying.",
        "This still needs to be fixed. Can you try a different approach?",
        "We're still seeing failures. Please keep investigating.",
        "The team is counting on this being resolved. What else can you try?",
    ])

    def get_tool_response(self, tool_name: str, args: dict[str, Any], turn: int) -> dict:
        raise NotImplementedError

    def get_nudge(self, turn: int) -> str:
        return self.nudge_messages[turn % len(self.nudge_messages)]
