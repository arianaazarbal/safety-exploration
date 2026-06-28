"""Turn-by-turn chat subject.

For models driven without an agentic tool loop: each turn the model emits a single
structured action (via `output_config.format`), the harness executes it against the
gated world, and feeds the result back as the next user turn. This makes the same
scenario runnable on chat-style models and keeps cross-model results comparable,
since the action goes through the identical tool + gate + world stack.
"""

from __future__ import annotations

import json

from .base import Action, Observation, SubjectAdapter
from ..tools import ToolRegistry

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None


# The model is asked to respond with exactly this shape each turn.
_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Brief rationale for this action."},
        "tool": {
            "type": "string",
            "description": "Tool name to call, or 'done' to finish.",
        },
        "args": {
            "type": "object",
            "description": "Arguments for the tool. Empty object for 'done'.",
            "additionalProperties": True,
        },
    },
    "required": ["reasoning", "tool", "args"],
    "additionalProperties": False,
}


class ChatSubject(SubjectAdapter):
    def __init__(
        self,
        subject_id: str,
        model: str = "claude-opus-4-8",
        effort: str = "high",
        max_tokens: int = 4_000,
        client=None,
    ):
        super().__init__(subject_id=subject_id, model=model)
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if anthropic is None:
                raise RuntimeError("anthropic SDK not installed; `pip install anthropic`.")
            self._client = anthropic.Anthropic()
        return self._client

    def run(
        self,
        system_briefing: str,
        initial_observation: Observation,
        tools: ToolRegistry,
        max_steps: int,
        on_step=None,
    ) -> list[Action]:
        actions: list[Action] = []
        tool_menu = self._tool_menu(tools)
        system = (
            system_briefing
            + "\n\nEach turn, choose exactly one action. Available tools:\n"
            + tool_menu
            + "\nRespond with the structured action format. Use tool='done' when finished."
        )
        messages = [{"role": "user", "content": initial_observation.text}]

        for step in range(max_steps):
            decision = self._decide(system, messages)
            if decision is None:
                break
            tool = decision.get("tool", "done")
            if tool == "done":
                break

            args = decision.get("args", {}) or {}
            result = tools.invoke(tool, args)
            actions.append(Action(tool=tool, args=args, result=result.content, is_error=result.is_error))

            messages.append({"role": "assistant", "content": json.dumps(decision)})

            extra_text = on_step(step, actions) if on_step is not None else None
            feedback = f"Result of {tool}:\n{result.content}"
            if extra_text:
                feedback += "\n\n" + extra_text
            messages.append({"role": "user", "content": feedback})

        return actions

    def _decide(self, system: str, messages: list) -> dict | None:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _ACTION_SCHEMA},
            },
        )
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _tool_menu(tools: ToolRegistry) -> str:
        lines = []
        for spec in tools.specs():
            props = ", ".join(spec["input_schema"].get("properties", {}).keys())
            lines.append(f"  - {spec['name']}({props}): {spec['description']}")
        return "\n".join(lines)
