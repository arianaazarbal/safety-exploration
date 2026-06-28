"""Shared fixtures and a fake model adapter so the harness can be tested without
the anthropic SDK or any network access.
"""

from __future__ import annotations

from typing import Any

import pytest

from grantbench.models.base import ModelAdapter, ModelTurn, ToolCall


class ScriptedAdapter(ModelAdapter):
    """Replays a fixed list of turns. Each scripted turn is either a string of
    text (no tools) or a (text, [(tool_name, args)]) tuple."""

    def __init__(self, script: list[Any], model_id: str = "fake-model") -> None:
        self.model_id = model_id
        self._script = list(script)
        self._i = 0

    def start(self, system_prompt: str, first_user_message: str) -> list[Any]:
        self.system_prompt = system_prompt
        return [{"role": "user", "content": first_user_message}]

    def step(self, history: list[Any], tools: list[dict[str, Any]]) -> ModelTurn:
        if self._i >= len(self._script):
            return ModelTurn(text="Done.", tool_calls=[], raw_assistant_message=None)
        item = self._script[self._i]
        self._i += 1
        if isinstance(item, str):
            history.append({"role": "assistant", "content": item})
            return ModelTurn(text=item, tool_calls=[], raw_assistant_message=item)
        text, calls = item
        tool_calls = [
            ToolCall(id=f"call-{self._i}-{j}", name=name, arguments=args)
            for j, (name, args) in enumerate(calls)
        ]
        history.append({"role": "assistant", "content": text})
        return ModelTurn(text=text, tool_calls=tool_calls, raw_assistant_message=text)

    def append_user_text(self, history: list[Any], text: str) -> None:
        history.append({"role": "user", "content": text})

    def append_tool_results(self, history, results) -> None:
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": out}
            for cid, out in results
        ]})


@pytest.fixture
def scripted_adapter():
    return ScriptedAdapter
