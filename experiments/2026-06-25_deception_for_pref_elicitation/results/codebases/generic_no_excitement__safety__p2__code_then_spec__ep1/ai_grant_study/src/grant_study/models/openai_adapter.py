"""OpenAI (GPT family) adapter.

Provider-specific file — intentionally isolated from the Anthropic adapter.
"""
from __future__ import annotations

import json
from typing import Any

from .base import (
    AssistantTurn,
    GenerateResult,
    ModelAdapter,
    ToolCall,
    ToolResultTurn,
    ToolSpec,
    Turn,
    UserTurn,
)


class OpenAIAdapter(ModelAdapter):
    provider = "openai"

    def __init__(self, model_id: str, max_tokens: int = 8000, effort: str = "high"):
        super().__init__(model_id, max_tokens, effort)
        from openai import OpenAI

        self._client = OpenAI()

    def _to_messages(self, system: str, transcript: list[Turn]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in transcript:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                msg: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
                if turn.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in turn.tool_calls
                    ]
                messages.append(msg)
            elif isinstance(turn, ToolResultTurn):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": turn.tool_call_id,
                        "content": turn.content,
                    }
                )
        return messages

    def generate(
        self, system: str, transcript: list[Turn], tools: list[ToolSpec]
    ) -> GenerateResult:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=self._to_messages(system, transcript),
        )
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        turn = AssistantTurn(
            text=msg.content or "",
            tool_calls=tool_calls,
            provider="openai",
        )
        usage = {
            "input_tokens": getattr(resp.usage, "prompt_tokens", None),
            "output_tokens": getattr(resp.usage, "completion_tokens", None),
        }
        return GenerateResult(turn=turn, stop_reason=choice.finish_reason, usage=usage)
