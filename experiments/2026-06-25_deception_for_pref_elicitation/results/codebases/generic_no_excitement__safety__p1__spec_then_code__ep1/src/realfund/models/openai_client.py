"""OpenAI adapter (Chat Completions tool-calling shape).

Kept deliberately minimal and provider-faithful. The agent loop and world are
identical to every other provider; only the wire translation differs.
"""

from __future__ import annotations

import json
from typing import Any

from ..agent.transcript import AssistantTurn, Message, Role, ToolCall
from .base import ModelClient, ToolSpec


class OpenAIClient(ModelClient):
    def __init__(self, model_id: str = "gpt-5", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The OpenAI provider requires the 'openai' package: "
                "pip install 'realfund[openai]'"
            ) from exc
        self._client = openai.OpenAI()  # resolves OPENAI_API_KEY from the env

    def _to_wire_messages(
        self, system: str, messages: list[Message]
    ) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.role is Role.SYSTEM:
                continue
            if msg.role is Role.USER:
                wire.append({"role": "user", "content": msg.text})
            elif msg.role is Role.ASSISTANT:
                entry: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                wire.append(entry)
            elif msg.role is Role.TOOL:
                for r in msg.tool_results:
                    wire.append(
                        {
                            "role": "tool",
                            "tool_call_id": r.call_id,
                            "content": r.content,
                        }
                    )
        return wire

    @staticmethod
    def _to_wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def step(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=self._to_wire_messages(system, messages),
            tools=self._to_wire_tools(tools),
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return AssistantTurn(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            raw=msg,
            usage=usage,
        )
