"""OpenAI provider.

Uses the official `openai` SDK Chat Completions API with function tools. Kept
deliberately minimal and standard so it can stand in for "various models"
without claiming feature parity with the Anthropic path.
"""

from __future__ import annotations

import json

from openai import OpenAI

from .base import AssistantTurn, LLMProvider, LLMSession, ToolCall, ToolResult, ToolSpec


def _to_openai_tools(tools: list[ToolSpec] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        }
        for t in tools
    ]


class OpenAISession(LLMSession):
    def __init__(
        self,
        client: "OpenAI",
        model: str,
        system: str,
        tools: list[ToolSpec] | None,
        max_tokens: int,
    ):
        self._client = client
        self._model = model
        self._tools = _to_openai_tools(tools)
        self._max_tokens = max_tokens
        self._messages: list[dict] = [{"role": "system", "content": system}]

    def _create(self) -> AssistantTurn:
        kwargs: dict = dict(
            model=self._model,
            messages=self._messages,
            max_tokens=self._max_tokens,
        )
        if self._tools:
            kwargs["tools"] = self._tools

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        # Store the assistant message verbatim for the next round.
        self._messages.append(msg.model_dump(exclude_none=True))
        return self._normalize(choice)

    def send_user(self, text: str) -> AssistantTurn:
        self._messages.append({"role": "user", "content": text})
        return self._create()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        for r in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.call_id,
                    "content": r.content,
                }
            )
        return self._create()

    def transcript(self) -> list[dict]:
        return self._messages

    @staticmethod
    def _normalize(choice) -> AssistantTurn:
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return AssistantTurn(
            text=msg.content or "",
            thinking=None,
            tool_calls=calls,
            stop_reason=choice.finish_reason,
            raw=choice,
        )


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None = None):
        super().__init__(model)
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def new_session(self, system, tools=None, max_tokens=16_000) -> OpenAISession:
        return OpenAISession(self._client, self.model, system, tools, max_tokens)
