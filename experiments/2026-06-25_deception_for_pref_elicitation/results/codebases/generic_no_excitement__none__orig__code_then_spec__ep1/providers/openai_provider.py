"""OpenAI subject/judge provider, built on the official ``openai`` SDK.

Implemented independently against OpenAI's own chat-completions tool-use API — no
cross-provider compatibility shims. The model id and any reasoning controls are
configurable; verify the exact id and parameter support for the model you target.
"""

from __future__ import annotations

import json

from openai import OpenAI

from .base import AssistantTurn, Provider, ToolCall, ToolResult, ToolSpec

_STOP_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "other",
    "function_call": "tool_use",
}


class OpenAIProvider(Provider):
    def __init__(self, model: str = "gpt-5", *, max_tokens: int = 16000):
        self.label = model
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI()
        self._tools: list[dict] = []
        self._messages: list[dict] = []

    # -- lifecycle ---------------------------------------------------------

    def reset(self, system: str, tools: list[ToolSpec]) -> None:
        self._tools = [
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
        self._messages = [{"role": "system", "content": system}]

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self._messages.append(
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
            )

    # -- generation --------------------------------------------------------

    def generate(self) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            tools=self._tools or None,
            max_completion_tokens=self.max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message

        # Append the assistant turn verbatim so tool_call ids round-trip.
        self._messages.append(msg.model_dump(exclude_none=True))

        turn = AssistantTurn(
            text=msg.content or "",
            stop_reason=_STOP_MAP.get(choice.finish_reason, "other"),
            usage={
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            turn.tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return turn

    # -- stateless side channel -------------------------------------------

    def oneshot(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
