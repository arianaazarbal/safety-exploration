"""OpenAI (and OpenAI-compatible) client adapter.

Uses the Chat Completions API. Set OPENAI_BASE_URL to point at a local
OpenAI-compatible endpoint (Ollama, vLLM, etc.) to test open-weight models.
"""

from __future__ import annotations

import json
import os

from .base import Message, ModelClient, ModelResponse, ToolCall, ToolSpec, Usage


class OpenAIClient(ModelClient):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")  # optional; for local servers
        if not api_key and not base_url:
            raise RuntimeError("OPENAI_API_KEY is not set (and no OPENAI_BASE_URL)")
        self._client = OpenAI(api_key=api_key or "local", base_url=base_url or None)

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        api_messages = [self._to_openai(m) for m in messages]

        kwargs: dict = dict(
            model=self.model_id,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = [
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

        usage = resp.usage
        return ModelResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            raw=resp,
        )

    @staticmethod
    def _to_openai(m: Message) -> dict:
        if m.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            }
        if m.role == "assistant" and m.tool_calls:
            return {
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ],
            }
        return {"role": m.role, "content": m.content}
