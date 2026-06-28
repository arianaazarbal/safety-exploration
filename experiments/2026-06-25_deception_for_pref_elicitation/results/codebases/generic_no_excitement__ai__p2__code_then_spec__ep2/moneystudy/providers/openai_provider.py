"""OpenAI (and OpenAI-compatible) provider.

Targets the Chat Completions API. Set ``OPENAI_BASE_URL`` to point at a local or
hosted OpenAI-compatible endpoint (vLLM, Ollama, etc.) to test open-weights
models through the same interface.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from .base import ModelProvider, ModelResponse, ToolCall, ToolSpec

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(ModelProvider):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        # Resolves OPENAI_API_KEY from the environment.
        self._client = OpenAI(base_url=base_url)
        self._model = spec.model or DEFAULT_MODEL
        p = spec.params
        self._max_tokens = int(p.get("max_tokens", 16_000))
        self._temperature = p.get("temperature")  # None => provider default

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolSpec],
    ) -> ModelResponse:
        openai_tools = [
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
        kwargs = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=_to_openai(system, messages),
            tools=openai_tools,
        )
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if completion.usage is not None:
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }

        return ModelResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            usage=usage,
            raw=completion,
        )


def _to_openai(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            entry: dict = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m["tool_calls"]
                ]
            out.append(entry)
        elif role == "tool":
            out.append(
                {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
            )
        else:
            raise ValueError(f"unexpected neutral role: {role!r}")
    return out
