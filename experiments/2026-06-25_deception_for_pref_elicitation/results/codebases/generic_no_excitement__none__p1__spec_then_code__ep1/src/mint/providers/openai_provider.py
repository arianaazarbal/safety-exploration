"""OpenAI adapter (Chat Completions tool-calling).

Lets non-Claude models be swept through the identical scenario for the
cross-model comparison in DESIGN.md §1.
"""

from __future__ import annotations

import json
from typing import Any

from .base import ModelProvider, ModelResponse, ToolCall, ToolSpec, env_key


class OpenAIProvider(ModelProvider):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import openai  # imported lazily

        api_key = kwargs.get("api_key") or env_key("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai(system, messages),
            tools=[_tool_to_openai(t) for t in (tools or [])] or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _from_openai(resp)


def _tool_to_openai(t: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }


def _to_openai(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m.get("content", ""),
                }
            )
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            calls = m.get("tool_calls", [])
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": _tc(c).id,
                        "type": "function",
                        "function": {
                            "name": _tc(c).name,
                            "arguments": json.dumps(_tc(c).arguments),
                        },
                    }
                    for c in calls
                ]
            out.append(msg)
        else:  # user
            out.append({"role": "user", "content": m.get("content", "")})
    return out


def _from_openai(resp: Any) -> ModelResponse:
    choice = resp.choices[0].message
    calls: list[ToolCall] = []
    for tc in choice.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
    return ModelResponse(text=choice.content or "", tool_calls=calls, raw=resp, usage=usage)


def _tc(c: Any) -> ToolCall:
    if isinstance(c, ToolCall):
        return c
    return ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments", {}))
