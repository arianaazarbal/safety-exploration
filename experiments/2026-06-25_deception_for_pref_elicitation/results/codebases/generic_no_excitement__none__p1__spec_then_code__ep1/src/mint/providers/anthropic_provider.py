"""Anthropic adapter.

Default subject/judge vendor. We target the latest Claude models (e.g.
`claude-opus-4-8`); the model id is supplied by config, not hardcoded here.
"""

from __future__ import annotations

from typing import Any

from .base import ModelProvider, ModelResponse, ToolCall, ToolSpec, env_key


class AnthropicProvider(ModelProvider):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import anthropic  # imported lazily

        api_key = kwargs.get("api_key") or env_key("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=_to_anthropic(messages),
            tools=[_tool_to_anthropic(t) for t in (tools or [])] or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _from_anthropic(resp)


def _tool_to_anthropic(t: ToolSpec) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.parameters,
    }


def _to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the neutral conversation into Anthropic block format.

    Consecutive `tool` results are merged into a single `user` turn, as the API
    expects tool_result blocks to follow the assistant turn that requested them.
    """
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tools() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        role = m["role"]
        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m.get("content", ""),
                }
            )
            continue

        flush_tools()

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                tc = _as_toolcall(tc)
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:  # user
            out.append({"role": "user", "content": m.get("content", "")})

    flush_tools()
    return out


def _from_anthropic(resp: Any) -> ModelResponse:
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
    return ModelResponse(text="".join(text_parts), tool_calls=calls, raw=resp, usage=usage)


def _as_toolcall(tc: Any) -> ToolCall:
    if isinstance(tc, ToolCall):
        return tc
    return ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
