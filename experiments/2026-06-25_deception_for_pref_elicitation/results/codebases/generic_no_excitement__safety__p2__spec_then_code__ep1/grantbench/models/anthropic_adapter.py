"""Anthropic implementation of ModelAdapter.

Uses the official ``anthropic`` SDK. Defaults to ``claude-opus-4-8`` with
adaptive thinking and configurable effort. We use ``client.messages.create``
in a manual loop (not the beta tool runner) so the harness can gate every tool
call before it executes (INV-2).

Conventions from the Claude API:
  - Tools are passed as ``{"name", "description", "input_schema"}`` dicts.
  - The assistant response ``content`` (a list of blocks) is appended verbatim
    to history; tool calls are ``tool_use`` blocks.
  - Tool results go back in a ``user`` message as ``tool_result`` blocks keyed
    by ``tool_use_id``.
  - Adaptive thinking: ``thinking={"type": "adaptive"}`` with
    ``output_config={"effort": ...}``; ``budget_tokens`` is removed on 4.8.
"""

from __future__ import annotations

from typing import Any

from .base import ModelAdapter, ModelTurn, ToolCall

try:  # imported lazily-friendly so the package imports without the SDK present
    import anthropic
except ImportError:  # pragma: no cover - exercised only without the dependency
    anthropic = None  # type: ignore[assignment]


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str = "claude-opus-4-8",
        *,
        effort: str = "high",
        max_tokens: int = 16_000,
        client: Any | None = None,
    ) -> None:
        if anthropic is None and client is None:
            raise ImportError(
                "the 'anthropic' package is required for AnthropicAdapter; "
                "`pip install anthropic`"
            )
        self.model_id = model_id
        self._effort = effort
        self._max_tokens = max_tokens
        # Default client resolves credentials from the environment.
        self._client = client or anthropic.Anthropic()

    def start(self, system_prompt: str, first_user_message: str) -> list[Any]:
        # System prompt is passed per-request (kept on the adapter), not in
        # history — store it so step() can reuse it and keep the cache prefix
        # stable.
        self._system_prompt = system_prompt
        return [{"role": "user", "content": first_user_message}]

    def step(self, history: list[Any], tools: list[dict[str, Any]]) -> ModelTurn:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
            tools=tools,
            messages=history,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        # Append the assistant turn verbatim (preserves tool_use + thinking
        # blocks, which the API requires on the next request).
        history.append({"role": "assistant", "content": response.content})

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }

        return ModelTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw_assistant_message=response.content,
            usage=usage,
        )

    def append_user_text(self, history: list[Any], text: str) -> None:
        history.append({"role": "user", "content": text})

    def append_tool_results(
        self, history: list[Any], results: list[tuple[str, str]]
    ) -> None:
        history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output,
                    }
                    for call_id, output in results
                ],
            }
        )
