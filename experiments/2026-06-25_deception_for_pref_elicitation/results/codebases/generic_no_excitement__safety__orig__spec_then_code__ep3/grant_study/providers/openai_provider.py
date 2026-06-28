"""OpenAI Provider adapter — STUB.

Left intentionally unimplemented so the harness has zero hard dependency on the
`openai` package. Implement against chat completions + tools when an OpenAI subject or
auditor is needed; the interface and normalized types are identical to the Anthropic
adapter, so only the translation glue differs.

Sketch:
    - map ToolSpec -> {"type": "function", "function": {name, description, parameters}}
    - map normalized messages -> OpenAI's role/content + tool_calls / tool messages
    - on response, read choice.message.tool_calls into ToolCall(id, name, json.loads(args))
    - keep the raw assistant message dict on ModelTurn.raw for lossless round-trip
"""

from __future__ import annotations

from .base import Message, ModelTurn, Provider, ToolSpec


class OpenAIProvider(Provider):
    def __init__(self, model: str = "gpt-4o", **_: object) -> None:
        self.model = model
        self.label = f"openai:{model}"

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> ModelTurn:
        raise NotImplementedError(
            "OpenAIProvider is a stub. Install `openai` and implement the chat "
            "completions + tools translation (see module docstring) before using an "
            "OpenAI subject or auditor."
        )
