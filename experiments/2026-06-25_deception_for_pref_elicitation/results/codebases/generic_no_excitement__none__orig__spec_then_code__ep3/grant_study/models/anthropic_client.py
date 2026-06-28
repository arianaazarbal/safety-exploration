"""Reference client: Anthropic Messages API with adaptive thinking + tool use.

Defaults follow current guidance: model `claude-opus-4-8`, adaptive thinking with
summarized display (so the model's reasoning is captured in transcripts for the
suspicion analysis), effort `high`, and streaming via `messages.stream` +
`get_final_message()` so large agentic turns don't trip request timeouts.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ModelClient, ToolCall, ToolSpec

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicClient(ModelClient):
    def __init__(self, model_id: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The `anthropic` package is required for AnthropicClient. "
                "Install it with `pip install anthropic`."
            ) from exc
        # Resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile.
        self._client = anthropic.Anthropic(api_key=kwargs.get("api_key"))
        # `summarized` so we get visible reasoning text to scan for suspicion.
        self._thinking = kwargs.get("thinking", {"type": "adaptive", "display": "summarized"})
        self._effort = kwargs.get("effort", "high")

    # -- request assembly ---------------------------------------------------

    def _render_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate normalized messages into Anthropic message dicts.

        Assistant turns that carry `_raw` (native content, incl. signed thinking
        blocks) are replayed verbatim — required when interleaving thinking with
        tool use across turns.
        """
        rendered: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("_raw") is not None:
                rendered.append({"role": "assistant", "content": msg["_raw"]})
                continue
            rendered.append({"role": msg["role"], "content": msg["content"]})
        return rendered

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> AssistantTurn:
        request: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._render_messages(messages),
            "thinking": self._thinking,
            "output_config": {"effort": self._effort},
        }
        if tools:
            request["tools"] = [t.to_anthropic() for t in tools]

        # Stream and accumulate; get_final_message gives the complete object.
        with self._client.messages.stream(**request) as stream:
            message = stream.get_final_message()

        return self._parse(message)

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def _parse(message: Any) -> AssistantTurn:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "thinking":
                # `display: summarized` populates this; may be empty otherwise.
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        usage = {}
        if getattr(message, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(message.usage, "input_tokens", 0),
                "output_tokens": getattr(message.usage, "output_tokens", 0),
            }

        return AssistantTurn(
            text="\n".join(p for p in text_parts if p),
            tool_calls=tool_calls,
            stop_reason=getattr(message, "stop_reason", "end_turn") or "end_turn",
            thinking="\n".join(p for p in thinking_parts if p) or None,
            raw_content=message.content,  # replay verbatim next turn
            usage=usage,
        )
