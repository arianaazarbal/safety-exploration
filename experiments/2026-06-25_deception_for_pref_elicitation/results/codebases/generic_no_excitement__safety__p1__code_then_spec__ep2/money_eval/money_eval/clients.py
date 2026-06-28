"""Provider clients behind a uniform, provider-agnostic interface.

The runner drives the agentic loop in terms of three abstractions:

  - `ModelTurn`  : a normalized view of one model response (text + tool calls + raw content)
  - `ModelClient`: per-provider implementation that owns its native message format

History is a provider-native list that the runner treats as opaque, mutating it only through
the client's append_* helpers. This keeps runner.py free of provider specifics while letting
each client speak its own SDK's dialect.

The Anthropic/Claude client is fully implemented. Other providers are stubs with the method
surface marked out — fill them in against their SDKs (the OpenAI/Gemini tool-call and
message shapes differ, but the normalized `ModelTurn` they must return is the same).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    # Provider-native assistant content, to be appended back into history verbatim.
    raw_assistant: Any = None


class ModelClient(Protocol):
    name: str
    model_id: str

    def new_history(self) -> list: ...

    def append_user(self, history: list, text: str) -> None: ...

    def append_assistant(self, history: list, turn: ModelTurn) -> None: ...

    def append_tool_results(self, history: list, results: list[tuple[str, str]]) -> None: ...

    def step(self, system: str, history: list, tools: list[dict], max_tokens: int) -> ModelTurn: ...

    def text_only(self, system: str, user: str, max_tokens: int = 4000) -> str: ...


# ---------------------------------------------------------------------------
# Anthropic / Claude
# ---------------------------------------------------------------------------


@dataclass
class ClaudeClient:
    """Claude client using the Anthropic Messages API (manual agentic loop)."""

    model_id: str
    name: str = "claude"
    # Effort level for output_config. None omits it entirely — required for models that
    # don't support the effort parameter (e.g. Haiku 4.5, which 400s if it's sent).
    effort: str | None = "high"
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import anthropic  # imported lazily so the package imports without the SDK present

        self._client = anthropic.Anthropic()

    def new_history(self) -> list:
        return []

    def append_user(self, history: list, text: str) -> None:
        history.append({"role": "user", "content": text})

    def append_assistant(self, history: list, turn: ModelTurn) -> None:
        # raw_assistant is the response.content list from the API.
        history.append({"role": "assistant", "content": turn.raw_assistant})

    def append_tool_results(self, history: list, results: list[tuple[str, str]]) -> None:
        history.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
                    for tool_use_id, text in results
                ],
            }
        )

    def step(self, system: str, history: list, tools: list[dict], max_tokens: int) -> ModelTurn:
        # Stream to stay clear of HTTP timeouts on larger max_tokens, then collect
        # the complete message for tool-call handling.
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "thinking": {"type": "adaptive"},
            "system": system,
            "tools": tools,
            "messages": history,
        }
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        with self._client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return ModelTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
            raw_assistant=response.content,
        )

    def text_only(self, system: str, user: str, max_tokens: int = 4000) -> str:
        with self._client.messages.stream(
            model=self.model_id,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            response = stream.get_final_message()
        return "\n".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Stubs for other providers.
#
# To add a provider, implement the ModelClient surface against its SDK. The only
# contract that matters is: step() must return a normalized ModelTurn, and the
# append_* helpers must keep that provider's native history valid (tool-call ids
# must round-trip between assistant tool calls and the tool results you send back).
# ---------------------------------------------------------------------------


@dataclass
class OpenAIClient:
    model_id: str
    name: str = "openai"

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "OpenAIClient is a stub. Implement against the OpenAI SDK: map tool calls "
            "(message.tool_calls) into ModelTurn.tool_calls, append assistant + 'tool' "
            "role results, and translate TOOL_SCHEMAS into the function-tool format."
        )


@dataclass
class GeminiClient:
    model_id: str
    name: str = "gemini"

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "GeminiClient is a stub. Implement against the google-genai SDK: map "
            "functionCall parts into ModelTurn.tool_calls and functionResponse parts back."
        )
