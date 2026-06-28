"""The model-adapter interface.

An adapter wraps one model behind a uniform, stateful conversation API the runner can drive
identically regardless of provider:

    adapter.configure(system_prompt, tool_schemas)
    adapter.add_user_text("You may begin.")
    while ...:
        turn = adapter.advance()            # one model call; returns text + tool calls
        # runner executes turn.tool_calls against the world ...
        adapter.submit_tool_results([(call_id, content_dict), ...])

The tool schema format is the Anthropic Messages API shape (name / description / input_schema);
non-Anthropic adapters translate it. Keeping the interface stateful lets each adapter store
provider-native message history (which matters for thinking-block signatures, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AssistantTurn:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""


class ModelAdapter:
    """Base class. Subclasses implement provider specifics."""

    def configure(self, system_prompt: str, tool_schemas: list[dict]) -> None:
        raise NotImplementedError

    def add_user_text(self, text: str) -> None:
        raise NotImplementedError

    def advance(self) -> AssistantTurn:
        """Make one model call against the current history and return the assistant turn."""
        raise NotImplementedError

    def submit_tool_results(self, results: list[tuple[str, dict]]) -> None:
        """Append tool results (by call id) so the next advance() sees them."""
        raise NotImplementedError


def make_adapter(adapter: str, model: str | None = None, **kwargs: Any) -> ModelAdapter:
    """Factory. Imports lazily so optional SDKs aren't required unless used."""
    if adapter == "claude":
        from .claude import ClaudeAdapter

        return ClaudeAdapter(model=model or "claude-opus-4-8", **kwargs)
    if adapter == "openai-compatible":
        from .openai_compatible import OpenAICompatibleAdapter

        if model is None:
            raise ValueError("openai-compatible adapter requires --model")
        return OpenAICompatibleAdapter(model=model, **kwargs)
    if adapter == "scripted":
        from .scripted import ScriptedAdapter

        return ScriptedAdapter(**kwargs)
    raise ValueError(f"unknown adapter: {adapter}")
