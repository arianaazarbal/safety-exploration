"""Anthropic (Claude) adapter — the reference implementation.

Uses the Anthropic SDK with adaptive thinking, configurable effort, and streaming with
get_final_message() (tool loops and large outputs require streaming to avoid HTTP
timeouts). A manual agentic loop is used by the runner, so this adapter exposes one model
turn at a time rather than running the loop itself.
"""

from __future__ import annotations

from .base import ModelAdapter, AssistantTurn, ToolCall, Usage

# Default model per the current Anthropic model lineup.
_DEFAULT_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 32000  # generous headroom; streaming is used so timeouts are not a concern


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model: str = _DEFAULT_MODEL, effort: str = "high", **kwargs):
        super().__init__(model=model, effort=effort, **kwargs)
        import anthropic  # imported lazily so the package loads without every provider SDK
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._system: str = ""
        self._messages: list[dict] = []
        self.effort = effort

    # ----- conversation state ---------------------------------------------

    def reset(self, system: str) -> None:
        self._system = system
        self._messages = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result,
                "is_error": is_error,
            }
            for (tool_call_id, result, is_error) in results
        ]
        self._messages.append({"role": "user", "content": content})

    def step(self, tool_specs: list[dict]) -> AssistantTurn:
        kwargs = dict(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=self._system,
            messages=self._messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if tool_specs:
            kwargs["tools"] = tool_specs

        # Stream and collect the final message (timeout-safe for long tool turns).
        with self._client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

        # Append the assistant turn verbatim (preserves thinking blocks + signatures).
        self._messages.append({"role": "assistant", "content": msg.content})

        text_parts, tool_calls = [], []
        for block in msg.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input or {}))

        usage = Usage(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
        self.usage.add(usage)
        return AssistantTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=msg.stop_reason or "end_turn",
            usage=usage,
        )

    # ----- stateless one-off ----------------------------------------------

    def simple_complete(self, system: str, user: str) -> str:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        ) as stream:
            msg = stream.get_final_message()
        self.usage.add(Usage(msg.usage.input_tokens, msg.usage.output_tokens))
        return "".join(b.text for b in msg.content if b.type == "text").strip()
