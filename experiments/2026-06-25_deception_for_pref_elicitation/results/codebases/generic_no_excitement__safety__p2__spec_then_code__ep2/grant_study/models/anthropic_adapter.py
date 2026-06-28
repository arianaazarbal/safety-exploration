"""Anthropic adapter — the authoritative model path.

Uses the official `anthropic` SDK with a manual agentic loop so the environment's
approval gate can intercept each tool call (DESIGN.md §4). Defaults to
`claude-opus-4-8` with adaptive thinking and high effort.

Manual loop (not the tool runner) is deliberate: the harness — not the SDK —
must own tool execution, because every call routes through Environment.execute()
where the safety chain lives. We never let the SDK auto-execute tools.
"""

from __future__ import annotations

from typing import Any

from .base import Conversation, ModelAdapter, ModelResponse, ToolCall

# Non-streaming max_tokens that stays under SDK HTTP timeouts (per claude-api skill).
_MAX_TOKENS = 16000


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model_id: str = "claude-opus-4-8") -> None:
        super().__init__(model_id)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise ImportError(
                "The 'anthropic' package is required for AnthropicAdapter. "
                "Install with: pip install anthropic"
            ) from exc
        # Resolves credentials from the environment (ANTHROPIC_API_KEY or an
        # `ant auth login` profile). Do not hardcode a key.
        self._client = anthropic.Anthropic()

    def start(self, system_prompt: str, user_prompt: str) -> Conversation:
        return Conversation(
            system=system_prompt,
            state=[{"role": "user", "content": user_prompt}],
        )

    def _to_native_tools(self, tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["parameters"],
            }
            for spec in tool_specs
        ]

    def run_turn(
        self, conversation: Conversation, tool_specs: list[dict[str, Any]]
    ) -> ModelResponse:
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=conversation.system,
            tools=self._to_native_tools(tool_specs),
            messages=conversation.state,
        )

        # Preserve the full assistant content (including thinking blocks with
        # their signatures) so multi-turn tool use stays valid.
        conversation.state.append({"role": "assistant", "content": resp.content})

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ModelResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            raw=resp,
        )

    def add_tool_results(
        self, conversation: Conversation, results: list[tuple[ToolCall, str]]
    ) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result_text,
            }
            for call, result_text in results
        ]
        conversation.state.append({"role": "user", "content": content})
