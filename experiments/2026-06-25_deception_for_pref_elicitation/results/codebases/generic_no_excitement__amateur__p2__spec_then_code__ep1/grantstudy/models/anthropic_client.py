"""Anthropic implementation of ModelClient.

Uses a manual agentic loop (not the SDK tool-runner) so that every tool call is
routed through the runner's human-in-the-loop dispatch and recorded in the
transcript. Tool inputs are consumed from the SDK's parsed ``block.input`` and
never raw-string-matched (the 4.x models may vary JSON string escaping).

Per current SDK guidance: adaptive thinking + an effort setting, no sampling
parameters, no last-assistant-turn prefills.
"""

from __future__ import annotations

from typing import Any

from .base import (
    Dispatch,
    EpisodeResult,
    ModelClient,
    ToolCall,
    ToolResult,
    ToolSpec,
    TranscriptEvent,
)

# max_tokens kept under the non-streaming SDK timeout guard (~16k); an agentic
# decision turn does not need more headroom than this.
_MAX_TOKENS = 16000
_VALID_EFFORT = {"low", "medium", "high", "xhigh", "max"}


class AnthropicModelClient(ModelClient):
    def __init__(self, *, label: str, model: str, effort: str = "high") -> None:
        # Imported lazily so the rest of the harness (config validation, report)
        # works without the anthropic package installed.
        import anthropic

        self.label = label
        self.model = model
        self.effort = effort if effort in _VALID_EFFORT else "high"
        self._client = anthropic.Anthropic()

    def _to_sdk_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def run_episode(
        self,
        *,
        system_prompt: str,
        opening_user_message: str,
        tools: list[ToolSpec],
        dispatch: Dispatch,
        max_turns: int,
    ) -> EpisodeResult:
        sdk_tools = self._to_sdk_tools(tools)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": opening_user_message}
        ]
        transcript: list[TranscriptEvent] = []
        result = EpisodeResult(label=self.label, final_text="")

        try:
            for _turn in range(max_turns):
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=_MAX_TOKENS,
                    system=system_prompt,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                    tools=sdk_tools,
                    messages=messages,
                )
                result.stop_reason = response.stop_reason or ""
                self._accumulate_usage(result, response)

                # Record model-visible content.
                final_text_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                for block in response.content:
                    if block.type == "text":
                        final_text_parts.append(block.text)
                        transcript.append(TranscriptEvent(role="model", text=block.text))
                    elif block.type == "thinking":
                        transcript.append(
                            TranscriptEvent(role="thinking", text=getattr(block, "thinking", ""))
                        )
                    elif block.type == "tool_use":
                        call = ToolCall(id=block.id, name=block.name, input=dict(block.input))
                        tool_calls.append(call)
                        transcript.append(TranscriptEvent(role="model", tool_call=call))

                if final_text_parts:
                    result.final_text = "\n".join(p for p in final_text_parts if p)

                # Server resumed-loop pause: re-send and continue.
                if response.stop_reason == "pause_turn":
                    messages.append({"role": "assistant", "content": response.content})
                    continue

                if response.stop_reason != "tool_use":
                    # end_turn, max_tokens, refusal, stop_sequence -> episode over.
                    break

                # Execute each tool call through the human-in-the-loop dispatch.
                messages.append({"role": "assistant", "content": response.content})
                tool_results_content: list[dict[str, Any]] = []
                for call in tool_calls:
                    tr: ToolResult = dispatch(call)
                    transcript.append(TranscriptEvent(role="tool", tool_result=tr))
                    tool_results_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.content,
                            "is_error": tr.is_error,
                        }
                    )
                messages.append({"role": "user", "content": tool_results_content})
            else:
                result.stop_reason = "max_turns_exhausted"
        except Exception as exc:  # noqa: BLE001 - surface provider errors per-run
            result.error = f"{type(exc).__name__}: {exc}"

        result.transcript = transcript
        return result

    def ask_followup(self, *, conversation_note: str, questions: str) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2000,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": f"{conversation_note}\n\n{questions}"}],
            )
            return "\n".join(b.text for b in response.content if b.type == "text")
        except Exception as exc:  # noqa: BLE001
            return f"[follow-up failed: {type(exc).__name__}: {exc}]"

    @staticmethod
    def _accumulate_usage(result: EpisodeResult, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for field_name in ("input_tokens", "output_tokens",
                            "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = getattr(usage, field_name, None)
            if value is not None:
                result.usage[field_name] = result.usage.get(field_name, 0) + value
