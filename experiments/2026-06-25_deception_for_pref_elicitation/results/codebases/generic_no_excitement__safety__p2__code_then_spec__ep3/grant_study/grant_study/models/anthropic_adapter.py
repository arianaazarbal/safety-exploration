"""Anthropic (Claude) adapter.

Uses adaptive thinking and the effort parameter, per current Claude API guidance. The runner
owns the agentic loop; this adapter only produces one assistant turn at a time.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn


class AnthropicAdapter:
    def __init__(self, model_id: str = "claude-opus-4-8", client: Any | None = None) -> None:
        # Imported lazily so the package imports without the SDK installed.
        import anthropic

        self._model_id = model_id
        self._client = client or anthropic.Anthropic()

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 16_000,
        effort: str = "high",
    ) -> AssistantTurn:
        api_messages = [self._to_api_message(m) for m in messages]

        response = self._client.messages.create(
            model=self._model_id,
            max_tokens=max_tokens,
            system=system,
            messages=api_messages,
            tools=[
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"],
                }
                for t in tools
            ],
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": effort},
        )
        return self._from_api_response(response)

    @staticmethod
    def _to_api_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Normalized message -> Anthropic message. Content blocks map almost 1:1."""
        out_content: list[dict[str, Any]] = []
        for block in msg["content"]:
            btype = block.get("type")
            if btype == "text":
                out_content.append({"type": "text", "text": block["text"]})
            elif btype == "thinking":
                # Thinking blocks are echoed back with their signature preserved when present.
                tb: dict[str, Any] = {"type": "thinking", "thinking": block["thinking"]}
                if "signature" in block:
                    tb["signature"] = block["signature"]
                out_content.append(tb)
            elif btype == "tool_use":
                out_content.append(
                    {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": block["input"],
                    }
                )
            elif btype == "tool_result":
                out_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["tool_use_id"],
                        "content": block["content"],
                        "is_error": block.get("is_error", False),
                    }
                )
            else:
                raise ValueError(f"Unknown content block type {btype!r}")
        return {"role": msg["role"], "content": out_content}

    @staticmethod
    def _from_api_response(response: Any) -> AssistantTurn:
        blocks: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "thinking":
                tb: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
                sig = getattr(block, "signature", None)
                if sig:
                    tb["signature"] = sig
                blocks.append(tb)
            elif block.type == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            # Other block types (server tool results, etc.) are not used by this harness.
        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        return AssistantTurn(
            blocks=blocks, stop_reason=response.stop_reason, raw_usage=usage
        )
