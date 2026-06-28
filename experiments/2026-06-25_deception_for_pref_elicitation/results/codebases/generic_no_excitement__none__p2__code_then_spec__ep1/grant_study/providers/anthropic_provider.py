"""Anthropic implementation of the Provider interface.

Uses the official ``anthropic`` SDK and a manual agentic loop (the loop itself
lives in ``grant_study.episode``; this class just performs one ``generate`` and
manages native message state). A manual loop — rather than the SDK tool runner —
is deliberate: the study needs to intercept, gate, and record every tool call
(especially ``finalize_allocation`` and ``request_disbursement``).
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..config import ModelSpec
from .base import ModelResponse, Provider, ToolCall, ToolResult, ToolSpec


class AnthropicProvider(Provider):
    def __init__(self, spec: ModelSpec, client: anthropic.Anthropic | None = None):
        super().__init__(spec)
        # Resolves credentials from the environment (ANTHROPIC_API_KEY, etc.).
        self.client = client or anthropic.Anthropic()
        self._messages: list[dict[str, Any]] = []   # native, includes thinking blocks
        self._log: list[dict[str, Any]] = []         # serializable mirror

    # -- history mutation ---------------------------------------------------- #
    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._log.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        content = []
        for r in results:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": r.tool_use_id,
                "content": r.content,
            }
            if r.is_error:
                block["is_error"] = True
            content.append(block)
        self._messages.append({"role": "user", "content": content})
        self._log.append({
            "role": "tool_results",
            "results": [
                {"tool_use_id": r.tool_use_id, "content": r.content, "is_error": r.is_error}
                for r in results
            ],
        })

    # -- generation ---------------------------------------------------------- #
    def generate(self, system: str, tools: list[ToolSpec]) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.spec.model_id,
            "max_tokens": self.spec.max_tokens,
            "system": system,
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
            "messages": self._messages,
        }
        if self.spec.thinking:
            # Capture summarized reasoning — it is a primary object of study.
            thinking_cfg: dict[str, Any] = {"type": "adaptive"}
            if self.spec.thinking_display:
                thinking_cfg["display"] = self.spec.thinking_display
            kwargs["thinking"] = thinking_cfg
        if self.spec.effort:
            kwargs["output_config"] = {"effort": self.spec.effort}

        resp = self.client.messages.create(**kwargs)

        # Preserve the assistant turn verbatim (thinking blocks + signatures) so
        # subsequent turns validate.
        self._messages.append({"role": "assistant", "content": resp.content})

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", None),
            "output_tokens": getattr(resp.usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", None),
        }

        self._log.append({
            "role": "assistant",
            "thinking": "\n".join(p for p in thinking_parts if p),
            "text": "\n".join(text_parts),
            "tool_calls": [{"id": c.id, "name": c.name, "input": c.input} for c in tool_calls],
            "stop_reason": resp.stop_reason,
            "usage": usage,
        })

        return ModelResponse(
            text="\n".join(text_parts),
            thinking="\n".join(p for p in thinking_parts if p),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            usage=usage,
        )

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return self._log
